"""
Interactive Prediction Analyzer
Analyze individual predictions with attention visualization
Shows which days the model focused on and why
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

from attention_lstm_model import LSTMAttentionModel, prepare_sequences
from attention_visualizer import AttentionVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionAnalyzer:
    """
    Analyze individual predictions with attention weights

    Provides insights into:
    - Which days the model focused on
    - What features were important on those days
    - Why the model made its prediction
    """

    def __init__(self, model: LSTMAttentionModel, visualizer: Optional[AttentionVisualizer] = None):
        """
        Initialize analyzer

        Args:
            model: Trained LSTM Attention model
            visualizer: Attention visualizer (creates new if None)
        """
        self.model = model
        self.visualizer = visualizer or AttentionVisualizer()

    def analyze_prediction(self,
                          sequence: np.ndarray,
                          feature_names: List[str],
                          dates: Optional[List] = None,
                          actual_label: Optional[int] = None,
                          ticker: Optional[str] = None) -> Dict:
        """
        Comprehensive analysis of a single prediction

        Args:
            sequence: Input sequence (window_size, n_features)
            feature_names: List of feature names
            dates: Optional list of dates for sequence
            actual_label: Optional actual label
            ticker: Optional ticker symbol

        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Analyzing prediction{f' for {ticker}' if ticker else ''}...")

        # Get model analysis
        model_analysis = self.model.analyze_prediction(sequence, dates)

        # Extract attention weights and prediction
        attention = np.array(model_analysis['attention_weights'])
        prediction = model_analysis['prediction']

        # Identify key days
        top_k = 5
        top_indices = np.argsort(attention)[-top_k:][::-1]

        # Analyze features on top attention days
        top_day_features = []
        for idx in top_indices:
            day_features = sequence[idx]
            # Get top 3 features for this day
            top_feat_idx = np.argsort(np.abs(day_features))[-3:][::-1]

            top_day_features.append({
                'day_index': int(idx),
                'date': dates[idx] if dates else f"Day {idx}",
                'attention_weight': float(attention[idx]),
                'top_features': [
                    {
                        'name': feature_names[i],
                        'value': float(day_features[i])
                    }
                    for i in top_feat_idx
                ]
            })

        # Pattern detection
        attention_pattern = self._detect_attention_pattern(attention)

        # Feature importance across sequence
        feature_importance = self._calculate_feature_importance(sequence, attention, feature_names)

        # Compile analysis
        analysis = {
            'model_analysis': model_analysis,
            'ticker': ticker,
            'prediction': {
                'probability': float(prediction),
                'class': 'WINNER' if prediction > 0.5 else 'NOT_WINNER',
                'confidence': float(abs(prediction - 0.5) * 2 * 100)
            },
            'actual': {
                'class': 'WINNER' if actual_label == 1 else 'NOT_WINNER' if actual_label == 0 else 'UNKNOWN',
                'correct': bool(prediction > 0.5) == bool(actual_label) if actual_label is not None else None
            },
            'attention_analysis': {
                'pattern': attention_pattern,
                'mean_weight': float(attention.mean()),
                'std_weight': float(attention.std()),
                'max_weight': float(attention.max()),
                'concentration': float(attention.max() / attention.mean())  # How focused
            },
            'top_attention_days': top_day_features,
            'feature_importance': feature_importance[:10]  # Top 10 features
        }

        return analysis

    def _detect_attention_pattern(self, attention: np.ndarray) -> str:
        """
        Detect the pattern of attention across time

        Returns:
            Pattern description (e.g., 'early_focused', 'late_focused', 'distributed')
        """
        third = len(attention) // 3

        # Calculate attention concentration in different periods
        early_attention = attention[:third].sum() / attention.sum()
        mid_attention = attention[third:2*third].sum() / attention.sum()
        late_attention = attention[2*third:].sum() / attention.sum()

        # Determine pattern
        if early_attention > 0.5:
            return 'early_focused'
        elif late_attention > 0.5:
            return 'late_focused'
        elif mid_attention > 0.5:
            return 'mid_focused'
        else:
            # Check if distributed evenly
            if max(early_attention, mid_attention, late_attention) < 0.4:
                return 'evenly_distributed'
            else:
                return 'mixed'

    def _calculate_feature_importance(self,
                                     sequence: np.ndarray,
                                     attention: np.ndarray,
                                     feature_names: List[str]) -> List[Dict]:
        """
        Calculate importance of each feature weighted by attention

        Args:
            sequence: Feature sequence (window_size, n_features)
            attention: Attention weights (window_size,)
            feature_names: Feature names

        Returns:
            List of feature importance dictionaries
        """
        # Weight features by attention
        weighted_features = sequence * attention[:, np.newaxis]

        # Calculate importance as weighted mean absolute value
        importance = np.abs(weighted_features).mean(axis=0)

        # Create list of feature importance
        feature_importance = [
            {
                'name': feature_names[i],
                'importance': float(importance[i]),
                'rank': rank + 1
            }
            for rank, i in enumerate(np.argsort(importance)[::-1])
        ]

        return feature_importance

    def visualize_prediction(self,
                            sequence: np.ndarray,
                            feature_names: List[str],
                            dates: Optional[List] = None,
                            actual_label: Optional[int] = None,
                            ticker: Optional[str] = None,
                            save_dir: Optional[str] = None,
                            show: bool = True) -> Dict[str, plt.Figure]:
        """
        Create all visualizations for a prediction

        Args:
            sequence: Input sequence
            feature_names: Feature names
            dates: Optional dates
            actual_label: Optional actual label
            ticker: Optional ticker
            save_dir: Optional directory to save figures
            show: Whether to display figures

        Returns:
            Dictionary of figure objects
        """
        import matplotlib.pyplot as plt

        # Get analysis
        analysis = self.analyze_prediction(sequence, feature_names, dates, actual_label, ticker)

        # Prepare data
        attention = np.array(analysis['model_analysis']['attention_weights'])
        prediction = analysis['prediction']['probability']

        # Create save paths
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            prefix = f"{ticker}_" if ticker else ""
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            prefix = ""
            timestamp = ""

        figures = {}

        # 1. Single attention plot
        title = f"Attention Weights - {ticker}" if ticker else "Attention Weights"
        save_path = save_dir / f"{prefix}attention_{timestamp}.png" if save_dir else None

        fig1 = self.visualizer.plot_single_attention(
            attention,
            dates=dates,
            prediction=prediction,
            actual_label=actual_label,
            title=title,
            save_path=save_path,
            show=show
        )
        figures['attention'] = fig1

        # 2. Attention with features
        title = f"Attention with Features - {ticker}" if ticker else "Attention with Features"
        save_path = save_dir / f"{prefix}attention_features_{timestamp}.png" if save_dir else None

        fig2 = self.visualizer.plot_attention_with_features(
            attention,
            sequence,
            feature_names,
            dates=dates,
            top_k_features=5,
            title=title,
            save_path=save_path,
            show=show
        )
        figures['attention_features'] = fig2

        # 3. Interactive plot (if plotly available)
        try:
            save_path = save_dir / f"{prefix}interactive_{timestamp}.html" if save_dir else None
            fig3 = self.visualizer.create_interactive_attention_plot(
                attention,
                dates=dates,
                prediction=prediction,
                feature_data=sequence,
                feature_names=feature_names,
                title=title,
                save_path=save_path
            )
            if fig3:
                figures['interactive'] = fig3
        except Exception as e:
            logger.warning(f"Could not create interactive plot: {e}")

        return figures

    def generate_text_report(self, analysis: Dict) -> str:
        """
        Generate human-readable text report of analysis

        Args:
            analysis: Analysis dictionary from analyze_prediction()

        Returns:
            Formatted text report
        """
        report = []
        report.append("="*80)
        report.append("PREDICTION ANALYSIS REPORT")
        report.append("="*80)

        if analysis.get('ticker'):
            report.append(f"Ticker: {analysis['ticker']}")

        report.append("")

        # Prediction
        report.append("PREDICTION")
        report.append("-"*80)
        pred = analysis['prediction']
        report.append(f"Class: {pred['class']}")
        report.append(f"Probability: {pred['probability']:.4f}")
        report.append(f"Confidence: {pred['confidence']:.1f}%")

        # Actual (if available)
        actual = analysis['actual']
        if actual['class'] != 'UNKNOWN':
            report.append(f"\nActual Class: {actual['class']}")
            report.append(f"Prediction Correct: {'YES' if actual['correct'] else 'NO'}")

        report.append("")

        # Attention Analysis
        report.append("ATTENTION ANALYSIS")
        report.append("-"*80)
        att = analysis['attention_analysis']
        report.append(f"Pattern: {att['pattern'].replace('_', ' ').title()}")
        report.append(f"Mean Weight: {att['mean_weight']:.5f}")
        report.append(f"Max Weight: {att['max_weight']:.5f}")
        report.append(f"Concentration Factor: {att['concentration']:.2f}x")
        report.append("")

        # Top Attention Days
        report.append("TOP 5 IMPORTANT DAYS")
        report.append("-"*80)

        for i, day in enumerate(analysis['top_attention_days'], 1):
            report.append(f"\n{i}. {day['date']} (Attention: {day['attention_weight']:.4f})")
            report.append("   Top Features:")
            for feat in day['top_features']:
                report.append(f"     • {feat['name']}: {feat['value']:.4f}")

        report.append("")

        # Feature Importance
        report.append("OVERALL FEATURE IMPORTANCE (TOP 10)")
        report.append("-"*80)

        for feat in analysis['feature_importance']:
            report.append(f"{feat['rank']:2d}. {feat['name']:<40} {feat['importance']:.6f}")

        report.append("")
        report.append("="*80)

        return "\n".join(report)

    def compare_predictions(self,
                           sequences: List[np.ndarray],
                           feature_names: List[str],
                           labels: Optional[List[int]] = None,
                           tickers: Optional[List[str]] = None,
                           save_path: Optional[str] = None,
                           show: bool = True):
        """
        Compare attention patterns across multiple predictions

        Args:
            sequences: List of sequences
            feature_names: Feature names
            labels: Optional labels
            tickers: Optional ticker symbols
            save_path: Path to save comparison plot
            show: Whether to display plot
        """
        # Get predictions and attention for all sequences
        predictions = []
        attentions = []

        for seq in sequences:
            if len(seq.shape) == 2:
                seq = seq.reshape(1, *seq.shape)
            pred = self.model.predict(seq)[0]
            att = self.model.get_attention_weights(seq)[0]
            predictions.append(pred)
            attentions.append(att)

        attention_matrix = np.array(attentions)
        predictions_array = np.array(predictions)

        # Create comparison visualization
        title = "Attention Pattern Comparison"
        if tickers:
            title += f" ({len(tickers)} tickers)"

        fig = self.visualizer.plot_multiple_attentions(
            attention_matrix,
            n_samples=len(sequences),
            predictions=predictions_array,
            labels=np.array(labels) if labels else None,
            title=title,
            save_path=save_path,
            show=show
        )

        return fig


def batch_analyze_predictions(model_path: str,
                              data_path: str,
                              n_samples: int = 10,
                              output_dir: str = 'output/attention_analysis') -> List[Dict]:
    """
    Batch analyze multiple predictions

    Args:
        model_path: Path to saved model
        data_path: Path to data
        n_samples: Number of samples to analyze
        output_dir: Output directory for results

    Returns:
        List of analysis results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading model from {model_path}")
    model = LSTMAttentionModel()
    model.load_model(model_path)

    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)

    # Prepare sequences
    X, y, dates = prepare_sequences(df)

    # Select random samples
    indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)

    # Create analyzer
    analyzer = PredictionAnalyzer(model)

    # Analyze each sample
    results = []
    for i, idx in enumerate(indices):
        logger.info(f"\nAnalyzing sample {i+1}/{len(indices)}")

        sequence = X[idx]
        label = y[idx]
        sequence_dates = dates[idx]

        # Get ticker if available
        ticker = df.iloc[idx]['symbol'] if 'symbol' in df.columns else None

        # Analyze
        analysis = analyzer.analyze_prediction(
            sequence,
            feature_names=df.columns.tolist(),
            dates=sequence_dates,
            actual_label=label,
            ticker=ticker
        )

        # Generate report
        report = analyzer.generate_text_report(analysis)

        # Save report
        report_path = output_path / f"analysis_{i+1}_{ticker if ticker else 'sample'}.txt"
        with open(report_path, 'w') as f:
            f.write(report)

        # Create visualizations
        analyzer.visualize_prediction(
            sequence,
            feature_names=df.columns.tolist(),
            dates=sequence_dates,
            actual_label=label,
            ticker=ticker,
            save_dir=output_path / f"sample_{i+1}",
            show=False
        )

        results.append(analysis)

    logger.info(f"\n✓ Batch analysis complete! Results saved to {output_path}")

    return results


if __name__ == "__main__":
    logger.info("Attention Prediction Analyzer")
    logger.info("\nFeatures:")
    logger.info("  • Individual prediction analysis")
    logger.info("  • Attention pattern detection")
    logger.info("  • Feature importance analysis")
    logger.info("  • Multiple visualization types")
    logger.info("  • Text report generation")
    logger.info("  • Batch analysis support")
    logger.info("\nReady to analyze predictions!")
