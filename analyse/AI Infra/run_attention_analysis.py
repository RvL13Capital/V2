"""
Complete Example: LSTM Attention Model Training and Analysis
Demonstrates full workflow from data loading to attention visualization
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import sys

# Import our attention modules
from attention_lstm_model import LSTMAttentionModel, prepare_sequences
from attention_visualizer import AttentionVisualizer
from attention_prediction_analyzer import PredictionAnalyzer, batch_analyze_predictions

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_data(n_patterns: int = 5000,
                           window_size: int = 30,
                           n_features: int = 20) -> pd.DataFrame:
    """
    Generate synthetic time-series data for testing

    In production, replace with actual pattern data loading
    """
    logger.info(f"Generating {n_patterns} synthetic patterns...")

    np.random.seed(42)

    data = []

    for i in range(n_patterns):
        symbol = f"TICK{i % 100:03d}"

        # Generate time series for this pattern
        for day in range(window_size + 1):  # +1 for label day
            timestamp = datetime(2020, 1, 1) + pd.Timedelta(days=i*window_size + day)

            # Generate features (volume patterns, technical indicators)
            features = {
                'symbol': symbol,
                'timestamp': timestamp,
            }

            # Volume features (key for pattern detection)
            for period in [3, 5, 10]:
                features[f'vol_strength_{period}d'] = np.random.random() * 2
                features[f'vol_ratio_{period}d'] = np.random.random() * 3
                features[f'accum_score_{period}d'] = np.random.random() * 100

            # Technical features
            features['bbw'] = np.random.random() * 30
            features['adx'] = np.random.random() * 50
            features['rsi'] = 30 + np.random.random() * 40
            features['obv_trend'] = np.random.randn()

            # Consecutive patterns
            features['consec_vol_up'] = np.random.randint(0, 6)
            features['sequence_strength'] = np.random.random()

            data.append(features)

    df = pd.DataFrame(data)

    # Add winner labels (based on patterns in last few days)
    df['is_winner'] = 0

    # Winners if high volume strength in last 5 days
    for symbol in df['symbol'].unique():
        symbol_data = df[df['symbol'] == symbol].copy()
        if len(symbol_data) > 5:
            last_5_vol = symbol_data.iloc[-6:-1]['vol_strength_5d'].mean()
            if last_5_vol > 1.3:  # High volume accumulation
                df.loc[df['symbol'] == symbol, 'is_winner'] = 1

    logger.info(f"Generated {len(df)} data points")
    logger.info(f"Winner rate: {df['is_winner'].mean():.1%}")

    return df


def main():
    """Main execution function"""

    logger.info("="*100)
    logger.info("LSTM ATTENTION MODEL - TRAINING AND ANALYSIS")
    logger.info("="*100)
    logger.info("")

    # Configuration
    WINDOW_SIZE = 30
    USE_SYNTHETIC_DATA = True  # Set to False to use real data
    DATA_PATH = r'C:\Users\Pfenn\OneDrive\Desktop\nothing-main\analyse\AI Infra\data\raw\60day_constrained_patterns.parquet'
    OUTPUT_DIR = 'output/attention_analysis'
    MODEL_PATH = 'models/lstm_attention_model.keras'

    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Load Data
    logger.info("Step 1/6: Loading Data...")

    if USE_SYNTHETIC_DATA or not Path(DATA_PATH).exists():
        logger.info("Using synthetic data for demonstration")
        df = generate_synthetic_data(n_patterns=2000, window_size=WINDOW_SIZE)
    else:
        logger.info(f"Loading real data from {DATA_PATH}")
        df = pd.read_parquet(DATA_PATH)

    logger.info(f"Loaded {len(df)} data points from {df['symbol'].nunique() if 'symbol' in df.columns else 'N/A'} symbols")

    # Step 2: Prepare Sequences
    logger.info("\nStep 2/6: Preparing Sequences...")

    # Get feature columns
    exclude_cols = ['symbol', 'timestamp', 'is_winner']
    feature_columns = [col for col in df.columns if col not in exclude_cols]

    logger.info(f"Using {len(feature_columns)} features")
    logger.info(f"Features: {', '.join(feature_columns[:5])}...")

    X, y, dates = prepare_sequences(
        df,
        window_size=WINDOW_SIZE,
        feature_columns=feature_columns
    )

    # Split data
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
        X, y, dates,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    logger.info(f"Train: {len(X_train)} sequences")
    logger.info(f"Test: {len(X_test)} sequences")

    # Step 3: Build and Train Model
    logger.info("\nStep 3/6: Building and Training Model...")

    model = LSTMAttentionModel(
        window_size=WINDOW_SIZE,
        n_features=len(feature_columns),
        lstm_units=64,
        dropout_rate=0.3,
        learning_rate=0.001
    )

    # Build model
    model.build_model()
    logger.info("\nModel Architecture:")
    print(model.get_model_summary())

    # Train model
    logger.info("\nTraining model...")
    history = model.train(
        X_train, y_train,
        X_test, y_test,
        epochs=30,
        batch_size=32,
        verbose=1
    )

    # Save model
    model_path = Path('models')
    model_path.mkdir(exist_ok=True)
    model_file = model_path / 'lstm_attention_model.keras'
    model.save_model(str(model_file))
    logger.info(f"\nModel saved to {model_file}")

    # Step 4: Evaluate Model
    logger.info("\nStep 4/6: Evaluating Model...")

    predictions = model.predict(X_test)
    pred_classes = (predictions > 0.5).astype(int)

    from sklearn.metrics import classification_report, confusion_matrix

    logger.info("\nClassification Report:")
    print(classification_report(y_test, pred_classes, target_names=['NOT_WINNER', 'WINNER']))

    logger.info("\nConfusion Matrix:")
    print(confusion_matrix(y_test, pred_classes))

    # Step 5: Analyze Attention Patterns
    logger.info("\nStep 5/6: Analyzing Attention Patterns...")

    # Create visualizer and analyzer
    visualizer = AttentionVisualizer()
    analyzer = PredictionAnalyzer(model, visualizer)

    # Analyze a few interesting cases
    interesting_cases = []

    # Find true positives
    tp_idx = np.where((pred_classes == 1) & (y_test == 1))[0]
    if len(tp_idx) > 0:
        interesting_cases.append(('True Positive', tp_idx[0]))

    # Find true negatives
    tn_idx = np.where((pred_classes == 0) & (y_test == 0))[0]
    if len(tn_idx) > 0:
        interesting_cases.append(('True Negative', tn_idx[0]))

    # Find false positives
    fp_idx = np.where((pred_classes == 1) & (y_test == 0))[0]
    if len(fp_idx) > 0:
        interesting_cases.append(('False Positive', fp_idx[0]))

    # Find false negatives
    fn_idx = np.where((pred_classes == 0) & (y_test == 1))[0]
    if len(fn_idx) > 0:
        interesting_cases.append(('False Negative', fn_idx[0]))

    # Analyze each case
    for case_type, idx in interesting_cases:
        logger.info(f"\nAnalyzing {case_type} example...")

        sequence = X_test[idx]
        label = y_test[idx]
        sequence_dates = dates_test[idx]

        # Analyze
        analysis = analyzer.analyze_prediction(
            sequence,
            feature_names=feature_columns,
            dates=sequence_dates,
            actual_label=label,
            ticker=f"Example_{case_type.replace(' ', '_')}"
        )

        # Generate text report
        report = analyzer.generate_text_report(analysis)
        print("\n" + report)

        # Save report
        report_file = output_path / f"analysis_{case_type.replace(' ', '_')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)

        # Create visualizations
        case_dir = output_path / case_type.replace(' ', '_')
        analyzer.visualize_prediction(
            sequence,
            feature_names=feature_columns,
            dates=sequence_dates,
            actual_label=label,
            ticker=case_type,
            save_dir=str(case_dir),
            show=False  # Don't show, just save
        )

        logger.info(f"✓ Analysis saved to {case_dir}")

    # Step 6: Attention Pattern Analysis
    logger.info("\nStep 6/6: Attention Pattern Analysis...")

    # Get attention weights for all test samples
    all_attentions = model.get_attention_weights(X_test)

    # Visualize attention distribution
    fig = visualizer.plot_attention_distribution(
        all_attentions,
        title="Attention Distribution Across All Test Samples",
        save_path=str(output_path / "attention_distribution.png"),
        show=False
    )

    # Compare attention patterns between winners and losers
    winner_attention = all_attentions[y_test == 1]
    loser_attention = all_attentions[y_test == 0]

    if len(winner_attention) > 0 and len(loser_attention) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        # Winners
        axes[0].plot(winner_attention.mean(axis=0), 'g-', linewidth=2, label='Mean')
        axes[0].fill_between(
            range(WINDOW_SIZE),
            winner_attention.mean(axis=0) - winner_attention.std(axis=0),
            winner_attention.mean(axis=0) + winner_attention.std(axis=0),
            alpha=0.3
        )
        axes[0].set_title('Attention Pattern - Winners', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Day in Sequence')
        axes[0].set_ylabel('Attention Weight')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Losers
        axes[1].plot(loser_attention.mean(axis=0), 'r-', linewidth=2, label='Mean')
        axes[1].fill_between(
            range(WINDOW_SIZE),
            loser_attention.mean(axis=0) - loser_attention.std(axis=0),
            loser_attention.mean(axis=0) + loser_attention.std(axis=0),
            alpha=0.3
        )
        axes[1].set_title('Attention Pattern - Losers', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Day in Sequence')
        axes[1].set_ylabel('Attention Weight')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.tight_layout()
        comparison_path = output_path / "attention_winner_vs_loser.png"
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        logger.info(f"✓ Comparison plot saved to {comparison_path}")
        plt.close()

    # Final Summary
    logger.info("\n" + "="*100)
    logger.info("ANALYSIS COMPLETE!")
    logger.info("="*100)
    logger.info(f"\nModel Performance:")
    logger.info(f"  • Test Accuracy: {(pred_classes == y_test).mean():.1%}")
    logger.info(f"  • Winner Recall: {(pred_classes[y_test == 1] == 1).mean():.1%}")
    logger.info(f"  • Winner Precision: {(y_test[pred_classes == 1] == 1).mean():.1%}")

    logger.info(f"\nAttention Insights:")
    logger.info(f"  • Mean attention weight: {all_attentions.mean():.5f}")
    logger.info(f"  • Max attention weight: {all_attentions.max():.5f}")
    logger.info(f"  • Attention concentration: {all_attentions.max() / all_attentions.mean():.2f}x")

    # Identify most important days on average
    mean_attention = all_attentions.mean(axis=0)
    top_days = np.argsort(mean_attention)[-5:][::-1]
    logger.info(f"\n  • Top 5 most important days (on average):")
    for i, day in enumerate(top_days, 1):
        logger.info(f"    {i}. Day {day} (weight: {mean_attention[day]:.5f})")

    logger.info(f"\nAll results saved to: {output_path.absolute()}")
    logger.info("="*100)

    return model, analyzer, history


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt

    try:
        model, analyzer, history = main()
        print("\n✓ SUCCESS! Check output/attention_analysis/ for results")
    except Exception as e:
        logger.error(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
