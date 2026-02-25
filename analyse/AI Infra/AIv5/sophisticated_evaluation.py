"""
Sophisticated Pattern Evaluation Framework
==========================================
Evaluates predictions with consistency tracking, flip-flop penalties,
and confidence-based assessment.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import pickle


class SophisticatedEvaluator:
    """
    Advanced evaluation framework that:
    1. Keeps original snapshot labels (100-day outcome from that snapshot)
    2. Tracks pattern's eventual outcome for context
    3. Evaluates prediction consistency across pattern evolution
    4. Penalizes flip-flopping and K4→K5 switches
    5. Allows confidence-based predictions
    """

    def __init__(self):
        # Define outcome hierarchy
        self.outcome_hierarchy = {
            'K4_EXCEPTIONAL': 5,
            'K3_STRONG': 4,
            'K2_QUALITY': 3,
            'K1_MINIMAL': 2,
            'K0_STAGNANT': 1,
            'K5_FAILED': 0
        }

        # Define penalties for prediction switches
        self.switch_penalties = {
            # From K4 predictions
            ('K4', 'K5'): -10.0,  # Worst: predicted exceptional, turned to failure
            ('K4', 'K0'): -3.0,   # Bad: predicted exceptional, went stagnant
            ('K4', 'K1'): -2.5,   # Bad: predicted exceptional, minimal gain
            ('K4', 'K2'): -1.5,   # Moderate: predicted exceptional, only quality
            ('K4', 'K3'): -0.5,   # Minor: still a good outcome

            # From K3 predictions
            ('K3', 'K5'): -5.0,   # Very bad: predicted strong, turned to failure
            ('K3', 'K0'): -2.0,   # Bad: predicted strong, went stagnant
            ('K3', 'K1'): -1.5,   # Bad: predicted strong, minimal gain
            ('K3', 'K2'): -0.5,   # Minor: still positive outcome
            ('K3', 'K4'): 1.0,    # Good: upgraded to exceptional

            # Maintaining predictions (consistency bonus)
            ('K4', 'K4'): 2.0,    # Excellent: maintained K4 conviction
            ('K3', 'K3'): 1.0,    # Good: maintained K3 conviction

            # Improving predictions
            ('K0', 'K4'): 0.5,    # Improved recognition
            ('K1', 'K4'): 0.5,
            ('K2', 'K4'): 0.5,
            ('K0', 'K3'): 0.3,
            ('K1', 'K3'): 0.3,
        }

    def create_dual_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create dual labeling: snapshot_label (eventual_label)
        """
        print("\n" + "="*60)
        print("CREATING DUAL LABELS")
        print("="*60)

        # Create pattern IDs
        df['pattern_id'] = df.apply(
            lambda row: f"{row['ticker']}_{row['upper_boundary']:.4f}_{row['lower_boundary']:.4f}",
            axis=1
        )

        # Find eventual outcome for each pattern
        pattern_outcomes = {}
        for pattern_id, group in df.groupby('pattern_id'):
            # Get the best outcome this pattern achieves
            outcomes = group['outcome_class'].values
            best_outcome = max(outcomes, key=lambda x: self.outcome_hierarchy.get(x, -1))
            pattern_outcomes[pattern_id] = best_outcome

        # Add eventual outcome column
        df['eventual_outcome'] = df['pattern_id'].map(pattern_outcomes)

        # Create dual label
        df['dual_label'] = df['outcome_class'] + ' (' + df['eventual_outcome'] + ')'

        # Print statistics
        print(f"Total snapshots: {len(df)}")
        print(f"Unique patterns: {len(pattern_outcomes)}")

        # Count how many snapshots differ from eventual outcome
        mismatched = df['outcome_class'] != df['eventual_outcome']
        print(f"Snapshots with different eventual outcome: {mismatched.sum()} ({mismatched.sum()/len(df):.1%})")

        return df

    def evaluate_predictions(self, df: pd.DataFrame, predictions: np.ndarray,
                           probabilities: np.ndarray) -> Dict:
        """
        Evaluate predictions with consistency tracking and penalties.
        """
        print("\n" + "="*60)
        print("SOPHISTICATED EVALUATION")
        print("="*60)

        # Map predictions to classes
        class_names = ['K0_STAGNANT', 'K1_MINIMAL', 'K2_QUALITY', 'K3_STRONG', 'K4_EXCEPTIONAL', 'K5_FAILED']

        # Group by pattern
        results = defaultdict(list)
        pattern_scores = {}

        for pattern_id, group in df.groupby('pattern_id'):
            # Sort by snapshot date
            group = group.sort_values('snapshot_date')
            indices = group.index

            # Get predictions and actuals for this pattern
            pattern_preds = [class_names[predictions[i]] for i in indices]
            pattern_actuals = group['outcome_class'].values
            pattern_eventual = group['eventual_outcome'].values[0]
            pattern_probs = probabilities[indices] if probabilities is not None else None

            # Evaluate pattern consistency
            score = self._evaluate_pattern_consistency(
                pattern_preds, pattern_actuals, pattern_eventual, pattern_probs
            )

            pattern_scores[pattern_id] = score
            results['patterns'].append({
                'pattern_id': pattern_id,
                'predictions': pattern_preds,
                'actuals': pattern_actuals.tolist(),
                'eventual': pattern_eventual,
                'score': score
            })

        # Calculate aggregate metrics
        metrics = self._calculate_aggregate_metrics(results, pattern_scores)

        return metrics

    def _evaluate_pattern_consistency(self, predictions: List[str], actuals: List[str],
                                     eventual: str, probabilities: np.ndarray = None) -> float:
        """
        Evaluate consistency of predictions for a single pattern.
        """
        score = 0.0

        # Check if pattern eventually becomes K4 or K3
        is_high_value = eventual in ['K4_EXCEPTIONAL', 'K3_STRONG']

        # Evaluate prediction transitions
        for i in range(len(predictions)):
            pred = predictions[i].split('_')[0]  # Get K0, K1, etc.
            actual = actuals[i].split('_')[0]

            # Base score for correct prediction
            if predictions[i] == actuals[i]:
                score += 1.0

            # Bonus for early correct high-value detection
            if is_high_value and pred in ['K3', 'K4'] and i < len(predictions) / 2:
                score += 2.0  # Bonus for early detection

            # Check consistency with next prediction
            if i < len(predictions) - 1:
                next_pred = predictions[i + 1].split('_')[0]
                switch_key = (pred, next_pred)

                # Apply switch penalty/bonus
                if switch_key in self.switch_penalties:
                    score += self.switch_penalties[switch_key]

            # Special handling for confidence
            if probabilities is not None:
                conf_score = self._evaluate_confidence(
                    probabilities[i], actuals[i], eventual
                )
                score += conf_score

        return score

    def _evaluate_confidence(self, probs: np.ndarray, actual: str, eventual: str) -> float:
        """
        Evaluate confidence-based predictions.
        """
        class_names = ['K0_STAGNANT', 'K1_MINIMAL', 'K2_QUALITY', 'K3_STRONG', 'K4_EXCEPTIONAL', 'K5_FAILED']

        # Get confidence for actual class
        actual_idx = class_names.index(actual)
        actual_conf = probs[actual_idx]

        # Get confidence for eventual class
        eventual_idx = class_names.index(eventual)
        eventual_conf = probs[eventual_idx]

        score = 0.0

        # Reward high confidence in correct prediction
        if actual_conf > 0.6:
            score += actual_conf

        # Bonus for high confidence in eventual outcome (even if early)
        if eventual_conf > 0.3 and eventual in ['K3_STRONG', 'K4_EXCEPTIONAL']:
            score += eventual_conf * 0.5

        # Penalty for high confidence in wrong high-value prediction
        k4_conf = probs[4]
        k3_conf = probs[3]
        if k4_conf > 0.5 and eventual == 'K5_FAILED':
            score -= 5.0  # Heavy penalty for confident K4 that fails

        return score

    def _calculate_aggregate_metrics(self, results: Dict, pattern_scores: Dict) -> Dict:
        """
        Calculate aggregate evaluation metrics.
        """
        print("\n" + "-"*60)
        print("EVALUATION METRICS")
        print("-"*60)

        metrics = {}

        # Calculate average pattern score
        avg_score = np.mean(list(pattern_scores.values()))
        metrics['avg_pattern_score'] = avg_score

        # Find patterns with good/bad consistency
        good_patterns = [p for p, s in pattern_scores.items() if s > 5.0]
        bad_patterns = [p for p, s in pattern_scores.items() if s < -2.0]

        metrics['consistent_patterns'] = len(good_patterns)
        metrics['inconsistent_patterns'] = len(bad_patterns)
        metrics['total_patterns'] = len(pattern_scores)

        print(f"Average Pattern Score: {avg_score:.2f}")
        print(f"Consistent Patterns (score > 5): {len(good_patterns)} ({len(good_patterns)/len(pattern_scores):.1%})")
        print(f"Inconsistent Patterns (score < -2): {len(bad_patterns)} ({len(bad_patterns)/len(pattern_scores):.1%})")

        # Analyze K4 early detection
        k4_patterns = [r for r in results['patterns'] if r['eventual'] == 'K4_EXCEPTIONAL']
        if k4_patterns:
            early_detected = sum(1 for p in k4_patterns
                               if 'K4' in str(p['predictions'][:len(p['predictions'])//2]))
            metrics['k4_early_detection_rate'] = early_detected / len(k4_patterns)
            print(f"\nK4 Early Detection Rate: {metrics['k4_early_detection_rate']:.1%}")

        return metrics


def run_sophisticated_evaluation():
    """
    Run the sophisticated evaluation on existing predictions.
    """
    print("="*60)
    print("SOPHISTICATED EVALUATION PIPELINE")
    print("="*60)

    # Load data
    data_file = Path("output/patterns_labeled_enhanced_20251104_193956.parquet")
    if not data_file.exists():
        print(f"Error: {data_file} not found")
        return

    print(f"\nLoading data from {data_file}")
    df = pd.read_parquet(data_file)

    # Initialize evaluator
    evaluator = SophisticatedEvaluator()

    # Create dual labels
    df = evaluator.create_dual_labels(df)

    # Load model and generate predictions (if available)
    model_file = Path("output/models/xgboost_noleakage_20251104_212857.pkl")
    if model_file.exists():
        print(f"\nLoading model from {model_file}")
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
            model = model_data['model']
            feature_names = model_data['feature_names']

        # Prepare features
        available_features = [f for f in feature_names if f in df.columns]
        X = df[available_features].fillna(0).replace([np.inf, -np.inf], 0)

        # Generate predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)

        # Run sophisticated evaluation
        metrics = evaluator.evaluate_predictions(df, predictions, probabilities)

        # Save results
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        print("""
        Based on the sophisticated evaluation:

        1. Train models to maintain conviction:
           - If predicting K4, stay with K3/K4 predictions
           - Penalize flip-flopping in loss function

        2. Implement confidence thresholds:
           - Only predict K4 with >50% confidence
           - Allow "uncertain" predictions when confidence is low

        3. Use temporal features:
           - Track prediction history within patterns
           - Include "days since first K4 prediction" as feature

        4. Consider ensemble approach:
           - One model for early detection (aggressive)
           - One model for confirmation (conservative)
           - Combine based on pattern age
        """)

    else:
        print("No model found for evaluation. Train a model first.")

    return df


def create_enhanced_training_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create enhanced labels for training that incorporate eventual outcomes.
    """
    print("\n" + "="*60)
    print("CREATING ENHANCED TRAINING LABELS")
    print("="*60)

    # Create pattern IDs
    df['pattern_id'] = df.apply(
        lambda row: f"{row['ticker']}_{row['upper_boundary']:.4f}_{row['lower_boundary']:.4f}",
        axis=1
    )

    # Find eventual outcome for each pattern
    pattern_outcomes = {}
    for pattern_id, group in df.groupby('pattern_id'):
        outcomes = group['outcome_class'].values
        hierarchy = {'K4_EXCEPTIONAL': 5, 'K3_STRONG': 4, 'K2_QUALITY': 3,
                    'K1_MINIMAL': 2, 'K0_STAGNANT': 1, 'K5_FAILED': 0}
        best_outcome = max(outcomes, key=lambda x: hierarchy.get(x, -1))
        pattern_outcomes[pattern_id] = best_outcome

    df['eventual_outcome'] = df['pattern_id'].map(pattern_outcomes)

    # Create weighted labels based on both snapshot and eventual outcomes
    df['training_label'] = df['outcome_class']  # Default to snapshot label

    # For patterns that eventually become K4/K3, adjust early labels
    high_value_patterns = df[df['eventual_outcome'].isin(['K4_EXCEPTIONAL', 'K3_STRONG'])]

    for pattern_id, group in high_value_patterns.groupby('pattern_id'):
        group = group.sort_values('snapshot_date')
        eventual = group['eventual_outcome'].iloc[0]

        # For early snapshots of eventual K4/K3 patterns, upgrade labels slightly
        for i, (idx, row) in enumerate(group.iterrows()):
            if i < len(group) / 2:  # First half of pattern
                current = row['outcome_class']
                # If currently K0/K1, upgrade to K2
                if current in ['K0_STAGNANT', 'K1_MINIMAL'] and eventual == 'K4_EXCEPTIONAL':
                    df.at[idx, 'training_label'] = 'K2_QUALITY'
                elif current in ['K0_STAGNANT'] and eventual == 'K3_STRONG':
                    df.at[idx, 'training_label'] = 'K1_MINIMAL'

    print("Label adjustments:")
    changes = (df['training_label'] != df['outcome_class']).sum()
    print(f"  Modified labels: {changes} ({changes/len(df):.1%})")

    return df


if __name__ == "__main__":
    # Run evaluation
    df = run_sophisticated_evaluation()

    # Create enhanced training labels
    if df is not None:
        df_enhanced = create_enhanced_training_labels(df)

        # Save enhanced dataset
        output_file = Path("output/patterns_sophisticated_labels.parquet")
        df_enhanced.to_parquet(output_file, index=False)
        print(f"\nEnhanced dataset saved to: {output_file}")