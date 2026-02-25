"""
Feature Importance Tracking and Analysis
=========================================
Tracks feature importance across models and training runs,
provides visualization and selection recommendations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class FeatureImportanceTracker:
    """
    Comprehensive feature importance tracking system.
    """

    def __init__(self, save_dir: str = 'output/feature_analysis'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.importance_history = []
        self.feature_rankings = {}
        self.stability_scores = {}

    def track_importance(self, feature_importance: Dict, model_name: str,
                        timestamp: Optional[str] = None):
        """
        Track feature importance from a model.
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        entry = {
            'timestamp': timestamp,
            'model': model_name,
            'importance': feature_importance
        }

        self.importance_history.append(entry)
        self._update_rankings(feature_importance, model_name)

    def _update_rankings(self, feature_importance: Dict, model_name: str):
        """
        Update feature rankings based on new importance scores.
        """
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(),
                                key=lambda x: x[1], reverse=True)

        # Update rankings
        if model_name not in self.feature_rankings:
            self.feature_rankings[model_name] = {}

        for rank, (feature, importance) in enumerate(sorted_features, 1):
            if feature not in self.feature_rankings[model_name]:
                self.feature_rankings[model_name][feature] = []
            self.feature_rankings[model_name][feature].append(rank)

    def calculate_stability_scores(self) -> Dict[str, float]:
        """
        Calculate stability scores for each feature across runs.
        """
        stability_scores = {}

        for model_name, rankings in self.feature_rankings.items():
            for feature, rank_history in rankings.items():
                if len(rank_history) < 2:
                    continue

                # Calculate coefficient of variation
                rank_array = np.array(rank_history)
                cv = np.std(rank_array) / np.mean(rank_array) if np.mean(rank_array) > 0 else 0

                # Stability score (inverse of CV, normalized)
                stability = 1 / (1 + cv)

                if feature not in stability_scores:
                    stability_scores[feature] = []
                stability_scores[feature].append(stability)

        # Average stability across models
        self.stability_scores = {
            feature: np.mean(scores)
            for feature, scores in stability_scores.items()
        }

        return self.stability_scores

    def get_top_features(self, n: int = 20, min_stability: float = 0.7) -> List[str]:
        """
        Get top N features based on average importance and stability.
        """
        # Calculate average importance
        feature_importance_avg = {}

        for entry in self.importance_history:
            for feature, importance in entry['importance'].items():
                if feature not in feature_importance_avg:
                    feature_importance_avg[feature] = []
                feature_importance_avg[feature].append(importance)

        # Average across all runs
        feature_scores = {}
        for feature, importances in feature_importance_avg.items():
            avg_importance = np.mean(importances)
            stability = self.stability_scores.get(feature, 0)

            # Combined score (weighted average)
            if stability >= min_stability:
                feature_scores[feature] = avg_importance * 0.7 + stability * 0.3
            else:
                feature_scores[feature] = avg_importance * 0.5  # Penalize unstable features

        # Sort and return top N
        sorted_features = sorted(feature_scores.items(),
                                key=lambda x: x[1], reverse=True)

        return [feature for feature, _ in sorted_features[:n]]

    def visualize_importance(self, top_n: int = 30):
        """
        Create visualization of feature importance.
        """
        if not self.importance_history:
            print("No importance data to visualize")
            return

        # Aggregate importance scores
        feature_importance_all = {}

        for entry in self.importance_history:
            model = entry['model']
            for feature, importance in entry['importance'].items():
                key = f"{feature}_{model}"
                if key not in feature_importance_all:
                    feature_importance_all[key] = []
                feature_importance_all[key].append(importance)

        # Calculate mean importance
        mean_importance = {
            key: np.mean(values)
            for key, values in feature_importance_all.items()
        }

        # Get top features
        sorted_features = sorted(mean_importance.items(),
                                key=lambda x: x[1], reverse=True)[:top_n]

        # Create plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Top feature importance
        ax1 = axes[0, 0]
        features, importances = zip(*sorted_features)
        y_pos = np.arange(len(features))
        ax1.barh(y_pos, importances)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([f.split('_')[0][:20] for f in features], fontsize=8)
        ax1.set_xlabel('Importance Score')
        ax1.set_title('Top Feature Importance')
        ax1.invert_yaxis()

        # Plot 2: Stability scores
        ax2 = axes[0, 1]
        if self.stability_scores:
            stability_sorted = sorted(self.stability_scores.items(),
                                    key=lambda x: x[1], reverse=True)[:20]
            features_s, scores_s = zip(*stability_sorted)
            y_pos_s = np.arange(len(features_s))
            colors = ['green' if s >= 0.7 else 'orange' if s >= 0.5 else 'red'
                     for s in scores_s]
            ax2.barh(y_pos_s, scores_s, color=colors)
            ax2.set_yticks(y_pos_s)
            ax2.set_yticklabels([f[:20] for f in features_s], fontsize=8)
            ax2.set_xlabel('Stability Score')
            ax2.set_title('Feature Stability (Green>0.7, Orange>0.5, Red<0.5)')
            ax2.invert_yaxis()

        # Plot 3: Feature importance over time
        ax3 = axes[1, 0]
        top_features_to_track = self.get_top_features(5)
        for feature in top_features_to_track:
            importance_over_time = []
            timestamps = []

            for entry in self.importance_history:
                if feature in entry['importance']:
                    importance_over_time.append(entry['importance'][feature])
                    timestamps.append(entry['timestamp'])

            if importance_over_time:
                ax3.plot(range(len(importance_over_time)),
                        importance_over_time, label=feature[:20], marker='o')

        ax3.set_xlabel('Training Run')
        ax3.set_ylabel('Importance Score')
        ax3.set_title('Top 5 Feature Importance Over Time')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        # Plot 4: Feature categories
        ax4 = axes[1, 1]
        categories = self._categorize_features()
        if categories:
            labels = list(categories.keys())
            sizes = list(categories.values())
            colors = plt.cm.Set3(range(len(labels)))
            ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
            ax4.set_title('Feature Categories Distribution')

        plt.tight_layout()
        save_path = self.save_dir / f'feature_importance_{datetime.now():%Y%m%d_%H%M%S}.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()
        print(f"Visualization saved to {save_path}")

    def _categorize_features(self) -> Dict[str, int]:
        """
        Categorize features by type.
        """
        categories = {
            'Price': 0,
            'Volume': 0,
            'Volatility': 0,
            'Pattern': 0,
            'Anomaly': 0,
            'Character': 0,
            'Attention': 0,
            'Market': 0,
            'Other': 0
        }

        for entry in self.importance_history:
            for feature in entry['importance'].keys():
                feature_lower = feature.lower()
                if any(x in feature_lower for x in ['price', 'close', 'high', 'low', 'open']):
                    categories['Price'] += 1
                elif any(x in feature_lower for x in ['volume', 'vol']):
                    categories['Volume'] += 1
                elif any(x in feature_lower for x in ['volatility', 'bbw', 'atr', 'std']):
                    categories['Volatility'] += 1
                elif any(x in feature_lower for x in ['pattern', 'days', 'boundary']):
                    categories['Pattern'] += 1
                elif any(x in feature_lower for x in ['anomaly', 'outlier']):
                    categories['Anomaly'] += 1
                elif any(x in feature_lower for x in ['character', 'rating']):
                    categories['Character'] += 1
                elif any(x in feature_lower for x in ['attention', 'weight']):
                    categories['Attention'] += 1
                elif any(x in feature_lower for x in ['market', 'regime', 'momentum']):
                    categories['Market'] += 1
                else:
                    categories['Other'] += 1

        # Remove empty categories
        return {k: v for k, v in categories.items() if v > 0}

    def recommend_features(self, target_features: int = 50) -> Dict:
        """
        Recommend optimal feature set based on importance and stability.
        """
        recommendations = {
            'must_have': [],
            'recommended': [],
            'optional': [],
            'remove': []
        }

        # Calculate scores
        self.calculate_stability_scores()

        # Get all features with scores
        feature_scores = {}
        for entry in self.importance_history:
            for feature, importance in entry['importance'].items():
                if feature not in feature_scores:
                    feature_scores[feature] = {
                        'importance': [],
                        'stability': self.stability_scores.get(feature, 0)
                    }
                feature_scores[feature]['importance'].append(importance)

        # Calculate final scores
        final_scores = {}
        for feature, data in feature_scores.items():
            avg_importance = np.mean(data['importance'])
            stability = data['stability']

            # Combined score
            final_score = avg_importance * 0.6 + stability * 0.4
            final_scores[feature] = {
                'score': final_score,
                'importance': avg_importance,
                'stability': stability
            }

        # Sort by score
        sorted_features = sorted(final_scores.items(),
                                key=lambda x: x[1]['score'], reverse=True)

        # Categorize recommendations
        for i, (feature, data) in enumerate(sorted_features):
            if i < target_features * 0.3:  # Top 30%
                if data['stability'] >= 0.8 and data['importance'] >= 0.1:
                    recommendations['must_have'].append(feature)
                else:
                    recommendations['recommended'].append(feature)
            elif i < target_features:  # Next 70%
                recommendations['optional'].append(feature)
            else:  # Beyond target
                if data['importance'] < 0.01 or data['stability'] < 0.3:
                    recommendations['remove'].append(feature)

        return recommendations

    def generate_report(self) -> str:
        """
        Generate comprehensive feature importance report.
        """
        report = []
        report.append("="*60)
        report.append("FEATURE IMPORTANCE ANALYSIS REPORT")
        report.append("="*60)
        report.append(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
        report.append(f"Total training runs: {len(self.importance_history)}")

        # Calculate statistics
        self.calculate_stability_scores()

        # Top features
        top_features = self.get_top_features(20)
        report.append("\n" + "="*60)
        report.append("TOP 20 FEATURES")
        report.append("="*60)

        for i, feature in enumerate(top_features, 1):
            stability = self.stability_scores.get(feature, 0)
            report.append(f"{i:2}. {feature:40} (Stability: {stability:.2f})")

        # Recommendations
        recommendations = self.recommend_features()
        report.append("\n" + "="*60)
        report.append("FEATURE RECOMMENDATIONS")
        report.append("="*60)

        report.append(f"\nMUST HAVE ({len(recommendations['must_have'])} features):")
        for feature in recommendations['must_have'][:10]:
            report.append(f"  • {feature}")

        report.append(f"\nRECOMMENDED ({len(recommendations['recommended'])} features):")
        for feature in recommendations['recommended'][:10]:
            report.append(f"  • {feature}")

        report.append(f"\nCONSIDER REMOVING ({len(recommendations['remove'])} features):")
        for feature in recommendations['remove'][:10]:
            report.append(f"  • {feature}")

        # Stability analysis
        report.append("\n" + "="*60)
        report.append("STABILITY ANALYSIS")
        report.append("="*60)

        stable_features = [f for f, s in self.stability_scores.items() if s >= 0.8]
        unstable_features = [f for f, s in self.stability_scores.items() if s < 0.5]

        report.append(f"\nHighly stable features (stability >= 0.8): {len(stable_features)}")
        report.append(f"Unstable features (stability < 0.5): {len(unstable_features)}")

        # Category analysis
        categories = self._categorize_features()
        if categories:
            report.append("\n" + "="*60)
            report.append("FEATURE CATEGORIES")
            report.append("="*60)
            total = sum(categories.values())
            for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                pct = count / total * 100
                report.append(f"  {category:12}: {count:4} ({pct:5.1f}%)")

        # Save report
        report_text = "\n".join(report)
        save_path = self.save_dir / f'feature_report_{datetime.now():%Y%m%d_%H%M%S}.txt'
        with open(save_path, 'w') as f:
            f.write(report_text)

        print(report_text)
        print(f"\nReport saved to {save_path}")

        return report_text

    def save_state(self):
        """
        Save tracker state to file.
        """
        state = {
            'importance_history': self.importance_history,
            'feature_rankings': self.feature_rankings,
            'stability_scores': self.stability_scores
        }

        save_path = self.save_dir / 'tracker_state.json'
        with open(save_path, 'w') as f:
            json.dump(state, f, indent=2, default=str)

        print(f"Tracker state saved to {save_path}")

    def load_state(self):
        """
        Load tracker state from file.
        """
        load_path = self.save_dir / 'tracker_state.json'
        if load_path.exists():
            with open(load_path, 'r') as f:
                state = json.load(f)
            self.importance_history = state['importance_history']
            self.feature_rankings = state['feature_rankings']
            self.stability_scores = state['stability_scores']
            print(f"Tracker state loaded from {load_path}")
        else:
            print(f"No saved state found at {load_path}")


def integrate_with_training(model, X_train, feature_names, tracker, model_name):
    """
    Helper function to integrate tracker with training pipeline.
    """
    # Get feature importance from model
    if hasattr(model, 'feature_importances_'):
        importance_scores = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance_scores = np.abs(model.coef_[0])
    else:
        print(f"Model {model_name} does not have feature importance")
        return

    # Create importance dictionary
    importance_dict = dict(zip(feature_names, importance_scores))

    # Track importance
    tracker.track_importance(importance_dict, model_name)

    # Print top features
    sorted_features = sorted(importance_dict.items(),
                            key=lambda x: x[1], reverse=True)[:10]

    print(f"\nTop 10 features for {model_name}:")
    for feature, importance in sorted_features:
        print(f"  {feature}: {importance:.4f}")


if __name__ == "__main__":
    # Example usage
    tracker = FeatureImportanceTracker()

    # Simulate tracking from multiple models
    example_importance = {
        'price_to_upper_pct': 0.15,
        'volume_anomaly_score': 0.12,
        'compression_character': 0.10,
        'attention_weighted_return': 0.09,
        'pattern_readiness': 0.08,
        'volatility_regime': 0.07,
        'market_momentum': 0.06,
        'boundary_test_frequency': 0.05,
        'anomaly_persistence': 0.04,
        'price_smoothness': 0.03
    }

    tracker.track_importance(example_importance, 'xgboost')

    # Generate report
    tracker.generate_report()

    print("\nFeature Importance Tracker initialized successfully!")
    print("Use integrate_with_training() to track importance during model training")