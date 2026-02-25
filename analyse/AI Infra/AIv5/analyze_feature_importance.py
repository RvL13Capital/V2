"""
Feature Importance Analysis
============================
Analyze which features are most important for detecting BIG_WINNER patterns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from collections import defaultdict


class FeatureImportanceAnalyzer:
    """
    Analyze feature importance across trained models.
    """

    def __init__(self):
        # Define feature categories
        self.feature_categories = {
            'Pattern Metrics': [
                'days_in_pattern', 'days_qualifying', 'days_active', 'days_since_activation',
                'range_width_pct', 'consolidation_quality_score'
            ],
            'Volatility (BBW)': [
                'current_bbw_20', 'current_bbw_percentile', 'bbw_percentile',
                'baseline_bbw_avg', 'bbw_compression_ratio', 'bbw_slope_20d', 'bbw_std_20d'
            ],
            'Trend (ADX)': [
                'current_adx', 'adx_slope_20d'
            ],
            'Volume': [
                'current_volume', 'current_volume_ratio_20', 'baseline_volume_avg',
                'volume_compression_ratio'
            ],
            'Price Position': [
                'current_price', 'start_price', 'price_position_in_range',
                'price_distance_from_upper_pct', 'price_distance_from_lower_pct',
                'distance_from_power_pct'
            ],
            'Range & Volatility': [
                'current_range_ratio', 'avg_range_20d', 'price_volatility_20d',
                'baseline_volatility', 'volatility_compression_ratio'
            ],
            'Boundaries': [
                'current_high', 'current_low'
            ]
        }

        # Reverse mapping: feature -> category
        self.feature_to_category = {}
        for category, features in self.feature_categories.items():
            for feature in features:
                self.feature_to_category[feature] = category

    def load_model(self, model_path: Path) -> Tuple[object, List[str]]:
        """Load a trained model and its feature names."""
        print(f"\nLoading model from {model_path}")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        model = model_data['model']
        feature_names = model_data['feature_names']

        print(f"  Model type: {type(model).__name__}")
        print(f"  Features: {len(feature_names)}")

        return model, feature_names

    def extract_importance(self, model, feature_names: List[str],
                          model_name: str) -> pd.DataFrame:
        """Extract feature importance from a model."""
        # Get importance scores
        if hasattr(model, 'feature_importances_'):
            importance_scores = model.feature_importances_
        elif hasattr(model, 'get_score'):
            # LightGBM get_score returns dict
            score_dict = model.get_score(importance_type='gain')
            importance_scores = np.array([score_dict.get(f, 0.0) for f in feature_names])
        else:
            raise ValueError(f"Cannot extract importance from {type(model)}")

        # Create dataframe
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores,
            'model': model_name
        })

        # Add category
        df['category'] = df['feature'].map(self.feature_to_category).fillna('Other')

        # Normalize importance
        df['importance_pct'] = df['importance'] / df['importance'].sum() * 100

        # Sort by importance
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)

        return df

    def analyze_category_importance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate importance by category."""
        category_importance = df.groupby('category').agg({
            'importance': 'sum',
            'importance_pct': 'sum',
            'feature': 'count'
        }).rename(columns={'feature': 'num_features'})

        category_importance = category_importance.sort_values('importance', ascending=False)

        return category_importance

    def compare_models(self, importance_dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """Compare feature importance across multiple models."""
        # Combine all dataframes
        combined = pd.concat(importance_dfs, ignore_index=True)

        # Pivot to compare models side-by-side
        comparison = combined.pivot_table(
            index='feature',
            columns='model',
            values='importance_pct',
            fill_value=0
        )

        # Add average importance
        comparison['avg_importance'] = comparison.mean(axis=1)
        comparison = comparison.sort_values('avg_importance', ascending=False)

        return comparison

    def plot_top_features(self, df: pd.DataFrame, model_name: str,
                         top_n: int = 20, save_path: Path = None):
        """Plot top N features by importance."""
        top_features = df.head(top_n)

        plt.figure(figsize=(12, 8))

        # Color by category
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.feature_categories)))
        category_colors = dict(zip(self.feature_categories.keys(), colors))
        bar_colors = [category_colors.get(cat, 'gray') for cat in top_features['category']]

        plt.barh(range(len(top_features)), top_features['importance_pct'], color=bar_colors)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance (%)')
        plt.title(f'Top {top_n} Features - {model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved plot to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_category_importance(self, category_df: pd.DataFrame,
                                model_name: str, save_path: Path = None):
        """Plot importance by category."""
        plt.figure(figsize=(10, 6))

        plt.bar(range(len(category_df)), category_df['importance_pct'])
        plt.xticks(range(len(category_df)), category_df.index, rotation=45, ha='right')
        plt.ylabel('Total Importance (%)')
        plt.title(f'Feature Category Importance - {model_name}')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved plot to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_model_comparison(self, comparison_df: pd.DataFrame,
                            top_n: int = 15, save_path: Path = None):
        """Plot feature importance comparison across models."""
        top_features = comparison_df.head(top_n)

        # Remove avg_importance column for plotting
        plot_data = top_features.drop(columns=['avg_importance'])

        plt.figure(figsize=(12, 8))

        x = np.arange(len(plot_data))
        width = 0.8 / len(plot_data.columns)

        for i, col in enumerate(plot_data.columns):
            offset = (i - len(plot_data.columns) / 2) * width + width / 2
            plt.barh(x + offset, plot_data[col], width, label=col)

        plt.yticks(x, plot_data.index)
        plt.xlabel('Importance (%)')
        plt.title(f'Top {top_n} Features - Model Comparison')
        plt.legend(loc='lower right')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved plot to {save_path}")
        else:
            plt.show()

        plt.close()

    def print_summary(self, df: pd.DataFrame, model_name: str):
        """Print summary of feature importance."""
        print(f"\n{'='*60}")
        print(f"FEATURE IMPORTANCE SUMMARY - {model_name}")
        print(f"{'='*60}")

        print(f"\nTop 15 Features:")
        print(f"{'Rank':<6} {'Feature':<35} {'Importance':<12} {'Category':<20}")
        print("-" * 75)

        for idx, row in df.head(15).iterrows():
            print(f"{idx+1:<6} {row['feature']:<35} {row['importance_pct']:>10.2f}%  {row['category']:<20}")

        # Category summary
        category_df = self.analyze_category_importance(df)

        print(f"\n{'='*60}")
        print("CATEGORY IMPORTANCE")
        print(f"{'='*60}")
        print(f"{'Category':<30} {'Total %':<12} {'# Features':<12}")
        print("-" * 60)

        for category, row in category_df.iterrows():
            print(f"{category:<30} {row['importance_pct']:>10.2f}%  {row['num_features']:>10}")

    def analyze_all_models(self, model_dir: Path, output_dir: Path = None):
        """Analyze all trained models in a directory."""
        print("="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)

        if output_dir is None:
            output_dir = model_dir / 'feature_importance'
        output_dir.mkdir(exist_ok=True)

        # Find all model files
        model_files = list(model_dir.glob('*_improved_*.pkl'))

        if not model_files:
            print(f"No models found in {model_dir}")
            return

        print(f"\nFound {len(model_files)} models:")
        for mf in model_files:
            print(f"  - {mf.name}")

        # Analyze each model
        all_importance_dfs = []

        for model_file in model_files:
            model_name = model_file.stem

            # Load model
            model, feature_names = self.load_model(model_file)

            # Extract importance
            importance_df = self.extract_importance(model, feature_names, model_name)
            all_importance_dfs.append(importance_df)

            # Print summary
            self.print_summary(importance_df, model_name)

            # Plot top features
            plot_path = output_dir / f'{model_name}_top_features.png'
            self.plot_top_features(importance_df, model_name, top_n=20, save_path=plot_path)

            # Plot category importance
            category_df = self.analyze_category_importance(importance_df)
            category_plot_path = output_dir / f'{model_name}_category_importance.png'
            self.plot_category_importance(category_df, model_name, save_path=category_plot_path)

            # Save importance to CSV
            csv_path = output_dir / f'{model_name}_importance.csv'
            importance_df.to_csv(csv_path, index=False)
            print(f"  Saved importance data to {csv_path}")

        # Compare models
        if len(all_importance_dfs) > 1:
            print(f"\n{'='*60}")
            print("MODEL COMPARISON")
            print(f"{'='*60}")

            comparison_df = self.compare_models(all_importance_dfs)

            # Print comparison
            print(f"\nTop 10 Features (Average Importance Across Models):")
            print(comparison_df.head(10))

            # Plot comparison
            comparison_plot_path = output_dir / 'model_comparison_top_features.png'
            self.plot_model_comparison(comparison_df, top_n=15, save_path=comparison_plot_path)

            # Save comparison to CSV
            comparison_csv_path = output_dir / 'model_comparison.csv'
            comparison_df.to_csv(comparison_csv_path)
            print(f"\n  Saved comparison to {comparison_csv_path}")

        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Results saved to: {output_dir}")


def main():
    """Main analysis pipeline."""
    print("="*60)
    print("FEATURE IMPORTANCE ANALYSIS PIPELINE")
    print("="*60)

    # Paths
    model_dir = Path("output/models")
    output_dir = Path("output/feature_importance")

    # Initialize analyzer
    analyzer = FeatureImportanceAnalyzer()

    # Analyze all models
    analyzer.analyze_all_models(model_dir, output_dir)


if __name__ == "__main__":
    main()
