"""
Pattern DNA Explorer - Interactive Sunburst Visualization
Analyzes the anatomy of successful and failed consolidation patterns
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import webbrowser
import os
from typing import Dict, List, Tuple, Optional


class PatternDNAExplorer:
    """Dissect the DNA of consolidation patterns to find winning combinations"""

    def __init__(self):
        self.df = None
        self.enriched_df = None
        self.outcome_classes = None

    def load_parquet(self, file_path: str) -> pd.DataFrame:
        """Load parquet file with proper date handling"""
        table = pq.read_table(file_path)

        # Convert to pandas
        data_dict = {}
        for col_name in table.column_names:
            col = table.column(col_name)
            if 'date' in str(col.type).lower():
                data_dict[col_name] = col.to_pylist()
            else:
                data_dict[col_name] = col.to_pandas()

        df = pd.DataFrame(data_dict)

        # Ensure date is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        self.df = df
        print(f"[OK] Loaded {len(df)} rows")
        return df

    def classify_outcomes(self) -> pd.DataFrame:
        """Classify patterns into outcome classes based on performance"""

        df = self.df.copy()

        # Define outcome classes based on 40-day gains
        def classify_outcome(gain):
            if gain >= 0.75:  # ≥75% gain
                return 'K4_Exceptional'
            elif gain >= 0.35:  # 35-75% gain
                return 'K3_Strong'
            elif gain >= 0.15:  # 15-35% gain
                return 'K2_Quality'
            elif gain >= 0.05:  # 5-15% gain
                return 'K1_Minimal'
            elif gain >= -0.05:  # -5% to 5% (stagnant)
                return 'K0_Stagnant'
            else:  # < -5% (failed)
                return 'K5_Failed'

        df['outcome_class'] = df['max_gain_40d'].apply(classify_outcome)

        # Strategic value assignment
        value_map = {
            'K4_Exceptional': 10,
            'K3_Strong': 3,
            'K2_Quality': 1,
            'K1_Minimal': -0.2,
            'K0_Stagnant': -2,
            'K5_Failed': -10
        }
        df['strategic_value'] = df['outcome_class'].map(value_map)

        self.outcome_classes = df['outcome_class'].value_counts()
        print("\nOutcome Classification:")
        for outcome, count in self.outcome_classes.items():
            percentage = count / len(df) * 100
            print(f"  {outcome}: {count} ({percentage:.1f}%)")

        return df

    def engineer_features(self) -> pd.DataFrame:
        """Engineer additional features for pattern DNA analysis"""

        df = self.enriched_df if self.enriched_df is not None else self.classify_outcomes()

        # 1. Volatility Categories (BBW)
        if 'bbw' in df.columns:
            bbw_percentiles = df['bbw'].quantile([0.33, 0.67])
            df['volatility_category'] = pd.cut(
                df['bbw'],
                bins=[-np.inf, bbw_percentiles[0.33], bbw_percentiles[0.67], np.inf],
                labels=['Low_Volatility', 'Medium_Volatility', 'High_Volatility']
            )

        # 2. Volume Categories
        if 'volume_ratio' in df.columns:
            df['volume_category'] = pd.cut(
                df['volume_ratio'],
                bins=[-np.inf, 0.5, 1.0, 2.0, np.inf],
                labels=['Very_Low_Volume', 'Low_Volume', 'Normal_Volume', 'High_Volume']
            )

        # 3. ATR Categories
        if 'atr_pct' in df.columns:
            atr_percentiles = df['atr_pct'].quantile([0.33, 0.67])
            df['atr_category'] = pd.cut(
                df['atr_pct'],
                bins=[-np.inf, atr_percentiles[0.33], atr_percentiles[0.67], np.inf],
                labels=['Low_ATR', 'Medium_ATR', 'High_ATR']
            )

        # 4. Range Categories
        if 'range_ratio' in df.columns:
            df['range_category'] = pd.cut(
                df['range_ratio'],
                bins=[-np.inf, 0.5, 0.75, 1.0, np.inf],
                labels=['Tight_Range', 'Normal_Range', 'Wide_Range', 'Very_Wide_Range']
            )

        # 5. Method Agreement Categories
        method_cols = ['method1_bollinger', 'method2_range_based',
                      'method3_volume_weighted', 'method4_atr_based']
        df['methods_agreed'] = df[method_cols].sum(axis=1)
        df['agreement_category'] = df['methods_agreed'].apply(
            lambda x: f'{x}_Method{"s" if x != 1 else ""}_Agree'
        )

        # 6. RSI Proxy (using price position in range)
        if 'high' in df.columns and 'low' in df.columns and 'price' in df.columns:
            # Calculate a simple RSI proxy based on price position
            df['price_position'] = (df['price'] - df['low']) / (df['high'] - df['low'])
            df['rsi_category'] = pd.cut(
                df['price_position'],
                bins=[0, 0.3, 0.7, 1.0],
                labels=['Oversold', 'Neutral', 'Overbought']
            )

        # 7. Temporal Features
        if 'date' in df.columns:
            df['month'] = df['date'].dt.month
            df['quarter'] = df['date'].dt.quarter
            df['day_of_week'] = df['date'].dt.dayofweek

            # Season categories
            df['season'] = pd.cut(
                df['month'],
                bins=[0, 3, 6, 9, 12],
                labels=['Q1_Winter', 'Q2_Spring', 'Q3_Summer', 'Q4_Fall']
            )

        self.enriched_df = df
        print(f"\n[OK] Engineered {len([col for col in df.columns if 'category' in col])} categorical features")
        return df

    def create_sunburst_dna(self,
                           inner_feature: str = 'outcome_class',
                           middle_feature: str = 'volatility_category',
                           outer_feature: str = 'volume_category',
                           save_path: str = "pattern_dna_sunburst.html") -> go.Figure:
        """Create interactive sunburst chart showing pattern DNA"""

        if self.enriched_df is None:
            self.engineer_features()

        df = self.enriched_df

        # Prepare hierarchical data
        sunburst_data = []

        # Build the hierarchy
        for outcome in df[inner_feature].unique():
            if pd.isna(outcome):
                continue

            outcome_df = df[df[inner_feature] == outcome]
            outcome_gain = outcome_df['max_gain_40d'].mean() * 100
            outcome_count = len(outcome_df)

            # Add inner ring (outcome)
            sunburst_data.append({
                'labels': outcome,
                'parents': '',
                'values': outcome_count,
                'gain': outcome_gain,
                'text': f'{outcome}<br>Count: {outcome_count}<br>Avg Gain: {outcome_gain:.1f}%'
            })

            # Middle ring
            for middle_val in outcome_df[middle_feature].unique():
                if pd.isna(middle_val):
                    continue

                middle_df = outcome_df[outcome_df[middle_feature] == middle_val]
                middle_gain = middle_df['max_gain_40d'].mean() * 100
                middle_count = len(middle_df)

                middle_label = f'{outcome}_{middle_val}'
                sunburst_data.append({
                    'labels': middle_val,
                    'parents': outcome,
                    'values': middle_count,
                    'gain': middle_gain,
                    'text': f'{middle_val}<br>Count: {middle_count}<br>Avg Gain: {middle_gain:.1f}%'
                })

                # Outer ring
                for outer_val in middle_df[outer_feature].unique():
                    if pd.isna(outer_val):
                        continue

                    outer_df = middle_df[middle_df[outer_feature] == outer_val]
                    outer_gain = outer_df['max_gain_40d'].mean() * 100
                    outer_count = len(outer_df)

                    sunburst_data.append({
                        'labels': outer_val,
                        'parents': middle_val,
                        'values': outer_count,
                        'gain': outer_gain,
                        'text': f'{outer_val}<br>Count: {outer_count}<br>Avg Gain: {outer_gain:.1f}%'
                    })

        # Convert to DataFrame for easier handling
        sunburst_df = pd.DataFrame(sunburst_data)

        # Create color scale based on gain
        fig = go.Figure(go.Sunburst(
            labels=sunburst_df['labels'],
            parents=sunburst_df['parents'],
            values=sunburst_df['values'],
            branchvalues="total",
            marker=dict(
                colors=sunburst_df['gain'],
                colorscale='RdYlGn',
                cmid=0,
                cmin=-20,
                cmax=50,
                colorbar=dict(
                    title="Avg Gain %",
                    thickness=20,
                    len=0.7
                )
            ),
            text=sunburst_df['text'],
            hovertemplate='<b>%{label}</b><br>%{text}<br>Size: %{value}<extra></extra>',
            textfont=dict(size=10)
        ))

        fig.update_layout(
            title={
                'text': f'<b>Pattern DNA Explorer</b><br>' +
                       f'<sub>Hierarchy: {inner_feature} → {middle_feature} → {outer_feature}</sub>',
                'font': {'size': 20},
                'x': 0.5,
                'xanchor': 'center'
            },
            width=1000,
            height=800,
            margin=dict(t=100, b=50, l=50, r=50)
        )

        # Save the figure
        fig.write_html(save_path)
        print(f"\n[OK] Sunburst DNA chart saved to: {save_path}")

        return fig

    def create_success_recipe_analysis(self, save_path: str = "pattern_recipes.html") -> go.Figure:
        """Analyze and visualize the most successful pattern recipes"""

        if self.enriched_df is None:
            self.engineer_features()

        df = self.enriched_df

        # Find winning combinations
        feature_combos = []

        # Analyze each combination
        for volatility in df['volatility_category'].unique():
            if pd.isna(volatility):
                continue
            for volume in df['volume_category'].unique():
                if pd.isna(volume):
                    continue
                for agreement in df['agreement_category'].unique():
                    if pd.isna(agreement):
                        continue

                    # Filter for this combination
                    combo_df = df[
                        (df['volatility_category'] == volatility) &
                        (df['volume_category'] == volume) &
                        (df['agreement_category'] == agreement)
                    ]

                    if len(combo_df) >= 3:  # Minimum sample size
                        feature_combos.append({
                            'volatility': str(volatility),
                            'volume': str(volume),
                            'agreement': str(agreement),
                            'count': len(combo_df),
                            'avg_gain_20d': combo_df['max_gain_20d'].mean() * 100,
                            'avg_gain_40d': combo_df['max_gain_40d'].mean() * 100,
                            'success_rate': (combo_df['max_gain_40d'] > 0.2).mean() * 100,
                            'risk_reward': abs(combo_df['max_gain_40d'].mean() /
                                             combo_df['max_loss_40d'].mean()) if combo_df['max_loss_40d'].mean() != 0 else 0,
                            'expected_value': combo_df['strategic_value'].mean()
                        })

        combo_df = pd.DataFrame(feature_combos)

        if len(combo_df) == 0:
            print("Not enough data for combination analysis")
            return None

        # Sort by expected value
        combo_df = combo_df.sort_values('expected_value', ascending=False)

        # Create visualization with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Top 20 Success Recipes by Expected Value',
                'Success Rate vs Average Gain',
                'Feature Importance Heatmap',
                'Risk/Reward Distribution'
            ),
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "heatmap"}, {"type": "box"}]
            ]
        )

        # 1. Top Success Recipes
        top_recipes = combo_df.head(20)
        top_recipes['label'] = (top_recipes['volatility'].str.split('_').str[0] + '/' +
                                top_recipes['volume'].str.split('_').str[0] + '/' +
                                top_recipes['agreement'].str.split('_').str[0])

        fig.add_trace(
            go.Bar(
                y=top_recipes['label'],
                x=top_recipes['expected_value'],
                orientation='h',
                marker_color=top_recipes['expected_value'],
                marker_colorscale='RdYlGn',
                text=[f'EV: {ev:.2f}<br>Gain: {g:.1f}%<br>N={n}'
                      for ev, g, n in zip(top_recipes['expected_value'],
                                         top_recipes['avg_gain_40d'],
                                         top_recipes['count'])],
                textposition='auto',
                hovertemplate='<b>%{y}</b><br>Expected Value: %{x:.2f}<br>%{text}<extra></extra>'
            ),
            row=1, col=1
        )

        # 2. Success Rate vs Gain Scatter
        fig.add_trace(
            go.Scatter(
                x=combo_df['success_rate'],
                y=combo_df['avg_gain_40d'],
                mode='markers',
                marker=dict(
                    size=combo_df['count'],
                    sizemode='area',
                    sizeref=2.*max(combo_df['count'])/(40.**2),
                    color=combo_df['expected_value'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="EV", x=0.98)
                ),
                text=[f'{v}<br>{vol}<br>{a}' for v, vol, a in
                      zip(combo_df['volatility'], combo_df['volume'], combo_df['agreement'])],
                hovertemplate='Success Rate: %{x:.1f}%<br>Avg Gain: %{y:.1f}%<br>%{text}<extra></extra>'
            ),
            row=1, col=2
        )

        # Add quadrant lines
        fig.add_shape(type="line", x0=50, x1=50, y0=0, y1=combo_df['avg_gain_40d'].max(),
                     line=dict(color="gray", dash="dash"), row=1, col=2)
        fig.add_shape(type="line", x0=0, x1=100, y0=20, y1=20,
                     line=dict(color="gray", dash="dash"), row=1, col=2)

        # 3. Feature Importance Heatmap
        # Calculate average gains for each feature value
        feature_importance = []

        for feature, feature_col in [('Volatility', 'volatility_category'),
                                    ('Volume', 'volume_category'),
                                    ('Agreement', 'agreement_category')]:
            for value in df[feature_col].unique():
                if pd.isna(value):
                    continue
                subset = df[df[feature_col] == value]
                feature_importance.append({
                    'feature': feature,
                    'value': str(value).split('_')[0],
                    'avg_gain': subset['max_gain_40d'].mean() * 100,
                    'count': len(subset)
                })

        importance_df = pd.DataFrame(feature_importance)
        pivot_importance = importance_df.pivot(index='feature', columns='value', values='avg_gain')

        fig.add_trace(
            go.Heatmap(
                z=pivot_importance.values,
                x=pivot_importance.columns,
                y=pivot_importance.index,
                colorscale='RdYlGn',
                text=[[f'{val:.1f}%' for val in row] for row in pivot_importance.values],
                texttemplate='%{text}',
                hovertemplate='%{y} - %{x}<br>Avg Gain: %{z:.1f}%<extra></extra>'
            ),
            row=2, col=1
        )

        # 4. Risk/Reward Distribution
        risk_reward_by_outcome = []
        for outcome in df['outcome_class'].unique():
            outcome_df = df[df['outcome_class'] == outcome]
            if len(outcome_df) > 0:
                risk_reward_by_outcome.append(
                    go.Box(
                        y=outcome_df['max_gain_40d'] * 100,
                        name=outcome.split('_')[0],
                        boxpoints='outliers',
                        marker_color=['red' if 'Failed' in outcome else
                                    'orange' if 'Stagnant' in outcome else
                                    'yellow' if 'Minimal' in outcome else
                                    'lightgreen' if 'Quality' in outcome else
                                    'green' if 'Strong' in outcome else
                                    'darkgreen'][0]
                    )
                )

        for trace in risk_reward_by_outcome:
            fig.add_trace(trace, row=2, col=2)

        # Update layout
        fig.update_layout(
            title={
                'text': '<b>Pattern Success Recipe Analysis</b><br>' +
                       '<sub>Discovering winning feature combinations</sub>',
                'font': {'size': 20},
                'x': 0.5,
                'xanchor': 'center'
            },
            height=900,
            showlegend=False
        )

        # Update axes
        fig.update_xaxes(title_text="Expected Value", row=1, col=1)
        fig.update_xaxes(title_text="Success Rate (%)", row=1, col=2)
        fig.update_yaxes(title_text="40-Day Gain (%)", row=1, col=2)
        fig.update_yaxes(title_text="40-Day Gain (%)", row=2, col=2)
        fig.update_xaxes(title_text="Outcome Class", row=2, col=2)

        # Save
        fig.write_html(save_path)
        print(f"\n[OK] Success recipe analysis saved to: {save_path}")

        # Print top recipes
        print("\n" + "="*60)
        print("TOP 5 SUCCESS RECIPES:")
        print("="*60)
        for idx, row in combo_df.head(5).iterrows():
            print(f"\n{idx+1}. {row['volatility']} + {row['volume']} + {row['agreement']}")
            print(f"   Expected Value: {row['expected_value']:.2f}")
            print(f"   Success Rate: {row['success_rate']:.1f}%")
            print(f"   Avg 40d Gain: {row['avg_gain_40d']:.1f}%")
            print(f"   Risk/Reward: {row['risk_reward']:.2f}")
            print(f"   Sample Size: {row['count']} patterns")

        return fig

    def create_interactive_explorer(self, save_path: str = "pattern_dna_explorer.html"):
        """Create the complete interactive Pattern DNA Explorer dashboard"""

        if self.enriched_df is None:
            self.engineer_features()

        # Create comprehensive dashboard
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Pattern DNA Sunburst',
                'Success Recipe Matrix',
                'Feature Impact Analysis',
                'Outcome Distribution',
                'Volatility DNA Map',
                'Volume DNA Map',
                'Method Agreement DNA',
                'Temporal DNA Patterns',
                'Risk Profile DNA'
            ),
            specs=[
                [{"type": "sunburst", "colspan": 2}, None, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "heatmap"}, {"type": "box"}],
                [{"type": "bar"}, {"type": "scatter"}, {"type": "table"}]
            ],
            column_widths=[0.4, 0.3, 0.3],
            row_heights=[0.4, 0.3, 0.3],
            horizontal_spacing=0.1,
            vertical_spacing=0.1
        )

        df = self.enriched_df

        # Main sunburst (spanning 2 columns)
        sunburst_data = self.prepare_sunburst_data('outcome_class', 'volatility_category', 'volume_category')

        fig.add_trace(
            go.Sunburst(
                labels=sunburst_data['labels'],
                parents=sunburst_data['parents'],
                values=sunburst_data['values'],
                branchvalues="total",
                marker=dict(
                    colors=sunburst_data['gain'],
                    colorscale='RdYlGn',
                    cmid=0,
                    colorbar=dict(title="Gain %", x=0.45)
                ),
                text=sunburst_data['text'],
                hovertemplate='<b>%{label}</b><br>%{text}<extra></extra>'
            ),
            row=1, col=1
        )

        # Outcome distribution bar chart
        outcome_stats = df.groupby('outcome_class').agg({
            'max_gain_40d': 'mean',
            'strategic_value': 'mean',
            'ticker': 'count'
        }).reset_index()
        outcome_stats.columns = ['outcome', 'avg_gain', 'avg_value', 'count']
        outcome_stats['avg_gain'] = outcome_stats['avg_gain'] * 100

        fig.add_trace(
            go.Bar(
                x=outcome_stats['outcome'].str.split('_').str[0],
                y=outcome_stats['count'],
                marker_color=outcome_stats['avg_value'],
                marker_colorscale='RdYlGn',
                text=[f'{c}<br>{g:.1f}%' for c, g in zip(outcome_stats['count'], outcome_stats['avg_gain'])],
                hovertemplate='<b>%{x}</b><br>Count: %{y}<br>Avg Gain: %{text}<extra></extra>'
            ),
            row=1, col=3
        )

        # Add explanatory text
        fig.add_annotation(
            text="<b>How to Read This Dashboard:</b><br>" +
                 "• <b>Sunburst:</b> Click segments to zoom in/out<br>" +
                 "• <b>Colors:</b> Green = High gains, Red = Losses<br>" +
                 "• <b>Size:</b> Number of patterns in category<br>" +
                 "• <b>Hover:</b> For detailed statistics",
            xref="paper", yref="paper",
            x=0.02, y=-0.08,
            showarrow=False,
            font=dict(size=11),
            align="left",
            bgcolor="lightyellow",
            bordercolor="gray",
            borderwidth=1
        )

        # Update layout
        fig.update_layout(
            title={
                'text': '<b>Pattern DNA Explorer</b><br>' +
                       '<sub>Interactive exploration of consolidation pattern anatomy</sub>',
                'font': {'size': 22},
                'x': 0.5,
                'xanchor': 'center'
            },
            height=1200,
            showlegend=False
        )

        # Save dashboard
        fig.write_html(save_path)
        print(f"\n[OK] Pattern DNA Explorer saved to: {save_path}")

        # Auto-open in browser
        webbrowser.open(f"file://{os.path.abspath(save_path)}")
        print("[OK] Opening Pattern DNA Explorer in browser...")

        return fig

    def prepare_sunburst_data(self, inner: str, middle: str, outer: str) -> pd.DataFrame:
        """Prepare data for sunburst visualization"""

        df = self.enriched_df
        sunburst_data = []

        for inner_val in df[inner].unique():
            if pd.isna(inner_val):
                continue

            inner_df = df[df[inner] == inner_val]
            inner_gain = inner_df['max_gain_40d'].mean() * 100
            inner_count = len(inner_df)

            sunburst_data.append({
                'labels': inner_val.split('_')[0] if '_' in str(inner_val) else str(inner_val),
                'parents': '',
                'values': inner_count,
                'gain': inner_gain,
                'text': f'N={inner_count}, Gain={inner_gain:.1f}%'
            })

            for middle_val in inner_df[middle].unique():
                if pd.isna(middle_val):
                    continue

                middle_df = inner_df[inner_df[middle] == middle_val]
                middle_gain = middle_df['max_gain_40d'].mean() * 100
                middle_count = len(middle_df)
                middle_label = str(middle_val).split('_')[0] if '_' in str(middle_val) else str(middle_val)

                sunburst_data.append({
                    'labels': middle_label,
                    'parents': inner_val.split('_')[0] if '_' in str(inner_val) else str(inner_val),
                    'values': middle_count,
                    'gain': middle_gain,
                    'text': f'N={middle_count}, Gain={middle_gain:.1f}%'
                })

                for outer_val in middle_df[outer].unique():
                    if pd.isna(outer_val):
                        continue

                    outer_df = middle_df[middle_df[outer] == outer_val]
                    outer_gain = outer_df['max_gain_40d'].mean() * 100
                    outer_count = len(outer_df)
                    outer_label = str(outer_val).split('_')[0] if '_' in str(outer_val) else str(outer_val)

                    sunburst_data.append({
                        'labels': outer_label,
                        'parents': middle_label,
                        'values': outer_count,
                        'gain': outer_gain,
                        'text': f'N={outer_count}, Gain={outer_gain:.1f}%'
                    })

        return pd.DataFrame(sunburst_data)


def main():
    """Run Pattern DNA Explorer"""
    import argparse

    parser = argparse.ArgumentParser(description='Pattern DNA Explorer - Dissect consolidation patterns')
    parser.add_argument('file', help='Path to parquet file')
    parser.add_argument('--output-dir', default='.', help='Output directory')
    parser.add_argument('--auto-open', action='store_true', help='Auto-open in browser')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("PATTERN DNA EXPLORER")
    print("="*60)

    # Initialize explorer
    explorer = PatternDNAExplorer()

    # Load data
    explorer.load_parquet(args.file)

    # Classify outcomes
    explorer.classify_outcomes()

    # Engineer features
    explorer.engineer_features()

    # Create visualizations
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nCreating visualizations...")

    # 1. Main sunburst DNA chart
    explorer.create_sunburst_dna(
        inner_feature='outcome_class',
        middle_feature='volatility_category',
        outer_feature='volume_category',
        save_path=str(output_dir / 'pattern_dna_sunburst.html')
    )

    # 2. Success recipe analysis
    explorer.create_success_recipe_analysis(
        save_path=str(output_dir / 'pattern_success_recipes.html')
    )

    # 3. Complete interactive explorer
    explorer.create_interactive_explorer(
        save_path=str(output_dir / 'pattern_dna_explorer_full.html')
    )

    print("\n" + "="*60)
    print("Pattern DNA Explorer Complete!")
    print("="*60)
    print(f"\nFiles created in {output_dir}:")
    print("  - pattern_dna_sunburst.html")
    print("  - pattern_success_recipes.html")
    print("  - pattern_dna_explorer_full.html")


if __name__ == "__main__":
    main()