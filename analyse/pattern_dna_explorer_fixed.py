"""
Pattern DNA Explorer - Fixed Version with Proper Visualization
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

        # 4. Method Agreement Categories
        method_cols = ['method1_bollinger', 'method2_range_based',
                      'method3_volume_weighted', 'method4_atr_based']
        df['methods_agreed'] = df[method_cols].sum(axis=1)
        df['agreement_category'] = df['methods_agreed'].apply(
            lambda x: f'{x}_Method{"s" if x != 1 else ""}_Agree'
        )

        self.enriched_df = df
        print(f"\n[OK] Engineered {len([col for col in df.columns if 'category' in col])} categorical features")
        return df

    def generate_insights(self, df: pd.DataFrame) -> Dict[str, str]:
        """Generate key insights from the data"""

        insights = {}

        # Calculate distribution percentages
        high_quality = ((df['outcome_class'] == 'K3_Strong').sum() +
                       (df['outcome_class'] == 'K4_Exceptional').sum())
        acceptable = (df['outcome_class'] == 'K2_Quality').sum()
        poor = ((df['outcome_class'] == 'K0_Stagnant').sum() +
               (df['outcome_class'] == 'K1_Minimal').sum() +
               (df['outcome_class'] == 'K5_Failed').sum())

        total = len(df)
        insights['high_quality_pct'] = round(high_quality / total * 100, 1)
        insights['acceptable_pct'] = round(acceptable / total * 100, 1)
        insights['poor_pct'] = round(poor / total * 100, 1)

        # Find best pattern
        best_idx = df['max_gain_40d'].idxmax()
        best_row = df.loc[best_idx]
        insights['best_pattern'] = f"{best_row['ticker']} ({best_row['max_gain_40d']*100:.1f}% gain)"

        # Find best recipe (feature combination)
        if 'volatility_category' in df.columns and 'volume_category' in df.columns:
            recipe_groups = df.groupby(['volatility_category', 'volume_category']).agg({
                'max_gain_40d': 'mean',
                'ticker': 'count'
            }).reset_index()
            recipe_groups = recipe_groups[recipe_groups['ticker'] >= 3]  # Min 3 samples

            if len(recipe_groups) > 0:
                best_recipe_idx = recipe_groups['max_gain_40d'].idxmax()
                best_recipe = recipe_groups.loc[best_recipe_idx]
                insights['best_recipe'] = (f"{best_recipe['volatility_category'].replace('_', ' ')} + "
                                          f"{best_recipe['volume_category'].replace('_', ' ')}")
            else:
                insights['best_recipe'] = "Insufficient data"
        else:
            insights['best_recipe'] = "Feature data unavailable"

        # Risk alert
        failure_rate = (df['outcome_class'] == 'K5_Failed').mean() * 100
        if failure_rate > 5:
            insights['risk_alert'] = f"High failure rate ({failure_rate:.1f}%)"
        elif failure_rate > 2:
            insights['risk_alert'] = f"Moderate failure rate ({failure_rate:.1f}%)"
        else:
            insights['risk_alert'] = f"Low failure rate ({failure_rate:.1f}%)"

        # Opportunity identification
        k4_count = (df['outcome_class'] == 'K4_Exceptional').sum()
        if k4_count > 0:
            insights['opportunity'] = f"{k4_count} exceptional patterns found (>75% gain)"
        else:
            k3_count = (df['outcome_class'] == 'K3_Strong').sum()
            insights['opportunity'] = f"{k3_count} strong patterns found (35-75% gain)"

        # Summary
        avg_gain = df['max_gain_40d'].mean() * 100
        success_rate = (df['max_gain_40d'] > 0.2).mean() * 100
        insights['summary'] = (f"Analyzed {len(df)} patterns | "
                              f"Avg gain: {avg_gain:.1f}% | "
                              f"Success rate (>20% gain): {success_rate:.1f}%")

        return insights

    def create_comprehensive_sunburst(self, save_path: str = "pattern_dna_complete.html"):
        """Create a complete Pattern DNA Explorer with all visualizations"""

        if self.enriched_df is None:
            self.engineer_features()

        df = self.enriched_df

        # Create figure with multiple subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                '<b>Pattern DNA Sunburst</b><br><sub>Click to explore hierarchies</sub>',
                '<b>Outcome Distribution</b><br><sub>Pattern performance classes</sub>',
                '<b>Top Success Recipes</b><br><sub>Best feature combinations</sub>',
                '<b>Volatility Impact</b><br><sub>BBW effect on gains</sub>',
                '<b>Volume Impact</b><br><sub>Volume ratio effect</sub>',
                '<b>Method Agreement</b><br><sub>Multi-method confirmation</sub>',
                '<b>Feature Correlation</b><br><sub>How features relate</sub>',
                '<b>Risk/Reward by Class</b><br><sub>Gain distributions</sub>',
                '<b>Success Metrics</b><br><sub>Key statistics</sub>'
            ),
            specs=[
                [{"type": "sunburst", "colspan": 2}, None, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "bar"}],
                [{"type": "heatmap"}, {"type": "box"}, {"type": "table"}]
            ],
            column_widths=[0.35, 0.35, 0.3],
            row_heights=[0.4, 0.3, 0.3],
            horizontal_spacing=0.08,
            vertical_spacing=0.12
        )

        # === 1. SUNBURST CHART ===
        sunburst_data = []

        # Build hierarchy: Outcome -> Volatility -> Volume
        for outcome in df['outcome_class'].unique():
            if pd.isna(outcome):
                continue

            outcome_df = df[df['outcome_class'] == outcome]
            outcome_gain = outcome_df['max_gain_40d'].mean() * 100
            outcome_count = len(outcome_df)

            # Simplify labels
            outcome_label = outcome.split('_')[0]  # K4, K3, etc.

            sunburst_data.append({
                'ids': outcome,
                'labels': outcome_label,
                'parents': '',
                'values': outcome_count,
                'colors': outcome_gain
            })

            if 'volatility_category' in df.columns:
                for vol in outcome_df['volatility_category'].unique():
                    if pd.isna(vol):
                        continue

                    vol_df = outcome_df[outcome_df['volatility_category'] == vol]
                    vol_gain = vol_df['max_gain_40d'].mean() * 100
                    vol_count = len(vol_df)
                    vol_label = str(vol).replace('_', ' ')

                    vol_id = f"{outcome}_{vol}"
                    sunburst_data.append({
                        'ids': vol_id,
                        'labels': vol_label,
                        'parents': outcome,
                        'values': vol_count,
                        'colors': vol_gain
                    })

                    if 'volume_category' in df.columns:
                        for volume in vol_df['volume_category'].unique():
                            if pd.isna(volume):
                                continue

                            volume_df = vol_df[vol_df['volume_category'] == volume]
                            volume_gain = volume_df['max_gain_40d'].mean() * 100
                            volume_count = len(volume_df)
                            volume_label = str(volume).replace('_', ' ')

                            sunburst_data.append({
                                'ids': f"{vol_id}_{volume}",
                                'labels': volume_label,
                                'parents': vol_id,
                                'values': volume_count,
                                'colors': volume_gain
                            })

        if sunburst_data:
            sunburst_df = pd.DataFrame(sunburst_data)

            fig.add_trace(
                go.Sunburst(
                    ids=sunburst_df['ids'],
                    labels=sunburst_df['labels'],
                    parents=sunburst_df['parents'],
                    values=sunburst_df['values'],
                    branchvalues="total",
                    marker=dict(
                        colors=sunburst_df['colors'],
                        colorscale='RdYlGn',
                        cmid=0,
                        cmin=-20,
                        cmax=50,
                        colorbar=dict(
                            title="Avg Gain %",
                            thickness=20,
                            x=1.02,  # Position to the right side
                            xanchor='left',
                            len=0.7,
                            y=0.5,
                            yanchor='middle'
                        )
                    ),
                    text=[f'Count: {v}<br>Gain: {c:.1f}%' for v, c in
                          zip(sunburst_df['values'], sunburst_df['colors'])],
                    hovertemplate='<b>%{label}</b><br>%{text}<extra></extra>',
                    textfont=dict(size=10)
                ),
                row=1, col=1
            )

        # === 2. OUTCOME DISTRIBUTION ===
        outcome_stats = df.groupby('outcome_class').agg({
            'max_gain_40d': 'mean',
            'ticker': 'count'
        }).reset_index()
        outcome_stats.columns = ['outcome', 'avg_gain', 'count']
        outcome_stats['avg_gain'] = outcome_stats['avg_gain'] * 100
        outcome_stats = outcome_stats.sort_values('avg_gain', ascending=True)

        fig.add_trace(
            go.Bar(
                x=outcome_stats['avg_gain'],
                y=outcome_stats['outcome'].str.replace('_', ' '),
                orientation='h',
                marker_color=outcome_stats['avg_gain'],
                marker_colorscale='RdYlGn',
                text=[f'{c} patterns<br>{g:.1f}%' for c, g in
                      zip(outcome_stats['count'], outcome_stats['avg_gain'])],
                textposition='auto',
                hovertemplate='<b>%{y}</b><br>Count: %{text}<br>Avg Gain: %{x:.1f}%<extra></extra>'
            ),
            row=1, col=3
        )

        # === 3. VOLATILITY SCATTER ===
        if 'bbw' in df.columns:
            sample_df = df.sample(min(500, len(df)))
            fig.add_trace(
                go.Scatter(
                    x=sample_df['bbw'],
                    y=sample_df['max_gain_40d'] * 100,
                    mode='markers',
                    marker=dict(
                        color=sample_df['max_gain_40d'] * 100,
                        colorscale='RdYlGn',
                        size=8,
                        opacity=0.6,
                        showscale=False
                    ),
                    text=[f'{t}<br>{d}' for t, d in
                          zip(sample_df['ticker'], sample_df['date'].dt.strftime('%Y-%m-%d'))],
                    hovertemplate='BBW: %{x:.4f}<br>40d Gain: %{y:.1f}%<br>%{text}<extra></extra>',
                    showlegend=False
                ),
                row=2, col=1
            )

        # === 4. VOLUME SCATTER ===
        if 'volume_ratio' in df.columns:
            sample_df = df.sample(min(500, len(df)))
            fig.add_trace(
                go.Scatter(
                    x=sample_df['volume_ratio'],
                    y=sample_df['max_gain_40d'] * 100,
                    mode='markers',
                    marker=dict(
                        color=sample_df['max_gain_40d'] * 100,
                        colorscale='Viridis',
                        size=8,
                        opacity=0.6,
                        showscale=False
                    ),
                    hovertemplate='Volume Ratio: %{x:.2f}<br>40d Gain: %{y:.1f}%<extra></extra>',
                    showlegend=False
                ),
                row=2, col=2
            )

        # === 5. METHOD AGREEMENT ===
        agreement_stats = df.groupby('agreement_category').agg({
            'max_gain_40d': 'mean',
            'ticker': 'count'
        }).reset_index()
        agreement_stats.columns = ['agreement', 'avg_gain', 'count']
        agreement_stats['avg_gain'] = agreement_stats['avg_gain'] * 100
        agreement_stats = agreement_stats.sort_values('agreement')

        fig.add_trace(
            go.Bar(
                x=agreement_stats['agreement'].str.replace('_', ' '),
                y=agreement_stats['avg_gain'],
                marker_color=agreement_stats['avg_gain'],
                marker_colorscale='Tealgrn',
                text=[f'{c}<br>{g:.1f}%' for c, g in
                      zip(agreement_stats['count'], agreement_stats['avg_gain'])],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Avg Gain: %{y:.1f}%<br>Count: %{text}<extra></extra>'
            ),
            row=2, col=3
        )

        # === 6. FEATURE CORRELATION HEATMAP ===
        # Calculate correlation between categorical features and gains
        feature_impact = []

        for feature in ['volatility_category', 'volume_category', 'agreement_category']:
            if feature in df.columns:
                for value in df[feature].unique():
                    if pd.isna(value):
                        continue
                    subset = df[df[feature] == value]
                    feature_impact.append({
                        'feature': feature.replace('_category', ''),
                        'value': str(value).split('_')[0],
                        'avg_gain': subset['max_gain_40d'].mean() * 100,
                        'count': len(subset)
                    })

        if feature_impact:
            impact_df = pd.DataFrame(feature_impact)
            pivot_impact = impact_df.pivot_table(
                index='feature',
                columns='value',
                values='avg_gain',
                aggfunc='mean'
            )

            fig.add_trace(
                go.Heatmap(
                    z=pivot_impact.values,
                    x=pivot_impact.columns,
                    y=pivot_impact.index,
                    colorscale='RdYlGn',
                    text=np.round(pivot_impact.values, 1),
                    texttemplate='%{text}%',
                    textfont={"size": 10},
                    hovertemplate='%{y} - %{x}<br>Avg Gain: %{z:.1f}%<extra></extra>',
                    showscale=False
                ),
                row=3, col=1
            )

        # === 7. OUTCOME BOX PLOTS ===
        box_data = []
        colors = {'K4': 'darkgreen', 'K3': 'green', 'K2': 'lightgreen',
                 'K1': 'yellow', 'K0': 'orange', 'K5': 'red'}

        for outcome in ['K4_Exceptional', 'K3_Strong', 'K2_Quality',
                       'K1_Minimal', 'K0_Stagnant', 'K5_Failed']:
            outcome_df = df[df['outcome_class'] == outcome]
            if len(outcome_df) > 0:
                box_data.append(
                    go.Box(
                        y=outcome_df['max_gain_40d'] * 100,
                        name=outcome.split('_')[0],
                        marker_color=colors.get(outcome.split('_')[0], 'gray'),
                        boxpoints='outliers'
                    )
                )

        for trace in box_data:
            fig.add_trace(trace, row=3, col=2)

        # === 8. SUMMARY TABLE ===
        # Calculate key metrics
        best_recipe = None
        if 'volatility_category' in df.columns and 'volume_category' in df.columns:
            recipe_groups = df.groupby(['volatility_category', 'volume_category']).agg({
                'max_gain_40d': 'mean',
                'ticker': 'count'
            }).reset_index()
            recipe_groups = recipe_groups[recipe_groups['ticker'] >= 5]  # Min 5 samples
            if len(recipe_groups) > 0:
                best_idx = recipe_groups['max_gain_40d'].idxmax()
                best = recipe_groups.loc[best_idx]
                best_recipe = f"{best['volatility_category']} + {best['volume_category']}"

        summary_data = [
            ['Metric', 'Value'],
            ['Total Patterns', f'{len(df):,}'],
            ['Exceptional (K4)', f'{(df["outcome_class"] == "K4_Exceptional").sum()}'],
            ['Strong (K3)', f'{(df["outcome_class"] == "K3_Strong").sum()}'],
            ['Quality (K2)', f'{(df["outcome_class"] == "K2_Quality").sum()}'],
            ['Failed (K5)', f'{(df["outcome_class"] == "K5_Failed").sum()}'],
            ['Avg 40d Gain', f'{df["max_gain_40d"].mean() * 100:.1f}%'],
            ['Best Recipe', best_recipe if best_recipe else 'N/A'],
            ['Success Rate', f'{(df["max_gain_40d"] > 0.2).mean() * 100:.1f}%']
        ]

        fig.add_trace(
            go.Table(
                header=dict(
                    values=['<b>Metric</b>', '<b>Value</b>'],
                    fill_color='lightblue',
                    align='left'
                ),
                cells=dict(
                    values=list(zip(*summary_data[1:])),
                    fill_color='white',
                    align='left'
                )
            ),
            row=3, col=3
        )

        # Add key insights as annotations
        insights = self.generate_insights(df)

        # Update layout with insights
        fig.update_layout(
            title={
                'text': '<b>Pattern DNA Explorer - Complete Analysis</b><br>' +
                       '<sub>Interactive exploration of consolidation pattern anatomy</sub><br>' +
                       f'<sub style="color: gray; font-size: 12px">{insights["summary"]}</sub>',
                'font': {'size': 20},
                'x': 0.5,
                'xanchor': 'center'
            },
            height=1100,
            showlegend=False,
            template='plotly_white',
            margin=dict(t=120, b=150, l=80, r=150),  # More space for color bar
            annotations=[
                # Add insight boxes
                dict(
                    text=f"<b>KEY INSIGHTS:</b><br>" +
                         f"• Best Pattern: {insights['best_pattern']}<br>" +
                         f"• Success Recipe: {insights['best_recipe']}<br>" +
                         f"• Risk Alert: {insights['risk_alert']}<br>" +
                         f"• Opportunity: {insights['opportunity']}",
                    xref="paper", yref="paper",
                    x=0.02, y=-0.12,
                    showarrow=False,
                    font=dict(size=11),
                    align="left",
                    bgcolor="lightyellow",
                    bordercolor="orange",
                    borderwidth=2
                ),
                dict(
                    text=f"<b>PATTERN DISTRIBUTION:</b><br>" +
                         f"• High Quality (K3+K4): {insights['high_quality_pct']}%<br>" +
                         f"• Acceptable (K2): {insights['acceptable_pct']}%<br>" +
                         f"• Poor (K0+K1+K5): {insights['poor_pct']}%",
                    xref="paper", yref="paper",
                    x=0.35, y=-0.12,
                    showarrow=False,
                    font=dict(size=11),
                    align="left",
                    bgcolor="lightgreen",
                    bordercolor="green",
                    borderwidth=2
                ),
                dict(
                    text=f"<b>NAVIGATION GUIDE:</b><br>" +
                         "• Click sunburst segments to zoom<br>" +
                         "• Hover for detailed statistics<br>" +
                         "• Green = Profitable, Red = Loss",
                    xref="paper", yref="paper",
                    x=0.65, y=-0.12,
                    showarrow=False,
                    font=dict(size=11),
                    align="left",
                    bgcolor="lightblue",
                    bordercolor="blue",
                    borderwidth=2
                )
            ]
        )

        # Update axes
        fig.update_xaxes(title_text="40-Day Gain (%)", row=1, col=3)
        fig.update_xaxes(title_text="BBW", row=2, col=1)
        fig.update_yaxes(title_text="40-Day Gain (%)", row=2, col=1)
        fig.update_xaxes(title_text="Volume Ratio", row=2, col=2)
        fig.update_yaxes(title_text="40-Day Gain (%)", row=2, col=2)
        fig.update_xaxes(title_text="Agreement", row=2, col=3)
        fig.update_yaxes(title_text="Avg 40-Day Gain (%)", row=2, col=3)
        fig.update_yaxes(title_text="40-Day Gain (%)", row=3, col=2)

        # Save and display
        fig.write_html(save_path)
        print(f"\n[OK] Complete Pattern DNA Explorer saved to: {save_path}")

        # Auto-open
        webbrowser.open(f"file://{os.path.abspath(save_path)}")
        print("[OK] Opening in browser...")

        return fig


def main():
    """Run Pattern DNA Explorer"""
    import argparse

    parser = argparse.ArgumentParser(description='Pattern DNA Explorer - Fixed Version')
    parser.add_argument('file', help='Path to parquet file')
    parser.add_argument('--output', default='pattern_dna_complete.html',
                       help='Output HTML file')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("PATTERN DNA EXPLORER - FIXED VERSION")
    print("="*60)

    # Initialize explorer
    explorer = PatternDNAExplorer()

    # Load data
    explorer.load_parquet(args.file)

    # Classify outcomes
    explorer.classify_outcomes()

    # Engineer features
    explorer.engineer_features()

    # Create comprehensive visualization
    explorer.create_comprehensive_sunburst(args.output)

    print("\n" + "="*60)
    print("Pattern DNA Explorer Complete!")
    print("="*60)


if __name__ == "__main__":
    main()