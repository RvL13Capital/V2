"""
Enhanced Interactive Dashboard for Consolidation Pattern Analysis
Provides comprehensive visualization with explanations and insights
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from datetime import datetime
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import webbrowser
import os


class ConsolidationDashboard:
    """Interactive dashboard for consolidation pattern analysis"""

    def __init__(self):
        self.df = None
        self.dashboard = None

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
        print(f"[OK] Loaded {len(df)} rows from {Path(file_path).name}")
        print(f"[OK] Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"[OK] Tickers: {', '.join(df['ticker'].unique())}")

        return df

    def create_dashboard(self, save_path: str = "consolidation_dashboard.html"):
        """Create comprehensive interactive dashboard with explanations"""

        if self.df is None:
            raise ValueError("No data loaded. Call load_parquet first.")

        # Create figure with custom layout
        fig = make_subplots(
            rows=4, cols=3,
            subplot_titles=(
                '<b>1. Method Success Rates</b><br><sub>% of signals with >10% gain in 20 days</sub>',
                '<b>2. Average Returns Comparison</b><br><sub>20-day vs 40-day average gains</sub>',
                '<b>3. Signal Distribution</b><br><sub>Frequency of consolidation signals</sub>',
                '<b>4. Risk/Reward Analysis</b><br><sub>Gain/Loss ratio (>1 is profitable)</sub>',
                '<b>5. Performance Scatter</b><br><sub>20d vs 40d gains correlation</sub>',
                '<b>6. Method Agreement Impact</b><br><sub>Performance when methods agree</sub>',
                '<b>7. Volatility vs Returns</b><br><sub>BBW impact on performance</sub>',
                '<b>8. Volume Analysis</b><br><sub>Volume ratio effect on gains</sub>',
                '<b>9. Top Performing Patterns</b><br><sub>Best consolidation breakouts</sub>',
                '<b>10. Temporal Patterns</b><br><sub>Performance over time</sub>',
                '<b>11. Method Correlation</b><br><sub>How methods overlap</sub>',
                '<b>12. Key Metrics Summary</b><br><sub>Overall statistics</sub>'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}, {"type": "pie"}],
                [{"type": "bar"}, {"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "heatmap"}, {"type": "table"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.12,
            column_widths=[0.33, 0.33, 0.34],
            row_heights=[0.25, 0.25, 0.25, 0.25]
        )

        # Color scheme
        method_colors = {
            'method1_bollinger': '#1f77b4',
            'method2_range_based': '#ff7f0e',
            'method3_volume_weighted': '#2ca02c',
            'method4_atr_based': '#d62728'
        }

        method_names = {
            'method1_bollinger': 'Bollinger',
            'method2_range_based': 'Range-Based',
            'method3_volume_weighted': 'Volume-Weighted',
            'method4_atr_based': 'ATR-Based'
        }

        # === CHART 1: Success Rates ===
        success_data = []
        for method_col, method_name in method_names.items():
            consolidated = self.df[self.df[method_col] == True]
            if len(consolidated) > 0:
                success_rate = (consolidated['max_gain_20d'] > 0.1).mean() * 100
                total_signals = len(consolidated)
                success_data.append({
                    'method': method_name,
                    'success_rate': success_rate,
                    'total': total_signals,
                    'color': method_colors[method_col]
                })

        if success_data:
            success_df = pd.DataFrame(success_data)
            fig.add_trace(
                go.Bar(
                    x=success_df['method'],
                    y=success_df['success_rate'],
                    marker_color=success_df['color'],
                    text=[f'{r:.1f}%<br>({t} signals)' for r, t in
                          zip(success_df['success_rate'], success_df['total'])],
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Success Rate: %{y:.1f}%<br>Total Signals: %{text}<extra></extra>'
                ),
                row=1, col=1
            )

        # === CHART 2: Average Returns ===
        returns_data = []
        for method_col, method_name in method_names.items():
            consolidated = self.df[self.df[method_col] == True]
            if len(consolidated) > 0:
                returns_data.append({
                    'method': method_name,
                    '20-day': consolidated['max_gain_20d'].mean() * 100,
                    '40-day': consolidated['max_gain_40d'].mean() * 100
                })

        for period, color in [('20-day', 'lightblue'), ('40-day', 'darkblue')]:
            values = [d[period] for d in returns_data]
            methods = [d['method'] for d in returns_data]
            fig.add_trace(
                go.Bar(
                    name=period,
                    x=methods,
                    y=values,
                    text=[f'{v:.1f}%' for v in values],
                    textposition='outside',
                    marker_color=color,
                    hovertemplate='<b>%{x}</b><br>' + period + ' Gain: %{y:.1f}%<extra></extra>'
                ),
                row=1, col=2
            )

        # === CHART 3: Signal Distribution ===
        signal_counts = []
        for method_col, method_name in method_names.items():
            count = self.df[method_col].sum()
            signal_counts.append({'method': method_name, 'count': count})

        signal_df = pd.DataFrame(signal_counts)
        fig.add_trace(
            go.Pie(
                labels=signal_df['method'],
                values=signal_df['count'],
                marker=dict(colors=[method_colors[k] for k in method_names.keys()]),
                textinfo='label+percent',
                hovertemplate='<b>%{label}</b><br>Signals: %{value}<br>Percentage: %{percent}<extra></extra>'
            ),
            row=1, col=3
        )

        # === CHART 4: Risk/Reward ===
        rr_data = []
        for method_col, method_name in method_names.items():
            consolidated = self.df[self.df[method_col] == True]
            if len(consolidated) > 0:
                avg_gain = consolidated['max_gain_40d'].mean()
                avg_loss = abs(consolidated['max_loss_40d'].mean())
                rr = avg_gain / avg_loss if avg_loss > 0 else 0
                rr_data.append({
                    'method': method_name,
                    'ratio': rr,
                    'color': 'green' if rr > 1 else 'red'
                })

        if rr_data:
            rr_df = pd.DataFrame(rr_data)
            fig.add_trace(
                go.Bar(
                    x=rr_df['method'],
                    y=rr_df['ratio'],
                    marker_color=rr_df['color'],
                    text=[f'{r:.2f}' for r in rr_df['ratio']],
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Risk/Reward: %{y:.2f}<br>%{text}<extra></extra>'
                ),
                row=2, col=1
            )

            # Add break-even line (only for bar chart, not pie)
            fig.add_shape(type="line",
                         x0=-0.5, x1=len(rr_df)-0.5,
                         y0=1, y1=1,
                         line=dict(color="gray", dash="dash"),
                         row=2, col=1)

        # === CHART 5: Performance Scatter ===
        # Sample data for performance
        sample_size = min(1000, len(self.df))
        sample_df = self.df.sample(sample_size)

        for method_col, method_name in method_names.items():
            method_data = sample_df[sample_df[method_col] == True]
            if len(method_data) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=method_data['max_gain_20d'] * 100,
                        y=method_data['max_gain_40d'] * 100,
                        mode='markers',
                        name=method_name,
                        marker=dict(
                            color=method_colors[method_col],
                            size=6,
                            opacity=0.6
                        ),
                        hovertemplate='<b>' + method_name + '</b><br>' +
                                     'Ticker: %{customdata[0]}<br>' +
                                     'Date: %{customdata[1]}<br>' +
                                     '20d Gain: %{x:.1f}%<br>' +
                                     '40d Gain: %{y:.1f}%<extra></extra>',
                        customdata=np.column_stack((method_data['ticker'],
                                                   method_data['date'].dt.strftime('%Y-%m-%d')))
                    ),
                    row=2, col=2
                )

        # Add diagonal reference line
        fig.add_shape(type="line", x0=0, y0=0, x1=100, y1=100,
                     line=dict(color="gray", dash="dash", width=1),
                     row=2, col=2)

        # === CHART 6: Method Agreement ===
        method_cols = ['method1_bollinger', 'method2_range_based',
                      'method3_volume_weighted', 'method4_atr_based']
        self.df['methods_agreed'] = self.df[method_cols].sum(axis=1)

        agreement_data = []
        for num in range(1, 5):
            agreed_df = self.df[self.df['methods_agreed'] == num]
            if len(agreed_df) > 0:
                agreement_data.append({
                    'Agreement': f'{num} Method{"s" if num > 1 else ""}',
                    'Count': len(agreed_df),
                    'Avg_20d': agreed_df['max_gain_20d'].mean() * 100,
                    'Avg_40d': agreed_df['max_gain_40d'].mean() * 100
                })

        if agreement_data:
            agreement_df = pd.DataFrame(agreement_data)
            fig.add_trace(
                go.Bar(
                    x=agreement_df['Agreement'],
                    y=agreement_df['Avg_40d'],
                    marker_color=['#ffd92f', '#8da0cb', '#fc8d62', '#66c2a5'][:len(agreement_df)],
                    text=[f'{g:.1f}%<br>({c} signals)' for g, c in
                          zip(agreement_df['Avg_40d'], agreement_df['Count'])],
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>40d Gain: %{y:.1f}%<br>Count: %{text}<extra></extra>'
                ),
                row=2, col=3
            )

        # === CHART 7: Volatility Analysis (BBW) ===
        fig.add_trace(
            go.Scatter(
                x=sample_df['bbw'],
                y=sample_df['max_gain_40d'] * 100,
                mode='markers',
                marker=dict(
                    color=sample_df['max_gain_40d'] * 100,
                    colorscale='RdYlGn',
                    size=5,
                    showscale=True,
                    colorbar=dict(
                        title="Gain %",
                        x=0.37,
                        y=0.37,
                        len=0.2,
                        thickness=15
                    ),
                    cmin=-20,
                    cmax=50
                ),
                text=[f'Ticker: {t}<br>Date: {d}' for t, d in
                     zip(sample_df['ticker'], sample_df['date'].dt.strftime('%Y-%m-%d'))],
                hovertemplate='BBW: %{x:.4f}<br>40d Gain: %{y:.1f}%<br>%{text}<extra></extra>',
                showlegend=False
            ),
            row=3, col=1
        )

        # Add target line
        fig.add_shape(type="line",
                     x0=sample_df['bbw'].min(), x1=sample_df['bbw'].max(),
                     y0=20, y1=20,
                     line=dict(color="green", dash="dash"),
                     row=3, col=1)

        # === CHART 8: Volume Analysis ===
        fig.add_trace(
            go.Scatter(
                x=sample_df['volume_ratio'],
                y=sample_df['max_gain_40d'] * 100,
                mode='markers',
                marker=dict(
                    color=sample_df['atr_pct'] * 100,
                    colorscale='Viridis',
                    size=5,
                    showscale=True,
                    colorbar=dict(
                        title="ATR %",
                        x=0.71,
                        y=0.37,
                        len=0.2,
                        thickness=15
                    )
                ),
                text=[f'Ticker: {t}<br>ATR%: {a:.2f}' for t, a in
                     zip(sample_df['ticker'], sample_df['atr_pct'] * 100)],
                hovertemplate='Volume Ratio: %{x:.2f}<br>40d Gain: %{y:.1f}%<br>%{text}<extra></extra>',
                showlegend=False
            ),
            row=3, col=2
        )

        # === CHART 9: Top Patterns ===
        top_patterns = self.df.nlargest(15, 'max_gain_40d')
        fig.add_trace(
            go.Bar(
                y=[f"{t} {d.strftime('%m/%d')}" for t, d in
                   zip(top_patterns['ticker'], top_patterns['date'])],
                x=top_patterns['max_gain_40d'] * 100,
                orientation='h',
                marker_color='green',
                text=[f'{g:.1f}%' for g in top_patterns['max_gain_40d'] * 100],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>40d Gain: %{x:.1f}%<extra></extra>'
            ),
            row=3, col=3
        )

        # === CHART 10: Temporal Analysis ===
        # Group by week for cleaner visualization
        weekly_data = self.df.copy()
        weekly_data['week'] = weekly_data['date'].dt.to_period('W')
        weekly_stats = weekly_data.groupby('week').agg({
            'max_gain_40d': 'mean',
            'method1_bollinger': 'sum'
        }).reset_index()

        weekly_stats['week_str'] = weekly_stats['week'].astype(str)

        fig.add_trace(
            go.Scatter(
                x=weekly_stats['week_str'],
                y=weekly_stats['max_gain_40d'] * 100,
                mode='lines+markers',
                name='Weekly Avg',
                line=dict(color='blue', width=2),
                marker=dict(size=6),
                hovertemplate='Week: %{x}<br>Avg Gain: %{y:.1f}%<extra></extra>'
            ),
            row=4, col=1
        )

        # === CHART 11: Method Correlation Heatmap ===
        correlation_matrix = []
        for method1 in method_names.keys():
            row = []
            for method2 in method_names.keys():
                overlap = (self.df[method1] & self.df[method2]).sum()
                total = self.df[method1].sum()
                correlation = overlap / total if total > 0 else 0
                row.append(correlation)
            correlation_matrix.append(row)

        fig.add_trace(
            go.Heatmap(
                z=correlation_matrix,
                x=list(method_names.values()),
                y=list(method_names.values()),
                colorscale='Blues',
                text=[[f'{val:.2f}' for val in row] for row in correlation_matrix],
                texttemplate='%{text}',
                textfont={"size": 10},
                hovertemplate='%{y} & %{x}<br>Overlap: %{z:.2f}<extra></extra>',
                showscale=False
            ),
            row=4, col=2
        )

        # === CHART 12: Summary Statistics Table ===
        total_signals = sum(self.df[col].sum() for col in method_names.keys())
        best_pattern = self.df.loc[self.df['max_gain_40d'].idxmax()]

        summary_data = [
            ['Metric', 'Value'],
            ['Total Records', f'{len(self.df):,}'],
            ['Date Range', f"{self.df['date'].min().date()} to {self.df['date'].max().date()}"],
            ['Total Signals', f'{total_signals:,}'],
            ['Unique Tickers', f'{self.df["ticker"].nunique()}'],
            ['Avg 20d Gain', f"{self.df['max_gain_20d'].mean() * 100:.1f}%"],
            ['Avg 40d Gain', f"{self.df['max_gain_40d'].mean() * 100:.1f}%"],
            ['Best Pattern', f"{best_pattern['ticker']} ({best_pattern['max_gain_40d']*100:.1f}%)"],
            ['Success Rate', f"{(self.df['max_gain_40d'] > 0.2).mean() * 100:.1f}%"]
        ]

        fig.add_trace(
            go.Table(
                header=dict(
                    values=['<b>Metric</b>', '<b>Value</b>'],
                    fill_color='lightblue',
                    align='left',
                    font=dict(size=12)
                ),
                cells=dict(
                    values=list(zip(*summary_data[1:])),
                    fill_color='white',
                    align='left',
                    font=dict(size=11),
                    height=25
                )
            ),
            row=4, col=3
        )

        # === UPDATE LAYOUT ===
        fig.update_layout(
            title={
                'text': '<b>Consolidation Pattern Analysis Dashboard</b><br>' +
                       '<sub>Interactive analysis of consolidation patterns and breakout performance</sub>',
                'font': {'size': 24},
                'x': 0.5,
                'xanchor': 'center'
            },
            showlegend=True,
            height=1400,
            hovermode='closest',
            template='plotly_white',
            margin=dict(t=100, b=50, l=50, r=50)
        )

        # Update axes labels
        fig.update_xaxes(title_text="Method", row=1, col=1)
        fig.update_yaxes(title_text="Success Rate (%)", row=1, col=1)

        fig.update_xaxes(title_text="Method", row=1, col=2)
        fig.update_yaxes(title_text="Average Gain (%)", row=1, col=2)

        fig.update_xaxes(title_text="Method", row=2, col=1)
        fig.update_yaxes(title_text="Risk/Reward Ratio", row=2, col=1)

        fig.update_xaxes(title_text="20-Day Gain (%)", row=2, col=2)
        fig.update_yaxes(title_text="40-Day Gain (%)", row=2, col=2)

        fig.update_xaxes(title_text="Agreement Level", row=2, col=3)
        fig.update_yaxes(title_text="Avg 40-Day Gain (%)", row=2, col=3)

        fig.update_xaxes(title_text="Bollinger Band Width", row=3, col=1)
        fig.update_yaxes(title_text="40-Day Gain (%)", row=3, col=1)

        fig.update_xaxes(title_text="Volume Ratio", row=3, col=2)
        fig.update_yaxes(title_text="40-Day Gain (%)", row=3, col=2)

        fig.update_xaxes(title_text="40-Day Gain (%)", row=3, col=3)
        fig.update_yaxes(title_text="Pattern", row=3, col=3)

        fig.update_xaxes(title_text="Week", row=4, col=1)
        fig.update_yaxes(title_text="Avg 40-Day Gain (%)", row=4, col=1)

        # Add annotations with explanations
        annotations = [
            dict(
                text="<b>How to Use This Dashboard:</b><br>" +
                     "• <b>Charts 1-3:</b> Compare detection methods effectiveness<br>" +
                     "• <b>Charts 4-6:</b> Analyze risk/reward and method interactions<br>" +
                     "• <b>Charts 7-8:</b> Understand volatility and volume impacts<br>" +
                     "• <b>Charts 9-10:</b> Identify best patterns and temporal trends<br>" +
                     "• <b>Chart 11:</b> See how methods overlap in detection<br>" +
                     "• <b>Chart 12:</b> Review overall statistics",
                xref="paper", yref="paper",
                x=0.02, y=-0.05,
                showarrow=False,
                font=dict(size=11),
                align="left",
                bgcolor="lightyellow",
                bordercolor="gray",
                borderwidth=1
            )
        ]

        fig.update_layout(annotations=annotations)

        # Save dashboard
        fig.write_html(save_path)
        self.dashboard = fig

        print(f"\n[OK] Dashboard created successfully!")
        print(f"[OK] Saved to: {save_path}")

        return fig

    def open_dashboard(self, file_path: str = "consolidation_dashboard.html"):
        """Open dashboard in web browser"""
        if os.path.exists(file_path):
            webbrowser.open(f"file://{os.path.abspath(file_path)}")
            print(f"[OK] Opening dashboard in browser...")
        else:
            print(f"[ERROR] Dashboard file not found: {file_path}")


def main():
    """Run dashboard creation and display"""
    import argparse

    parser = argparse.ArgumentParser(description='Create interactive consolidation pattern dashboard')
    parser.add_argument('file', help='Path to parquet file')
    parser.add_argument('--output', default='consolidation_dashboard.html',
                       help='Output HTML file name')
    parser.add_argument('--auto-open', action='store_true',
                       help='Automatically open dashboard in browser')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("CONSOLIDATION PATTERN INTERACTIVE DASHBOARD")
    print("="*60)

    # Create dashboard
    dashboard = ConsolidationDashboard()

    # Load data
    dashboard.load_parquet(args.file)

    # Create dashboard
    dashboard.create_dashboard(args.output)

    # Open if requested
    if args.auto_open:
        dashboard.open_dashboard(args.output)

    print("\n" + "="*60)
    print("Dashboard ready! Open the HTML file to view.")
    print("="*60)


if __name__ == "__main__":
    main()