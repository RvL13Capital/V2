"""
Visualization module for consolidation pattern analysis from parquet files
Creates both static matplotlib and interactive plotly visualizations
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

# Visualization imports
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.gridspec import GridSpec

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    print("Warning: Plotly not installed. Interactive visualizations will be limited.")
    print("Install with: pip install plotly")
    PLOTLY_AVAILABLE = False


class ConsolidationVisualizer:
    """Create visualizations for consolidation pattern data"""

    def __init__(self):
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        self.df = None

    def load_parquet(self, file_path: str) -> pd.DataFrame:
        """Load parquet file with proper date handling"""
        # Read with pyarrow
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
        return df

    def create_method_comparison_plot(self, save_path: Optional[str] = None):
        """Create comprehensive method comparison visualization"""

        if self.df is None:
            raise ValueError("No data loaded. Call load_parquet first.")

        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Method names mapping
        methods = {
            'method1_bollinger': 'Bollinger',
            'method2_range_based': 'Range-Based',
            'method3_volume_weighted': 'Volume-Weighted',
            'method4_atr_based': 'ATR-Based'
        }

        # 1. Success Rate Comparison (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        success_rates = []
        method_names = []

        for method_col, method_name in methods.items():
            consolidated = self.df[self.df[method_col] == True]
            if len(consolidated) > 0:
                success_rate = (consolidated['max_gain_20d'] > 0.1).mean() * 100
                success_rates.append(success_rate)
                method_names.append(method_name)

        bars = ax1.bar(method_names, success_rates, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax1.set_title('Success Rate (>10% gain in 20d)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_ylim(0, max(success_rates) * 1.2 if success_rates else 100)

        # Add value labels on bars
        for bar, rate in zip(bars, success_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)

        # 2. Average Gains Distribution (Top Middle)
        ax2 = fig.add_subplot(gs[0, 1])
        gains_20d = []
        gains_40d = []
        labels = []

        for method_col, method_name in methods.items():
            consolidated = self.df[self.df[method_col] == True]
            if len(consolidated) > 0:
                gains_20d.append(consolidated['max_gain_20d'].mean() * 100)
                gains_40d.append(consolidated['max_gain_40d'].mean() * 100)
                labels.append(method_name)

        if labels:
            x = np.arange(len(labels))
            width = 0.35
            bars1 = ax2.bar(x - width/2, gains_20d, width, label='20-day', color='skyblue')
            bars2 = ax2.bar(x + width/2, gains_40d, width, label='40-day', color='lightcoral')

            ax2.set_xlabel('Method')
            ax2.set_ylabel('Average Gain (%)')
            ax2.set_title('Average Gains Comparison', fontsize=12, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels, rotation=45, ha='right')
            ax2.legend()

            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                            f'{height:.1f}', ha='center', va='bottom', fontsize=9)

        # 3. Signal Frequency (Top Right)
        ax3 = fig.add_subplot(gs[0, 2])
        signal_counts = []
        signal_rates = []

        for method_col, method_name in methods.items():
            count = self.df[method_col].sum()
            rate = count / len(self.df) * 100
            signal_counts.append(count)
            signal_rates.append(rate)

        bars = ax3.bar(method_names, signal_counts, color=['#9467bd', '#8c564b', '#e377c2', '#7f7f7f'])
        ax3.set_title('Signal Frequency', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Number of Signals')

        # Add percentage labels
        for bar, rate in zip(bars, signal_rates):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)

        # 4. Risk/Reward Ratio (Middle Left)
        ax4 = fig.add_subplot(gs[1, 0])
        risk_rewards = []

        for method_col, method_name in methods.items():
            consolidated = self.df[self.df[method_col] == True]
            if len(consolidated) > 0:
                avg_gain = consolidated['max_gain_40d'].mean()
                avg_loss = abs(consolidated['max_loss_40d'].mean())
                if avg_loss > 0:
                    risk_rewards.append(avg_gain / avg_loss)
                else:
                    risk_rewards.append(0)

        bars = ax4.bar(method_names, risk_rewards, color=['#17becf', '#bcbd22', '#1f77b4', '#ff7f0e'])
        ax4.set_title('Risk/Reward Ratio (40-day)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Gain/Loss Ratio')
        ax4.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Break-even')
        ax4.legend()

        # Add value labels
        for bar, rr in zip(bars, risk_rewards):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{rr:.2f}', ha='center', va='bottom', fontsize=10)

        # 5. Gain Distribution Boxplot (Middle Center)
        ax5 = fig.add_subplot(gs[1, 1])
        gain_data = []
        labels = []

        for method_col, method_name in methods.items():
            consolidated = self.df[self.df[method_col] == True]
            if len(consolidated) > 0:
                gain_data.append(consolidated['max_gain_40d'].values * 100)
                labels.append(method_name)

        if gain_data:
            bp = ax5.boxplot(gain_data, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax5.set_title('40-Day Gain Distribution', fontsize=12, fontweight='bold')
            ax5.set_ylabel('Gain (%)')
            ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax5.axhline(y=20, color='green', linestyle='--', alpha=0.5, label='20% target')
            ax5.legend()

        # 6. Method Agreement Performance (Middle Right)
        ax6 = fig.add_subplot(gs[1, 2])
        method_cols = ['method1_bollinger', 'method2_range_based',
                      'method3_volume_weighted', 'method4_atr_based']
        self.df['methods_agreed'] = self.df[method_cols].sum(axis=1)

        agreement_gains = []
        agreement_labels = []

        for num_methods in range(1, 5):
            agreed_df = self.df[self.df['methods_agreed'] == num_methods]
            if len(agreed_df) > 0:
                agreement_gains.append(agreed_df['max_gain_40d'].mean() * 100)
                agreement_labels.append(f'{num_methods} Methods')

        if agreement_gains:
            bars = ax6.bar(agreement_labels, agreement_gains,
                          color=['#ffd92f', '#8da0cb', '#fc8d62', '#66c2a5'])
            ax6.set_title('Performance by Method Agreement', fontsize=12, fontweight='bold')
            ax6.set_ylabel('Average 40-Day Gain (%)')

            for bar, gain in zip(bars, agreement_gains):
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{gain:.1f}%', ha='center', va='bottom', fontsize=10)

        # 7. Time Series of Consolidation Signals (Bottom)
        ax7 = fig.add_subplot(gs[2, :])

        # Resample to daily counts
        daily_signals = pd.DataFrame()
        for method_col, method_name in methods.items():
            signal_df = self.df[self.df[method_col] == True].copy()
            if len(signal_df) > 0:
                signal_counts = signal_df.groupby(signal_df['date'].dt.date).size()
                daily_signals[method_name] = signal_counts

        if not daily_signals.empty:
            daily_signals = daily_signals.fillna(0)
            daily_signals.plot(ax=ax7, kind='line', marker='o', markersize=4, alpha=0.7)
            ax7.set_title('Consolidation Signals Over Time', fontsize=12, fontweight='bold')
            ax7.set_xlabel('Date')
            ax7.set_ylabel('Number of Signals')
            ax7.legend(loc='upper left')
            ax7.grid(True, alpha=0.3)

            # Format x-axis dates
            ax7.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax7.xaxis.set_major_locator(mdates.MonthLocator())
            plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.suptitle('Consolidation Pattern Analysis Dashboard', fontsize=16, fontweight='bold', y=0.98)

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")

        plt.show()
        return fig

    def create_interactive_dashboard(self, save_path: Optional[str] = None):
        """Create interactive Plotly dashboard"""

        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Cannot create interactive dashboard.")
            return None

        if self.df is None:
            raise ValueError("No data loaded. Call load_parquet first.")

        # Create subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Success Rate by Method', 'Average Gains', 'Signal Distribution',
                'Risk/Reward Ratio', 'Gain Scatter Plot', 'Method Agreement',
                'Price Movement Patterns', 'Volume Patterns', 'Temporal Heatmap'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}, {"type": "pie"}],
                [{"type": "bar"}, {"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "heatmap"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        methods = {
            'method1_bollinger': 'Bollinger',
            'method2_range_based': 'Range-Based',
            'method3_volume_weighted': 'Volume-Weighted',
            'method4_atr_based': 'ATR-Based'
        }

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        # 1. Success Rate
        success_rates = []
        method_names = []
        for method_col, method_name in methods.items():
            consolidated = self.df[self.df[method_col] == True]
            if len(consolidated) > 0:
                success_rate = (consolidated['max_gain_20d'] > 0.1).mean() * 100
                success_rates.append(success_rate)
                method_names.append(method_name)

        fig.add_trace(
            go.Bar(x=method_names, y=success_rates, marker_color=colors,
                  text=[f'{r:.1f}%' for r in success_rates],
                  textposition='outside'),
            row=1, col=1
        )

        # 2. Average Gains
        for i, (method_col, method_name) in enumerate(methods.items()):
            consolidated = self.df[self.df[method_col] == True]
            if len(consolidated) > 0:
                gains = [
                    consolidated['max_gain_20d'].mean() * 100,
                    consolidated['max_gain_40d'].mean() * 100
                ]
                fig.add_trace(
                    go.Bar(name=method_name, x=['20-day', '40-day'], y=gains,
                          marker_color=colors[i], showlegend=True,
                          text=[f'{g:.1f}%' for g in gains],
                          textposition='outside'),
                    row=1, col=2
                )

        # 3. Signal Distribution Pie
        signal_counts = []
        for method_col in methods.keys():
            signal_counts.append(self.df[method_col].sum())

        fig.add_trace(
            go.Pie(labels=list(methods.values()), values=signal_counts,
                  marker=dict(colors=colors)),
            row=1, col=3
        )

        # 4. Risk/Reward Ratio
        risk_rewards = []
        for method_col, method_name in methods.items():
            consolidated = self.df[self.df[method_col] == True]
            if len(consolidated) > 0:
                avg_gain = consolidated['max_gain_40d'].mean()
                avg_loss = abs(consolidated['max_loss_40d'].mean())
                risk_rewards.append(avg_gain / avg_loss if avg_loss > 0 else 0)

        fig.add_trace(
            go.Bar(x=method_names, y=risk_rewards,
                  marker_color=['green' if rr > 1 else 'red' for rr in risk_rewards],
                  text=[f'{rr:.2f}' for rr in risk_rewards],
                  textposition='outside'),
            row=2, col=1
        )

        # Add break-even line
        fig.add_hline(y=1, line_dash="dash", line_color="gray",
                     annotation_text="Break-even", row=2, col=1)

        # 5. Gain Scatter Plot
        for i, (method_col, method_name) in enumerate(methods.items()):
            consolidated = self.df[self.df[method_col] == True]
            if len(consolidated) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=consolidated['max_gain_20d'] * 100,
                        y=consolidated['max_gain_40d'] * 100,
                        mode='markers',
                        name=method_name,
                        marker=dict(color=colors[i], size=8, opacity=0.6)
                    ),
                    row=2, col=2
                )

        # Add reference lines
        fig.add_shape(type="line", x0=0, y0=0, x1=100, y1=100,
                     line=dict(color="gray", dash="dash"),
                     row=2, col=2)

        # 6. Method Agreement
        method_cols = ['method1_bollinger', 'method2_range_based',
                      'method3_volume_weighted', 'method4_atr_based']
        self.df['methods_agreed'] = self.df[method_cols].sum(axis=1)

        agreement_data = []
        for num in range(1, 5):
            agreed_df = self.df[self.df['methods_agreed'] == num]
            if len(agreed_df) > 0:
                agreement_data.append({
                    'Methods': f'{num} Methods',
                    'Count': len(agreed_df),
                    'Avg_Gain_40d': agreed_df['max_gain_40d'].mean() * 100
                })

        if agreement_data:
            agreement_df = pd.DataFrame(agreement_data)
            fig.add_trace(
                go.Bar(x=agreement_df['Methods'], y=agreement_df['Avg_Gain_40d'],
                      marker_color='purple',
                      text=[f'{g:.1f}%' for g in agreement_df['Avg_Gain_40d']],
                      textposition='outside'),
                row=2, col=3
            )

        # 7. Price Movement Patterns
        sample_data = self.df.sample(min(1000, len(self.df)))  # Sample for performance
        fig.add_trace(
            go.Scatter(
                x=sample_data['bbw'],
                y=sample_data['max_gain_40d'] * 100,
                mode='markers',
                marker=dict(
                    color=sample_data['max_gain_40d'] * 100,
                    colorscale='Viridis',
                    size=5,
                    showscale=True,
                    colorbar=dict(x=0.31, y=0.15, len=0.3)
                ),
                text=[f'Ticker: {t}<br>Date: {d}' for t, d in
                     zip(sample_data['ticker'], sample_data['date'])],
                hovertemplate='%{text}<br>BBW: %{x:.4f}<br>40d Gain: %{y:.1f}%'
            ),
            row=3, col=1
        )

        # 8. Volume Patterns
        fig.add_trace(
            go.Scatter(
                x=sample_data['volume_ratio'],
                y=sample_data['max_gain_40d'] * 100,
                mode='markers',
                marker=dict(
                    color=sample_data['atr_pct'] * 100,
                    colorscale='Plasma',
                    size=5,
                    showscale=True,
                    colorbar=dict(x=0.65, y=0.15, len=0.3)
                ),
                text=[f'Ticker: {t}<br>ATR%: {a:.2f}' for t, a in
                     zip(sample_data['ticker'], sample_data['atr_pct'] * 100)],
                hovertemplate='Volume Ratio: %{x:.2f}<br>40d Gain: %{y:.1f}%<br>%{text}'
            ),
            row=3, col=2
        )

        # 9. Temporal Heatmap
        # Create monthly aggregation
        self.df['month'] = self.df['date'].dt.to_period('M')
        monthly_data = self.df.groupby('month').agg({
            'max_gain_40d': 'mean',
            'method1_bollinger': 'sum'
        }).reset_index()

        if len(monthly_data) > 0:
            # Create a matrix for heatmap
            months = pd.date_range(start=self.df['date'].min(),
                                  end=self.df['date'].max(), freq='M')

            # Create heatmap data
            heatmap_data = []
            for method_col in methods.keys():
                method_monthly = self.df[self.df[method_col] == True].groupby(
                    self.df['date'].dt.to_period('M')
                )['max_gain_40d'].mean() * 100
                heatmap_data.append(method_monthly.values[:12] if len(method_monthly) >= 12
                                   else np.pad(method_monthly.values, (0, 12-len(method_monthly)), constant_values=0))

            fig.add_trace(
                go.Heatmap(
                    z=heatmap_data,
                    x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(heatmap_data[0])],
                    y=list(methods.values()),
                    colorscale='RdYlGn',
                    text=[[f'{val:.1f}%' for val in row] for row in heatmap_data],
                    texttemplate='%{text}',
                    textfont={"size": 10}
                ),
                row=3, col=3
            )

        # Update layout
        fig.update_layout(
            title_text="Consolidation Pattern Analysis - Interactive Dashboard",
            title_font_size=20,
            showlegend=True,
            height=1000,
            hovermode='closest'
        )

        # Update axes labels
        fig.update_xaxes(title_text="Method", row=1, col=1)
        fig.update_yaxes(title_text="Success Rate (%)", row=1, col=1)

        fig.update_xaxes(title_text="Period", row=1, col=2)
        fig.update_yaxes(title_text="Average Gain (%)", row=1, col=2)

        fig.update_xaxes(title_text="Method", row=2, col=1)
        fig.update_yaxes(title_text="Risk/Reward Ratio", row=2, col=1)

        fig.update_xaxes(title_text="20-Day Gain (%)", row=2, col=2)
        fig.update_yaxes(title_text="40-Day Gain (%)", row=2, col=2)

        fig.update_xaxes(title_text="Agreement Level", row=2, col=3)
        fig.update_yaxes(title_text="Avg 40-Day Gain (%)", row=2, col=3)

        fig.update_xaxes(title_text="BBW", row=3, col=1)
        fig.update_yaxes(title_text="40-Day Gain (%)", row=3, col=1)

        fig.update_xaxes(title_text="Volume Ratio", row=3, col=2)
        fig.update_yaxes(title_text="40-Day Gain (%)", row=3, col=2)

        fig.update_xaxes(title_text="Month", row=3, col=3)
        fig.update_yaxes(title_text="Method", row=3, col=3)

        if save_path:
            fig.write_html(save_path)
            print(f"Saved interactive dashboard to {save_path}")

        fig.show()
        return fig

    def create_pattern_analysis_plots(self, save_dir: Optional[str] = None):
        """Create detailed pattern analysis plots"""

        if self.df is None:
            raise ValueError("No data loaded. Call load_parquet first.")

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        # 1. Top Patterns Chart
        fig1, ax = plt.subplots(figsize=(14, 8))

        # Find top 20 patterns by 40-day gain
        top_patterns = self.df.nlargest(20, 'max_gain_40d').copy()
        top_patterns['label'] = top_patterns['ticker'] + ' ' + top_patterns['date'].dt.strftime('%Y-%m-%d')

        # Create bar chart
        bars = ax.barh(range(len(top_patterns)), top_patterns['max_gain_40d'] * 100,
                       color='green', alpha=0.7)

        # Add 20-day gains as overlay
        bars2 = ax.barh(range(len(top_patterns)), top_patterns['max_gain_20d'] * 100,
                        color='blue', alpha=0.5, label='20-day gain')

        ax.set_yticks(range(len(top_patterns)))
        ax.set_yticklabels(top_patterns['label'])
        ax.set_xlabel('Gain (%)', fontsize=12)
        ax.set_title('Top 20 Consolidation Breakout Patterns', fontsize=14, fontweight='bold')
        ax.legend(['40-day gain', '20-day gain'])
        ax.grid(True, alpha=0.3)

        # Add value labels
        for i, (v40, v20) in enumerate(zip(top_patterns['max_gain_40d'] * 100,
                                           top_patterns['max_gain_20d'] * 100)):
            ax.text(v40 + 1, i, f'{v40:.1f}%', va='center', fontsize=9)

        plt.tight_layout()
        if save_dir:
            plt.savefig(save_dir / 'top_patterns.png', dpi=100, bbox_inches='tight')
        plt.show()

        # 2. Volatility Analysis
        fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

        # BBW vs Gain
        ax = axes[0, 0]
        scatter = ax.scatter(self.df['bbw'], self.df['max_gain_40d'] * 100,
                           c=self.df['max_gain_40d'] * 100,
                           cmap='RdYlGn', alpha=0.5, s=20)
        ax.set_xlabel('Bollinger Band Width')
        ax.set_ylabel('40-Day Gain (%)')
        ax.set_title('BBW vs Performance')
        ax.axhline(y=20, color='green', linestyle='--', alpha=0.5, label='20% target')
        ax.legend()
        plt.colorbar(scatter, ax=ax)

        # ATR% vs Gain
        ax = axes[0, 1]
        scatter = ax.scatter(self.df['atr_pct'] * 100, self.df['max_gain_40d'] * 100,
                           c=self.df['max_gain_40d'] * 100,
                           cmap='RdYlGn', alpha=0.5, s=20)
        ax.set_xlabel('ATR %')
        ax.set_ylabel('40-Day Gain (%)')
        ax.set_title('ATR% vs Performance')
        ax.axhline(y=20, color='green', linestyle='--', alpha=0.5)
        plt.colorbar(scatter, ax=ax)

        # Volume Ratio vs Gain
        ax = axes[1, 0]
        scatter = ax.scatter(self.df['volume_ratio'], self.df['max_gain_40d'] * 100,
                           c=self.df['max_gain_40d'] * 100,
                           cmap='RdYlGn', alpha=0.5, s=20)
        ax.set_xlabel('Volume Ratio')
        ax.set_ylabel('40-Day Gain (%)')
        ax.set_title('Volume Ratio vs Performance')
        ax.axhline(y=20, color='green', linestyle='--', alpha=0.5)
        plt.colorbar(scatter, ax=ax)

        # Range Ratio vs Gain
        ax = axes[1, 1]
        scatter = ax.scatter(self.df['range_ratio'], self.df['max_gain_40d'] * 100,
                           c=self.df['max_gain_40d'] * 100,
                           cmap='RdYlGn', alpha=0.5, s=20)
        ax.set_xlabel('Range Ratio')
        ax.set_ylabel('40-Day Gain (%)')
        ax.set_title('Range Ratio vs Performance')
        ax.axhline(y=20, color='green', linestyle='--', alpha=0.5)
        plt.colorbar(scatter, ax=ax)

        plt.suptitle('Volatility Metrics vs Performance Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_dir:
            plt.savefig(save_dir / 'volatility_analysis.png', dpi=100, bbox_inches='tight')
        plt.show()

        # 3. Temporal Analysis
        fig3, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Monthly performance
        ax = axes[0]
        self.df['month'] = self.df['date'].dt.to_period('M')
        monthly_stats = self.df.groupby('month').agg({
            'max_gain_40d': ['mean', 'std', 'count']
        }).reset_index()

        months = monthly_stats['month'].astype(str)
        means = monthly_stats[('max_gain_40d', 'mean')] * 100
        stds = monthly_stats[('max_gain_40d', 'std')] * 100

        ax.bar(months, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
        ax.set_xlabel('Month')
        ax.set_ylabel('Average 40-Day Gain (%)')
        ax.set_title('Monthly Performance Analysis')
        ax.axhline(y=20, color='green', linestyle='--', alpha=0.5, label='20% target')
        ax.legend()
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Weekly pattern
        ax = axes[1]
        self.df['day_of_week'] = self.df['date'].dt.dayofweek
        weekly_stats = self.df.groupby('day_of_week')['max_gain_40d'].mean() * 100

        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        ax.bar([days[i] for i in weekly_stats.index], weekly_stats.values,
               color='coral', alpha=0.7)
        ax.set_xlabel('Day of Week')
        ax.set_ylabel('Average 40-Day Gain (%)')
        ax.set_title('Day of Week Pattern Analysis')
        ax.axhline(y=weekly_stats.mean(), color='red', linestyle='--',
                  alpha=0.5, label=f'Average: {weekly_stats.mean():.1f}%')
        ax.legend()

        plt.tight_layout()
        if save_dir:
            plt.savefig(save_dir / 'temporal_analysis.png', dpi=100, bbox_inches='tight')
        plt.show()

        return fig1, fig2, fig3


def main():
    """Run visualization on parquet files"""
    import argparse

    parser = argparse.ArgumentParser(description='Visualize consolidation patterns from parquet files')
    parser.add_argument('file', help='Path to parquet file')
    parser.add_argument('--output-dir', default='visualizations',
                       help='Output directory for plots')
    parser.add_argument('--interactive', action='store_true',
                       help='Create interactive Plotly dashboard')

    args = parser.parse_args()

    # Create visualizer
    viz = ConsolidationVisualizer()

    # Load data
    print(f"Loading {args.file}...")
    viz.load_parquet(args.file)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nCreating visualizations...")

    # Create static plots
    print("1. Creating method comparison dashboard...")
    viz.create_method_comparison_plot(output_dir / 'method_comparison_dashboard.png')

    print("2. Creating pattern analysis plots...")
    viz.create_pattern_analysis_plots(output_dir)

    # Create interactive dashboard if requested
    if args.interactive or PLOTLY_AVAILABLE:
        print("3. Creating interactive dashboard...")
        viz.create_interactive_dashboard(output_dir / 'interactive_dashboard.html')

    print(f"\nVisualizations saved to {output_dir}/")
    print("\nFiles created:")
    print("  - method_comparison_dashboard.png")
    print("  - top_patterns.png")
    print("  - volatility_analysis.png")
    print("  - temporal_analysis.png")
    if args.interactive or PLOTLY_AVAILABLE:
        print("  - interactive_dashboard.html")


if __name__ == "__main__":
    main()