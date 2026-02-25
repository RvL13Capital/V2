"""
Visualization Module for Consolidation Pattern Analysis
Creates comprehensive charts and visualizations for pattern analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class PatternVisualizer:
    """Create visualizations for consolidation pattern analysis"""
    
    def __init__(self, output_dir: str = './visualizations'):
        self.output_dir = output_dir
        self.color_palette = {
            'K0': '#808080',  # Gray - Stagnant
            'K1': '#FFA500',  # Orange - Minimal
            'K2': '#90EE90',  # Light Green - Quality
            'K3': '#00FF00',  # Green - Strong
            'K4': '#006400',  # Dark Green - Exceptional
            'K5': '#FF0000'   # Red - Failed
        }
        
    def plot_duration_distribution(self, duration_stats: Dict, save: bool = True) -> None:
        """Plot consolidation duration distributions by method"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Consolidation Duration Analysis by Detection Method', fontsize=16)
        
        # Flatten axes for easier iteration
        axes = axes.flatten()
        
        for idx, (method, stats) in enumerate(duration_stats.items()):
            if idx >= 4:
                break
                
            ax = axes[idx]
            
            # Extract distribution data
            distribution = stats.get('distribution', {})
            
            # Create bar plot
            bars = ax.bar(distribution.keys(), distribution.values(), color='steelblue', alpha=0.7)
            
            # Add statistics text
            stats_text = f"Mean: {stats['mean']:.1f} days\n"
            stats_text += f"Median: {stats['median']:.1f} days\n"
            stats_text += f"Std: {stats['std']:.1f} days"
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_title(f'{method.upper()} Method')
            ax.set_xlabel('Duration Range')
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom')
        
        # Hide unused subplots
        for idx in range(len(duration_stats), 4):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/duration_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_outcome_distribution(self, outcome_stats: Dict, save: bool = True) -> None:
        """Plot outcome class distributions by method"""
        
        # Prepare data for stacked bar chart
        methods = []
        outcome_data = {cls: [] for cls in ['K0', 'K1', 'K2', 'K3', 'K4', 'K5']}
        
        for method, stats in outcome_stats.items():
            methods.append(method)
            distribution = stats.get('outcome_distribution', {})
            
            for outcome_class in ['K0', 'K1', 'K2', 'K3', 'K4', 'K5']:
                outcome_data[outcome_class].append(distribution.get(outcome_class, 0))
        
        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bottom = np.zeros(len(methods))
        
        for outcome_class, values in outcome_data.items():
            bars = ax.bar(methods, values, bottom=bottom, 
                         label=outcome_class, color=self.color_palette[outcome_class])
            bottom += np.array(values)
            
            # Add percentage labels
            for idx, (bar, val) in enumerate(zip(bars, values)):
                if val > 0:
                    total = sum(outcome_data[oc][idx] for oc in outcome_data)
                    pct = (val / total * 100) if total > 0 else 0
                    if pct > 5:  # Only show if > 5%
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., 
                               bar.get_y() + height/2.,
                               f'{pct:.0f}%', ha='center', va='center', 
                               color='white', fontweight='bold')
        
        ax.set_title('Outcome Distribution by Detection Method', fontsize=14)
        ax.set_xlabel('Detection Method')
        ax.set_ylabel('Number of Patterns')
        ax.legend(title='Outcome Class', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add success rate annotation
        for idx, method in enumerate(methods):
            if method in outcome_stats:
                success_rate = outcome_stats[method].get('success_rate', 0)
                ax.text(idx, -5, f'{success_rate:.1%}', ha='center', va='top', 
                       fontsize=10, fontweight='bold')
        
        ax.text(len(methods)/2, -10, 'Success Rate', ha='center', va='top', 
               fontsize=10, style='italic')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/outcome_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_post_breakout_performance(self, performance_data: Dict, save: bool = True) -> None:
        """Plot post-breakout performance at different time intervals"""
        
        days = sorted(performance_data.keys())
        avg_gains = [performance_data[d]['avg_gain'] for d in days]
        avg_losses = [performance_data[d]['avg_loss'] for d in days]
        win_rates = [performance_data[d]['win_rate'] for d in days]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Average gains and losses
        x = np.arange(len(days))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, avg_gains, width, label='Avg Gain', color='green', alpha=0.7)
        bars2 = ax1.bar(x + width/2, avg_losses, width, label='Avg Loss', color='red', alpha=0.7)
        
        ax1.set_xlabel('Days After Breakout')
        ax1.set_ylabel('Average Performance (%)')
        ax1.set_title('Post-Breakout Performance Analysis')
        ax1.set_xticks(x)
        ax1.set_xticklabels(days)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        # Plot 2: Win rate over time
        ax2.plot(days, [wr * 100 for wr in win_rates], marker='o', linewidth=2, 
                markersize=8, color='blue')
        ax2.fill_between(days, 0, [wr * 100 for wr in win_rates], alpha=0.3, color='blue')
        
        ax2.set_xlabel('Days After Breakout')
        ax2.set_ylabel('Win Rate (%)')
        ax2.set_title('Win Rate Evolution Post-Breakout')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 100])
        
        # Add 50% reference line
        ax2.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% Baseline')
        ax2.legend()
        
        # Add value labels
        for day, wr in zip(days, win_rates):
            ax2.text(day, wr * 100 + 2, f'{wr:.1%}', ha='center', fontsize=9)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/post_breakout_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_heatmap(self, correlation_data: pd.DataFrame, title: str = "Feature Correlations", 
                      save: bool = True) -> None:
        """Create correlation heatmap"""
        
        plt.figure(figsize=(12, 10))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(correlation_data, dtype=bool))
        
        # Create heatmap
        sns.heatmap(correlation_data, mask=mask, annot=True, fmt='.2f', 
                   cmap='RdBu_r', center=0, square=True,
                   cbar_kws={"shrink": .8},
                   vmin=-1, vmax=1)
        
        plt.title(title, fontsize=14)
        plt.tight_layout()
        
        if save:
            filename = title.lower().replace(' ', '_')
            plt.savefig(f'{self.output_dir}/{filename}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_fakeout_analysis(self, fakeout_data: List[Dict], save: bool = True) -> None:
        """Visualize fakeout patterns and characteristics"""
        
        if not fakeout_data:
            logger.warning("No fakeout data to visualize")
            return
        
        df = pd.DataFrame(fakeout_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Fakeout Pattern Analysis', fontsize=16)
        
        # Plot 1: Fakeout rate by method
        if 'method' in df.columns:
            ax1 = axes[0, 0]
            fakeout_by_method = df.groupby('method')['is_fakeout'].mean()
            fakeout_by_method.plot(kind='bar', ax=ax1, color='coral')
            ax1.set_title('Fakeout Rate by Detection Method')
            ax1.set_ylabel('Fakeout Rate')
            ax1.set_xlabel('Method')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add percentage labels
            for idx, val in enumerate(fakeout_by_method):
                ax1.text(idx, val, f'{val:.1%}', ha='center', va='bottom')
        
        # Plot 2: Days to fakeout distribution
        if 'fakeout_days' in df.columns:
            ax2 = axes[0, 1]
            fakeout_df = df[df['is_fakeout'] == True]
            if not fakeout_df.empty:
                fakeout_df['fakeout_days'].hist(bins=10, ax=ax2, color='orange', alpha=0.7)
                ax2.set_title('Days Until Fakeout Detected')
                ax2.set_xlabel('Days')
                ax2.set_ylabel('Frequency')
                
                # Add statistics
                mean_days = fakeout_df['fakeout_days'].mean()
                median_days = fakeout_df['fakeout_days'].median()
                ax2.axvline(mean_days, color='red', linestyle='--', label=f'Mean: {mean_days:.1f}')
                ax2.axvline(median_days, color='blue', linestyle='--', label=f'Median: {median_days:.1f}')
                ax2.legend()
        
        # Plot 3: Fakeout types
        if 'fakeout_type' in df.columns:
            ax3 = axes[1, 0]
            fakeout_types = df[df['is_fakeout'] == True]['fakeout_type'].value_counts()
            if not fakeout_types.empty:
                fakeout_types.plot(kind='pie', ax=ax3, autopct='%1.1f%%')
                ax3.set_title('Fakeout Types Distribution')
                ax3.set_ylabel('')
        
        # Plot 4: Sustainability score distribution
        if 'sustainability_score' in df.columns:
            ax4 = axes[1, 1]
            
            # Compare sustainability scores for fakeouts vs valid breakouts
            fakeouts = df[df['is_fakeout'] == True]['sustainability_score'].dropna()
            valid = df[df['is_fakeout'] == False]['sustainability_score'].dropna()
            
            if len(fakeouts) > 0 and len(valid) > 0:
                ax4.hist([fakeouts, valid], label=['Fakeouts', 'Valid'], 
                        bins=15, alpha=0.7, color=['red', 'green'])
                ax4.set_title('Sustainability Score Distribution')
                ax4.set_xlabel('Sustainability Score')
                ax4.set_ylabel('Frequency')
                ax4.legend()
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/fakeout_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_pattern_chart(self, 
                                        price_data: pd.DataFrame,
                                        patterns: List[Dict],
                                        ticker: str) -> go.Figure:
        """Create interactive chart showing patterns on price data"""
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.03,
                           row_heights=[0.7, 0.3],
                           subplot_titles=(f'{ticker} - Consolidation Patterns', 'Volume'))
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=price_data.index,
                open=price_data['open'],
                high=price_data['high'],
                low=price_data['low'],
                close=price_data['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add patterns as shaded regions
        for pattern in patterns:
            start_date = pd.to_datetime(pattern['start_date'])
            end_date = pd.to_datetime(pattern['end_date'])
            
            # Determine color based on outcome
            outcome = pattern.get('outcome_class', 'unknown')
            color = self.color_palette.get(outcome, 'gray')
            
            # Add shaded region for consolidation period
            fig.add_vrect(
                x0=start_date, x1=end_date,
                fillcolor=color, opacity=0.2,
                layer="below", line_width=0,
                row=1, col=1
            )
            
            # Add boundary lines
            fig.add_trace(
                go.Scatter(
                    x=[start_date, end_date],
                    y=[pattern['upper_boundary'], pattern['upper_boundary']],
                    mode='lines',
                    line=dict(color='blue', width=1, dash='dash'),
                    name=f'Upper {pattern["detection_method"]}',
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[start_date, end_date],
                    y=[pattern['lower_boundary'], pattern['lower_boundary']],
                    mode='lines',
                    line=dict(color='red', width=1, dash='dash'),
                    name=f'Lower {pattern["detection_method"]}',
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Add volume
        if 'volume' in price_data.columns:
            fig.add_trace(
                go.Bar(
                    x=price_data.index,
                    y=price_data['volume'],
                    name='Volume',
                    marker_color='lightblue'
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f'{ticker} Consolidation Pattern Analysis',
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig
    
    def plot_method_comparison(self, comparison_df: pd.DataFrame, save: bool = True) -> None:
        """Create comprehensive method comparison visualization"""
        
        if comparison_df.empty:
            logger.warning("No data for method comparison")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Detection Method Performance Comparison', fontsize=16)
        
        # Plot 1: Success rates
        ax1 = axes[0, 0]
        comparison_df.plot(x='method', y='success_rate', kind='bar', ax=ax1, color='green', alpha=0.7)
        ax1.set_title('Success Rate by Method')
        ax1.set_ylabel('Success Rate')
        ax1.set_xlabel('')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Total patterns detected
        ax2 = axes[0, 1]
        comparison_df.plot(x='method', y='total_patterns', kind='bar', ax=ax2, color='blue', alpha=0.7)
        ax2.set_title('Total Patterns Detected')
        ax2.set_ylabel('Count')
        ax2.set_xlabel('')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Average duration
        ax3 = axes[0, 2]
        comparison_df.plot(x='method', y='avg_duration', kind='bar', ax=ax3, color='orange', alpha=0.7)
        ax3.set_title('Average Pattern Duration')
        ax3.set_ylabel('Days')
        ax3.set_xlabel('')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Exceptional rate
        ax4 = axes[1, 0]
        comparison_df.plot(x='method', y='exceptional_rate', kind='bar', ax=ax4, color='gold', alpha=0.7)
        ax4.set_title('Exceptional Pattern Rate (K4)')
        ax4.set_ylabel('Rate')
        ax4.set_xlabel('Method')
        ax4.tick_params(axis='x', rotation=45)
        
        # Plot 5: Failure rate
        ax5 = axes[1, 1]
        comparison_df.plot(x='method', y='failure_rate', kind='bar', ax=ax5, color='red', alpha=0.7)
        ax5.set_title('Failure Rate (K5)')
        ax5.set_ylabel('Rate')
        ax5.set_xlabel('Method')
        ax5.tick_params(axis='x', rotation=45)
        
        # Plot 6: Average max gain
        ax6 = axes[1, 2]
        comparison_df.plot(x='method', y='avg_max_gain', kind='bar', ax=ax6, color='purple', alpha=0.7)
        ax6.set_title('Average Maximum Gain')
        ax6.set_ylabel('Gain (%)')
        ax6.set_xlabel('Method')
        ax6.tick_params(axis='x', rotation=45)
        
        # Remove legends from individual plots
        for ax in axes.flat:
            ax.legend().set_visible(False)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/method_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_dashboard(self, analysis_results: Dict) -> None:
        """Create comprehensive dashboard with all key visualizations"""
        
        # This would typically use a dashboard library like Dash or Streamlit
        # For now, we'll create a multi-panel matplotlib figure
        
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Add various plots based on available data
        if 'summary' in analysis_results:
            ax1 = fig.add_subplot(gs[0, :])
            summary = analysis_results['summary']
            
            # Create summary text
            summary_text = f"Total Patterns: {summary.get('total_patterns', 0)}\n"
            summary_text += f"Success Rate: {summary.get('overall_success_rate', 0):.1%}\n"
            summary_text += f"Exceptional Patterns: {summary.get('exceptional_patterns', 0)}\n"
            summary_text += f"Failed Patterns: {summary.get('failed_patterns', 0)}"
            
            ax1.text(0.5, 0.5, summary_text, transform=ax1.transAxes,
                    fontsize=14, ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            ax1.set_title('Analysis Summary', fontsize=16, fontweight='bold')
            ax1.axis('off')
        
        plt.suptitle('Consolidation Pattern Analysis Dashboard', fontsize=18, fontweight='bold')
        
        if save:
            plt.savefig(f'{self.output_dir}/dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()