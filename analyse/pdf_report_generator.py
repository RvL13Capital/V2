"""
PDF Report Generator for Consolidation Pattern Analysis
Creates comprehensive PDF reports with visualizations and detailed metrics
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

logger = logging.getLogger(__name__)

class PDFReportGenerator:
    """Generate comprehensive PDF reports for pattern analysis"""
    
    def __init__(self, patterns: List[Dict], extended_analysis: Dict = None):
        self.patterns = patterns
        self.df = pd.DataFrame(patterns) if patterns else pd.DataFrame()
        self.extended_analysis = extended_analysis or {}
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def generate_report(self, output_path: str = None):
        """Generate complete PDF report"""
        
        if output_path is None:
            output_path = f'consolidation_analysis_report_{self.timestamp}.pdf'
        
        with PdfPages(output_path) as pdf:
            # Title page
            self._add_title_page(pdf)
            
            # Model explanation
            self._add_model_explanation(pdf)
            
            # Executive summary
            self._add_executive_summary(pdf)
            
            # Pattern distribution analysis
            self._add_pattern_distribution(pdf)
            
            # Time to targets analysis
            self._add_time_to_targets(pdf)
            
            # Consolidation duration analysis
            self._add_duration_analysis(pdf)
            
            # False breakout analysis
            self._add_false_breakout_analysis(pdf)
            
            # Detection method comparison
            self._add_method_comparison(pdf)
            
            # Risk/Reward metrics
            self._add_risk_reward_analysis(pdf)
            
            # Top performers
            self._add_top_performers(pdf)
            
            # Example patterns
            self._add_example_patterns(pdf)
            
            # Metadata
            d = pdf.infodict()
            d['Title'] = 'Consolidation Pattern Analysis Report'
            d['Author'] = 'AIv3 Analysis System'
            d['Subject'] = 'Pattern Detection and Analysis'
            d['Keywords'] = 'Consolidation, Patterns, Trading, Analysis'
            d['CreationDate'] = datetime.now()
        
        logger.info(f"PDF report saved to: {output_path}")
        return output_path
    
    def _add_title_page(self, pdf):
        """Add title page"""
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.7, 'CONSOLIDATION PATTERN', 
                ha='center', size=32, weight='bold')
        fig.text(0.5, 0.65, 'ANALYSIS REPORT', 
                ha='center', size=32, weight='bold')
        
        fig.text(0.5, 0.5, 'AIv3 Pattern Detection System', 
                ha='center', size=18, style='italic')
        
        fig.text(0.5, 0.35, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                ha='center', size=14)
        
        # Statistics box
        stats_text = f"""
Total Patterns Analyzed: {len(self.patterns):,}
Unique Tickers: {self.df['ticker'].nunique() if not self.df.empty else 0:,}
Date Range: {self.df['start_date'].min() if not self.df.empty else 'N/A'} to {self.df['end_date'].max() if not self.df.empty else 'N/A'}
        """
        
        fig.text(0.5, 0.2, stats_text, ha='center', size=12, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.5))
        
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _add_model_explanation(self, pdf):
        """Add model explanation page"""
        fig = plt.figure(figsize=(11, 8.5))
        
        title = "CONSOLIDATION DETECTION MODEL"
        explanation = """
The AIv3 system identifies consolidation patterns using a multi-factor approach:

1. PATTERN DETECTION CRITERIA:
   • Price Range: Maximum 15% between high and low over the period
   • Daily Volatility: Average daily range below 3%
   • Volume Contraction: Trading volume below 80% of 20-day average
   • Duration: Minimum 10 days, typically 20-30 days

2. OUTCOME CLASSIFICATION (K0-K5):
   • K4 (Exceptional): >75% gain - High-value explosive moves
   • K3 (Strong): 35-75% gain - Significant profitable patterns
   • K2 (Quality): 15-35% gain - Solid tradeable opportunities
   • K1 (Minimal): 5-15% gain - Small positive moves
   • K0 (Stagnant): <5% gain - No meaningful movement
   • K5 (Failed): Breakdown below support - Risk patterns to avoid

3. ANALYSIS APPROACH:
   • Scans historical data for consolidation periods
   • Evaluates 100-day forward performance post-breakout
   • Classifies patterns based on maximum gain achieved
   • Identifies optimal characteristics for successful patterns

4. KEY INSIGHTS:
   • Tighter consolidations often lead to stronger breakouts
   • Volume contraction is a key indicator of pending moves
   • Duration of 15-25 days shows optimal success rates
   • Failed patterns (K5) provide valuable risk indicators
        """
        
        fig.text(0.5, 0.95, title, ha='center', size=20, weight='bold')
        fig.text(0.1, 0.85, explanation, ha='left', va='top', size=10, 
                wrap=True, family='monospace')
        
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _add_executive_summary(self, pdf):
        """Add executive summary with key metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(11, 8.5))
        fig.suptitle('EXECUTIVE SUMMARY', size=20, weight='bold')
        
        if not self.df.empty:
            # Success rate pie chart
            ax = axes[0, 0]
            successful = len(self.df[self.df['outcome_class'].isin(['K2', 'K3', 'K4'])])
            failed = len(self.df[self.df['outcome_class'] == 'K5'])
            neutral = len(self.df) - successful - failed
            
            ax.pie([successful, neutral, failed], 
                  labels=['Successful', 'Neutral', 'Failed'],
                  colors=['green', 'gray', 'red'],
                  autopct='%1.1f%%')
            ax.set_title('Pattern Outcomes')
            
            # Outcome distribution
            ax = axes[0, 1]
            outcome_counts = self.df['outcome_class'].value_counts().sort_index()
            colors = ['red' if x == 'K5' else 'green' if x in ['K3', 'K4'] else 'gray' 
                     for x in outcome_counts.index]
            ax.bar(outcome_counts.index, outcome_counts.values, color=colors)
            ax.set_title('Outcome Distribution')
            ax.set_xlabel('Outcome Class')
            ax.set_ylabel('Count')
            
            # Gain distribution
            ax = axes[0, 2]
            gains = self.df['max_gain'].values
            ax.hist(gains, bins=50, edgecolor='black', alpha=0.7)
            ax.axvline(0, color='red', linestyle='--', alpha=0.5)
            ax.set_title('Gain Distribution')
            ax.set_xlabel('Max Gain (%)')
            ax.set_ylabel('Frequency')
            
            # Duration by outcome
            ax = axes[1, 0]
            for outcome in ['K0', 'K1', 'K2', 'K3', 'K4', 'K5']:
                data = self.df[self.df['outcome_class'] == outcome]['duration']
                if not data.empty:
                    ax.scatter([outcome] * len(data), data, alpha=0.5, s=20)
            ax.set_title('Duration by Outcome')
            ax.set_xlabel('Outcome Class')
            ax.set_ylabel('Duration (days)')
            
            # Top tickers by pattern count
            ax = axes[1, 1]
            top_tickers = self.df['ticker'].value_counts().head(10)
            ax.barh(range(len(top_tickers)), top_tickers.values)
            ax.set_yticks(range(len(top_tickers)))
            ax.set_yticklabels(top_tickers.index)
            ax.set_title('Top 10 Tickers by Pattern Count')
            ax.set_xlabel('Number of Patterns')
            
            # Risk/Reward scatter
            ax = axes[1, 2]
            if 'max_loss' in self.df.columns:
                ax.scatter(self.df['max_loss'], self.df['max_gain'], 
                          alpha=0.3, s=10)
                ax.set_xlabel('Max Loss (%)')
                ax.set_ylabel('Max Gain (%)')
                ax.set_title('Risk vs Reward')
                ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
                ax.axvline(0, color='gray', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _add_pattern_distribution(self, pdf):
        """Add pattern distribution analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('PATTERN DISTRIBUTION ANALYSIS', size=16, weight='bold')
        
        if not self.df.empty:
            # Monthly distribution
            ax = axes[0, 0]
            self.df['month'] = pd.to_datetime(self.df['start_date']).dt.month
            monthly = self.df.groupby('month').size()
            ax.bar(monthly.index, monthly.values)
            ax.set_title('Patterns by Month')
            ax.set_xlabel('Month')
            ax.set_ylabel('Count')
            ax.set_xticks(range(1, 13))
            
            # Year distribution
            ax = axes[0, 1]
            self.df['year'] = pd.to_datetime(self.df['start_date']).dt.year
            yearly = self.df.groupby('year').size()
            if len(yearly) > 0:
                ax.bar(yearly.index, yearly.values)
                ax.set_title('Patterns by Year')
                ax.set_xlabel('Year')
                ax.set_ylabel('Count')
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            # Boundary width distribution
            ax = axes[1, 0]
            if 'boundary_width_pct' in self.df.columns:
                ax.hist(self.df['boundary_width_pct'].dropna(), bins=30, edgecolor='black')
                ax.set_title('Boundary Width Distribution')
                ax.set_xlabel('Boundary Width (%)')
                ax.set_ylabel('Frequency')
            
            # Volume contraction distribution
            ax = axes[1, 1]
            if 'volume_contraction' in self.df.columns:
                ax.hist(self.df['volume_contraction'].dropna(), bins=30, edgecolor='black')
                ax.set_title('Volume Contraction Distribution')
                ax.set_xlabel('Volume Ratio')
                ax.set_ylabel('Frequency')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _add_time_to_targets(self, pdf):
        """Add time to targets analysis"""
        if 'time_to_targets' not in self.extended_analysis:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(11, 8.5))
        fig.suptitle('TIME TO TARGET ANALYSIS', size=16, weight='bold')
        
        targets_data = self.extended_analysis['time_to_targets']
        target_names = ['5%', '10%', '15%', '25%', '50%', 'Max']
        target_keys = ['target_5pct', 'target_10pct', 'target_15pct', 
                      'target_25pct', 'target_50pct', 'target_max']
        
        for idx, (ax, key, name) in enumerate(zip(axes.flat, target_keys, target_names)):
            if key in targets_data:
                data = targets_data[key]
                
                # Create bar chart
                metrics = ['Avg', 'Median', 'Min', 'Max']
                values = [data['avg_days'], data['median_days'], 
                         data['min_days'], data['max_days']]
                
                bars = ax.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
                ax.set_title(f'Days to {name} Gain')
                ax.set_ylabel('Days')
                
                # Add count annotation
                ax.text(0.5, 0.95, f"n={data['count']}", 
                       transform=ax.transAxes, ha='center')
                
                # Add value labels on bars
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _add_duration_analysis(self, pdf):
        """Add consolidation duration analysis"""
        if 'duration_by_outcome' not in self.extended_analysis:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(11, 8.5))
        fig.suptitle('CONSOLIDATION DURATION ANALYSIS', size=16, weight='bold')
        
        duration_data = self.extended_analysis['duration_by_outcome']
        
        # Duration by outcome class
        ax = axes[0, 0]
        outcomes = []
        avg_durations = []
        colors = []
        
        for outcome in ['K0', 'K1', 'K2', 'K3', 'K4', 'K5']:
            if outcome in duration_data:
                outcomes.append(outcome)
                avg_durations.append(duration_data[outcome]['avg_duration'])
                if outcome in ['K3', 'K4']:
                    colors.append('green')
                elif outcome == 'K5':
                    colors.append('red')
                else:
                    colors.append('gray')
        
        ax.bar(outcomes, avg_durations, color=colors)
        ax.set_title('Average Duration by Outcome')
        ax.set_xlabel('Outcome Class')
        ax.set_ylabel('Days')
        
        # Successful vs Failed comparison
        ax = axes[0, 1]
        if 'successful_overall' in duration_data and 'failed_overall' in duration_data:
            categories = ['Successful\n(K2-K4)', 'Failed\n(K5)']
            counts = [duration_data['successful_overall']['count'], 
                     duration_data['failed_overall']['count']]
            avg_dur = [duration_data['successful_overall']['avg_duration'],
                      duration_data['failed_overall']['avg_duration']]
            
            x = np.arange(len(categories))
            width = 0.35
            
            ax.bar(x - width/2, counts, width, label='Count', color='blue', alpha=0.7)
            ax2 = ax.twinx()
            ax2.bar(x + width/2, avg_dur, width, label='Avg Duration', color='orange', alpha=0.7)
            
            ax.set_xlabel('Category')
            ax.set_ylabel('Count', color='blue')
            ax2.set_ylabel('Avg Duration (days)', color='orange')
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.set_title('Successful vs Failed Patterns')
        
        # Duration distribution heatmap
        ax = axes[0, 2]
        if any('distribution' in v for v in duration_data.values() if isinstance(v, dict)):
            # Create matrix for heatmap
            buckets = ['10-15', '16-20', '21-30', '31-40', '41-50', '51-75', '76-100', '>100']
            matrix = []
            
            for outcome in ['K0', 'K1', 'K2', 'K3', 'K4', 'K5']:
                if outcome in duration_data and 'distribution' in duration_data[outcome]:
                    row = [duration_data[outcome]['distribution'].get(b, 0) for b in buckets]
                    matrix.append(row)
            
            if matrix:
                im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd')
                ax.set_xticks(range(len(buckets)))
                ax.set_xticklabels(buckets, rotation=45)
                ax.set_yticks(range(len(['K0', 'K1', 'K2', 'K3', 'K4', 'K5'])))
                ax.set_yticklabels(['K0', 'K1', 'K2', 'K3', 'K4', 'K5'])
                ax.set_title('Duration Distribution Heatmap')
                plt.colorbar(im, ax=ax)
        
        # Additional visualizations in remaining subplots
        # Can add more detailed analysis here
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _add_false_breakout_analysis(self, pdf):
        """Add false breakout analysis"""
        if 'false_breakouts' not in self.extended_analysis:
            return
        
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('FALSE BREAKOUT ANALYSIS', size=16, weight='bold')
        
        false_data = self.extended_analysis['false_breakouts']
        
        # Create text summary
        summary_text = f"""
FALSE BREAKOUT ANALYSIS SUMMARY

Total Upward Breakouts: {false_data.get('total_upward_breakouts', 0):,}
False Upward Breakouts: {false_data.get('false_upward_breakouts', 0):,}
True Upward Breakouts: {false_data.get('true_upward_breakouts', 0):,}
False Breakout Rate: {false_data.get('false_breakout_rate', 0):.1f}%

CHARACTERISTICS OF FALSE BREAKOUTS:
"""
        
        if 'characteristics' in false_data and false_data['characteristics']:
            chars = false_data['characteristics']
            summary_text += f"""
Average Duration: {chars.get('avg_duration', 0):.1f} days
Average Boundary Width: {chars.get('avg_boundary_width', 0):.2f}%
Average Initial Gain: {chars.get('avg_initial_gain', 0):.2f}%
"""
        
        fig.text(0.1, 0.8, summary_text, ha='left', va='top', size=12, 
                family='monospace')
        
        # Add visualization if data available
        if false_data.get('total_upward_breakouts', 0) > 0:
            ax = fig.add_subplot(2, 2, 3)
            
            labels = ['True Breakouts', 'False Breakouts']
            sizes = [false_data.get('true_upward_breakouts', 0),
                    false_data.get('false_upward_breakouts', 0)]
            colors = ['green', 'red']
            
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                  startangle=90)
            ax.set_title('Breakout Reliability')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _add_method_comparison(self, pdf):
        """Add detection method comparison"""
        if 'by_method' not in self.extended_analysis:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('DETECTION METHOD COMPARISON', size=16, weight='bold')
        
        method_data = self.extended_analysis['by_method']
        
        if method_data:
            methods = list(method_data.keys())
            
            # Total patterns by method
            ax = axes[0, 0]
            totals = [method_data[m]['total_patterns'] for m in methods]
            ax.bar(methods, totals)
            ax.set_title('Total Patterns by Method')
            ax.set_xlabel('Method')
            ax.set_ylabel('Count')
            
            # Success rate by method
            ax = axes[0, 1]
            success_rates = [method_data[m]['success_rate'] for m in methods]
            bars = ax.bar(methods, success_rates, color=['green' if sr > 15 else 'orange' if sr > 10 else 'red' for sr in success_rates])
            ax.set_title('Success Rate by Method')
            ax.set_xlabel('Method')
            ax.set_ylabel('Success Rate (%)')
            ax.axhline(y=15, color='green', linestyle='--', alpha=0.5)
            ax.axhline(y=10, color='orange', linestyle='--', alpha=0.5)
            
            # Positive vs Negative distribution
            ax = axes[1, 0]
            x = np.arange(len(methods))
            width = 0.35
            
            positives = [method_data[m]['positive_patterns'] for m in methods]
            negatives = [method_data[m]['negative_patterns'] for m in methods]
            
            ax.bar(x - width/2, positives, width, label='Positive', color='green', alpha=0.7)
            ax.bar(x + width/2, negatives, width, label='Negative', color='red', alpha=0.7)
            
            ax.set_xlabel('Method')
            ax.set_ylabel('Count')
            ax.set_title('Positive vs Negative Patterns')
            ax.set_xticks(x)
            ax.set_xticklabels(methods)
            ax.legend()
            
            # Exceptional rate by method
            ax = axes[1, 1]
            exceptional_rates = [method_data[m].get('exceptional_rate', 0) for m in methods]
            ax.bar(methods, exceptional_rates, color='gold')
            ax.set_title('Exceptional Pattern Rate (K4)')
            ax.set_xlabel('Method')
            ax.set_ylabel('Rate (%)')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _add_risk_reward_analysis(self, pdf):
        """Add risk/reward analysis"""
        if 'risk_reward' not in self.extended_analysis:
            return
        
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('RISK/REWARD ANALYSIS', size=16, weight='bold')
        
        rr_data = self.extended_analysis['risk_reward']
        
        # Create comprehensive summary
        summary_text = f"""
RISK/REWARD METRICS

Performance Metrics:
  • Average Gain: {rr_data.get('avg_gain', 0):.2f}%
  • Average Loss: {rr_data.get('avg_loss', 0):.2f}%
  • Win Rate: {rr_data.get('win_rate', 0):.1f}%
  
Risk Metrics:
  • Risk/Reward Ratio: {rr_data.get('risk_reward_ratio', 0):.2f}
  • Profit Factor: {rr_data.get('profit_factor', 0):.2f}
  • Sharpe Ratio: {rr_data.get('sharpe_ratio', 0):.2f}
  
Expected Value:
  • Expectancy: {rr_data.get('expectancy', 0):.2f}%
  
This indicates that on average, each pattern is expected to return {rr_data.get('expectancy', 0):.2f}%.
A positive expectancy suggests a profitable system over the long term.
"""
        
        fig.text(0.1, 0.7, summary_text, ha='left', va='top', size=11, 
                family='monospace')
        
        # Add visual representation
        ax = fig.add_subplot(2, 2, 3)
        
        # Win rate pie chart
        wins = rr_data.get('win_rate', 0)
        losses = 100 - wins
        
        ax.pie([wins, losses], labels=['Wins', 'Losses'], 
              colors=['green', 'red'], autopct='%1.1f%%')
        ax.set_title('Win/Loss Distribution')
        
        # Risk/Reward bar chart
        ax = fig.add_subplot(2, 2, 4)
        
        metrics = ['Avg Gain', 'Avg Loss', 'Expectancy']
        values = [rr_data.get('avg_gain', 0), 
                 -rr_data.get('avg_loss', 0),
                 rr_data.get('expectancy', 0)]
        colors = ['green', 'red', 'blue']
        
        bars = ax.bar(metrics, values, color=colors)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Risk/Reward Components')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}%', ha='center', 
                   va='bottom' if height > 0 else 'top')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _add_top_performers(self, pdf):
        """Add top performing tickers analysis"""
        if self.df.empty:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('TOP PERFORMING TICKERS', size=16, weight='bold')
        
        # Calculate ticker statistics
        ticker_stats = self.df.groupby('ticker').agg({
            'outcome_class': 'count',
            'max_gain': ['mean', 'max']
        }).round(2)
        ticker_stats.columns = ['count', 'avg_gain', 'max_gain']
        
        # Calculate success rate
        successful = self.df[self.df['outcome_class'].isin(['K2', 'K3', 'K4'])]
        ticker_success = successful.groupby('ticker').size()
        ticker_stats['success_rate'] = (ticker_success / ticker_stats['count'] * 100).fillna(0)
        
        # Top by average gain
        ax = axes[0, 0]
        top_gain = ticker_stats.nlargest(10, 'avg_gain')
        ax.barh(range(len(top_gain)), top_gain['avg_gain'].values)
        ax.set_yticks(range(len(top_gain)))
        ax.set_yticklabels(top_gain.index)
        ax.set_xlabel('Average Gain (%)')
        ax.set_title('Top 10 by Average Gain')
        
        # Top by success rate (min 5 patterns)
        ax = axes[0, 1]
        qualified = ticker_stats[ticker_stats['count'] >= 5]
        if not qualified.empty:
            top_success = qualified.nlargest(10, 'success_rate')
            ax.barh(range(len(top_success)), top_success['success_rate'].values)
            ax.set_yticks(range(len(top_success)))
            ax.set_yticklabels(top_success.index)
            ax.set_xlabel('Success Rate (%)')
            ax.set_title('Top 10 by Success Rate (min 5 patterns)')
        
        # Top by max gain achieved
        ax = axes[1, 0]
        top_max = ticker_stats.nlargest(10, 'max_gain')
        ax.barh(range(len(top_max)), top_max['max_gain'].values)
        ax.set_yticks(range(len(top_max)))
        ax.set_yticklabels(top_max.index)
        ax.set_xlabel('Maximum Gain (%)')
        ax.set_title('Top 10 by Maximum Gain')
        
        # Top by pattern count
        ax = axes[1, 1]
        top_count = ticker_stats.nlargest(10, 'count')
        ax.barh(range(len(top_count)), top_count['count'].values)
        ax.set_yticks(range(len(top_count)))
        ax.set_yticklabels(top_count.index)
        ax.set_xlabel('Number of Patterns')
        ax.set_title('Top 10 by Pattern Count')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _add_example_patterns(self, pdf):
        """Add example patterns for inspection"""
        if 'example_patterns' not in self.extended_analysis:
            return
        
        examples = self.extended_analysis['example_patterns']
        
        if not examples:
            return
        
        for idx, pattern in enumerate(examples[:5]):
            fig = plt.figure(figsize=(11, 8.5))
            fig.suptitle(f'EXAMPLE PATTERN {idx + 1}: {pattern.get("ticker", "N/A")}', 
                        size=16, weight='bold')
            
            # Create detailed pattern description
            description = f"""
PATTERN DETAILS:

Ticker: {pattern.get('ticker', 'N/A')}
Date Range: {pattern.get('start_date', 'N/A')} to {pattern.get('end_date', 'N/A')}
Duration: {pattern.get('duration', 'N/A')} days

BOUNDARIES:
  • Upper Boundary: ${pattern.get('upper_boundary', 0):.2f}
  • Lower Boundary: ${pattern.get('lower_boundary', 0):.2f}
  • Boundary Width: {pattern.get('boundary_width_pct', 0):.2f}%

METRICS:
  • Average Range: {pattern.get('avg_range', 0):.2f}%
  • Volume Contraction: {pattern.get('volume_contraction', 0):.2f}
  
OUTCOME:
  • Classification: {pattern.get('outcome_class', 'N/A')}
  • Maximum Gain: {pattern.get('max_gain', 0):.2f}%
  • Maximum Loss: {pattern.get('max_loss', 0):.2f}%

ANALYSIS:
"""
            
            # Add interpretation
            outcome = pattern.get('outcome_class', '')
            if outcome == 'K4':
                description += "  This was an EXCEPTIONAL pattern with >75% gain potential.\n"
                description += "  Key indicators suggest optimal entry at consolidation breakout.\n"
            elif outcome == 'K3':
                description += "  This was a STRONG pattern with 35-75% gain potential.\n"
                description += "  Solid opportunity for significant returns.\n"
            elif outcome == 'K2':
                description += "  This was a QUALITY pattern with 15-35% gain potential.\n"
                description += "  Reliable setup for moderate gains.\n"
            elif outcome == 'K5':
                description += "  This pattern FAILED with breakdown below support.\n"
                description += "  Important for understanding risk factors.\n"
            else:
                description += "  This pattern showed minimal movement.\n"
                description += "  Useful for identifying non-opportunities.\n"
            
            # Add key takeaways
            description += "\nKEY TAKEAWAYS:\n"
            
            if pattern.get('boundary_width_pct', 100) < 10:
                description += "  • Tight consolidation range indicates strong potential\n"
            if pattern.get('volume_contraction', 1) < 0.5:
                description += "  • Significant volume contraction before breakout\n"
            if pattern.get('duration', 0) >= 15 and pattern.get('duration', 0) <= 25:
                description += "  • Optimal duration range for successful patterns\n"
            
            fig.text(0.1, 0.9, description, ha='left', va='top', size=10, 
                    family='monospace')
            
            # Add visual representation if possible
            ax = fig.add_subplot(2, 1, 2)
            
            # Create a simple visual representation of the pattern
            days = range(pattern.get('duration', 20))
            upper = [pattern.get('upper_boundary', 100)] * len(days)
            lower = [pattern.get('lower_boundary', 90)] * len(days)
            
            ax.fill_between(days, lower, upper, alpha=0.3, color='blue', label='Consolidation Range')
            ax.plot(days, upper, 'b--', alpha=0.5, label='Upper Boundary')
            ax.plot(days, lower, 'r--', alpha=0.5, label='Lower Boundary')
            
            ax.set_xlabel('Days')
            ax.set_ylabel('Price')
            ax.set_title('Consolidation Pattern Visualization')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
    
    def add_custom_analysis(self, pdf, title: str, content: str):
        """Add custom analysis page"""
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle(title, size=16, weight='bold')
        
        fig.text(0.1, 0.9, content, ha='left', va='top', size=10, 
                wrap=True, family='monospace')
        
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()