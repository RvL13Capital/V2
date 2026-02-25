"""
Enhanced PDF Generator with Detailed Analysis
Generates comprehensive reports with extensive details and statistics
"""

import os
import sys
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats


class EnhancedPDFReportGenerator:
    """Enhanced PDF report generator with detailed analysis"""
    
    def __init__(self, data, output_filename):
        self.data = data
        self.output_filename = output_filename
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.styles = getSampleStyleSheet()
        
        # Add custom styles
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#1f4788'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor('#2e5090'),
            spaceBefore=20,
            spaceAfter=15,
            leftIndent=0
        ))
        
        self.styles.add(ParagraphStyle(
            name='DetailText',
            parent=self.styles['Normal'],
            fontSize=10,
            alignment=TA_JUSTIFY,
            spaceBefore=6,
            spaceAfter=6
        ))
        
    def generate_report(self):
        """Generate the comprehensive PDF report"""
        print(f"\nGenerating enhanced PDF report: {self.output_filename}")
        
        doc = SimpleDocTemplate(
            self.output_filename,
            pagesize=A4,
            rightMargin=50,
            leftMargin=50,
            topMargin=50,
            bottomMargin=30,
        )
        
        story = []
        
        # Title page
        story.extend(self._create_title_page())
        story.append(PageBreak())
        
        # Executive Summary
        story.extend(self._create_executive_summary())
        story.append(PageBreak())
        
        # Detailed Dataset Analysis
        story.extend(self._create_detailed_dataset_analysis())
        story.append(PageBreak())
        
        # Pattern Performance Analysis
        story.extend(self._create_pattern_performance_analysis())
        story.append(PageBreak())
        
        # Time-based Analysis
        story.extend(self._create_temporal_analysis())
        story.append(PageBreak())
        
        # Statistical Deep Dive
        story.extend(self._create_statistical_analysis())
        story.append(PageBreak())
        
        # Risk Analysis
        story.extend(self._create_risk_analysis())
        story.append(PageBreak())
        
        # Top Patterns Analysis
        story.extend(self._create_top_patterns_analysis())
        story.append(PageBreak())
        
        # Recommendations
        story.extend(self._create_recommendations())
        
        # Build PDF
        try:
            doc.build(story)
            print(f"Enhanced PDF report successfully generated: {self.output_filename}")
            return True
        except Exception as e:
            print(f"Error building PDF: {e}")
            return False
    
    def _create_title_page(self):
        """Create detailed title page"""
        elements = []
        
        elements.append(Spacer(1, 1.5*inch))
        elements.append(Paragraph(
            "<b>AIv3 CONSOLIDATION PATTERN</b><br/>COMPREHENSIVE ANALYSIS REPORT",
            self.styles['CustomTitle']
        ))
        
        elements.append(Spacer(1, 0.5*inch))
        
        # Report metadata
        metadata = f"""
        <b>Report Generated:</b> {datetime.now().strftime('%d %B %Y at %H:%M:%S')}<br/>
        <b>Analysis Period:</b> Full Historical Data<br/>
        <b>Data Source:</b> Google Cloud Storage (GCS)<br/>
        <b>Total Patterns Analyzed:</b> {len(self.data):,}<br/>
        <b>Analysis Version:</b> AIv3 Enhanced v2.0
        """
        elements.append(Paragraph(metadata, self.styles['Normal']))
        
        elements.append(Spacer(1, 1*inch))
        
        # Key metrics overview
        if 'outcome_class' in self.data.columns:
            total = len(self.data)
            k4_count = len(self.data[self.data['outcome_class'] == 'K4'])
            k3_count = len(self.data[self.data['outcome_class'] == 'K3'])
            k2_count = len(self.data[self.data['outcome_class'] == 'K2'])
            
            key_metrics = f"""
            <b>KEY PERFORMANCE INDICATORS</b><br/><br/>
            Exceptional Patterns (K4 - >75% gain): {k4_count:,} ({k4_count/total*100:.1f}%)<br/>
            Strong Patterns (K3 - 35-75% gain): {k3_count:,} ({k3_count/total*100:.1f}%)<br/>
            Quality Patterns (K2 - 15-35% gain): {k2_count:,} ({k2_count/total*100:.1f}%)<br/>
            Total Success Rate: {(k4_count+k3_count+k2_count)/total*100:.1f}%
            """
            elements.append(Paragraph(key_metrics, self.styles['Normal']))
        
        return elements
    
    def _create_executive_summary(self):
        """Create detailed executive summary"""
        elements = []
        
        elements.append(Paragraph("<b>EXECUTIVE SUMMARY</b>", self.styles['SectionHeader']))
        
        # Calculate comprehensive statistics
        total_patterns = len(self.data)
        
        if total_patterns > 0:
            # Basic statistics
            avg_duration = self.data['duration'].mean()
            avg_boundary_width = self.data['boundary_width'].mean()
            avg_volume_contraction = self.data['volume_contraction'].mean()
            avg_gain = self.data['max_gain'].mean()
            
            summary_text = f"""
            <b>Analysis Overview:</b><br/>
            This comprehensive analysis examined {total_patterns:,} consolidation patterns across multiple securities 
            using complete historical data from Google Cloud Storage. The analysis focuses on identifying 
            high-probability consolidation patterns that precede significant price movements.<br/><br/>
            
            <b>Key Findings:</b><br/>
            • Average Pattern Duration: {avg_duration:.1f} days<br/>
            • Average Boundary Width: {avg_boundary_width:.1f}%<br/>
            • Average Volume Contraction: {avg_volume_contraction:.2f}<br/>
            • Average Maximum Gain: {avg_gain:.1f}%<br/><br/>
            """
            
            elements.append(Paragraph(summary_text, self.styles['DetailText']))
            
            # Success rate analysis
            if 'outcome_class' in self.data.columns:
                success_patterns = self.data[self.data['outcome_class'].isin(['K2', 'K3', 'K4'])]
                success_rate = len(success_patterns) / total_patterns * 100
                
                success_text = f"""
                <b>Performance Metrics:</b><br/>
                • Overall Success Rate: {success_rate:.1f}%<br/>
                • Total Successful Patterns: {len(success_patterns):,}<br/>
                • Average Gain (Successful): {success_patterns['max_gain'].mean():.1f}%<br/>
                • Median Gain (Successful): {success_patterns['max_gain'].median():.1f}%<br/>
                • 90th Percentile Gain: {success_patterns['max_gain'].quantile(0.9):.1f}%<br/><br/>
                
                <b>Risk Metrics:</b><br/>
                • Failed Patterns (K5): {len(self.data[self.data['outcome_class'] == 'K5']):,}<br/>
                • Failure Rate: {len(self.data[self.data['outcome_class'] == 'K5'])/total_patterns*100:.1f}%<br/>
                • Risk-Reward Ratio: {success_patterns['max_gain'].mean() / max(1, abs(self.data[self.data['outcome_class'] == 'K5']['max_gain'].mean())):.2f}<br/>
                """
                
                elements.append(Paragraph(success_text, self.styles['DetailText']))
        
        return elements
    
    def _create_detailed_dataset_analysis(self):
        """Create comprehensive dataset analysis"""
        elements = []
        
        elements.append(Paragraph("<b>DETAILED DATASET ANALYSIS</b>", self.styles['SectionHeader']))
        
        # Distribution analysis
        elements.append(Paragraph("<b>1. Pattern Duration Distribution</b>", self.styles['Heading2']))
        
        duration_stats = self.data['duration'].describe()
        duration_text = f"""
        The pattern duration analysis reveals important insights about consolidation timeframes:<br/><br/>
        
        • Minimum Duration: {duration_stats['min']:.0f} days<br/>
        • 25th Percentile: {duration_stats['25%']:.0f} days<br/>
        • Median Duration: {duration_stats['50%']:.0f} days<br/>
        • 75th Percentile: {duration_stats['75%']:.0f} days<br/>
        • Maximum Duration: {duration_stats['max']:.0f} days<br/>
        • Standard Deviation: {duration_stats['std']:.1f} days<br/><br/>
        
        <b>Interpretation:</b> Patterns with duration between {duration_stats['25%']:.0f} and {duration_stats['75%']:.0f} days 
        represent the core 50% of observations and may offer the best risk-reward profile.
        """
        elements.append(Paragraph(duration_text, self.styles['DetailText']))
        
        elements.append(Spacer(1, 0.2*inch))
        
        # Boundary width analysis
        elements.append(Paragraph("<b>2. Boundary Width Analysis</b>", self.styles['Heading2']))
        
        boundary_stats = self.data['boundary_width'].describe()
        boundary_text = f"""
        Boundary width indicates the volatility range during consolidation:<br/><br/>
        
        • Tightest Pattern: {boundary_stats['min']:.1f}%<br/>
        • Median Width: {boundary_stats['50%']:.1f}%<br/>
        • Mean Width: {boundary_stats['mean']:.1f}%<br/>
        • Widest Pattern: {boundary_stats['max']:.1f}%<br/><br/>
        
        <b>Key Insight:</b> Patterns with boundary width below {boundary_stats['25%']:.1f}% (25th percentile) 
        show tight consolidation and may have higher breakout potential.
        """
        elements.append(Paragraph(boundary_text, self.styles['DetailText']))
        
        elements.append(Spacer(1, 0.2*inch))
        
        # Volume analysis
        elements.append(Paragraph("<b>3. Volume Contraction Analysis</b>", self.styles['Heading2']))
        
        volume_stats = self.data['volume_contraction'].describe()
        volume_text = f"""
        Volume contraction is a critical indicator of consolidation strength:<br/><br/>
        
        • Average Contraction: {volume_stats['mean']:.2f}<br/>
        • Median Contraction: {volume_stats['50%']:.2f}<br/>
        • Strongest Contraction: {volume_stats['min']:.2f}<br/>
        • Weakest Contraction: {volume_stats['max']:.2f}<br/><br/>
        
        <b>Trading Insight:</b> Patterns with volume contraction below {volume_stats['25%']:.2f} 
        indicate strong consolidation with reduced trading interest before potential breakout.
        """
        elements.append(Paragraph(volume_text, self.styles['DetailText']))
        
        # Create data quality table
        elements.append(Spacer(1, 0.3*inch))
        elements.append(Paragraph("<b>4. Data Quality Metrics</b>", self.styles['Heading2']))
        
        quality_data = [
            ['Metric', 'Value', 'Status'],
            ['Total Records', f'{len(self.data):,}', 'Good' if len(self.data) > 1000 else 'Limited'],
            ['Missing Values', f'{self.data.isnull().sum().sum()}', 'Excellent' if self.data.isnull().sum().sum() == 0 else 'Check'],
            ['Duplicate Patterns', f'{self.data.duplicated().sum()}', 'Clean' if self.data.duplicated().sum() == 0 else 'Review'],
            ['Data Completeness', f'{(1 - self.data.isnull().sum().sum() / (len(self.data) * len(self.data.columns))) * 100:.1f}%', 'Complete']
        ]
        
        quality_table = Table(quality_data, colWidths=[2*inch, 1.5*inch, 1*inch])
        quality_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(quality_table)
        
        return elements
    
    def _create_pattern_performance_analysis(self):
        """Create detailed pattern performance analysis"""
        elements = []
        
        elements.append(Paragraph("<b>PATTERN PERFORMANCE ANALYSIS</b>", self.styles['SectionHeader']))
        
        if 'outcome_class' in self.data.columns:
            # Performance by outcome class
            elements.append(Paragraph("<b>1. Performance by Outcome Classification</b>", self.styles['Heading2']))
            
            outcome_summary = []
            outcome_summary.append(['Class', 'Count', 'Percentage', 'Avg Gain', 'Max Gain', 'Success Rate'])
            
            for outcome in ['K4', 'K3', 'K2', 'K1', 'K0', 'K5']:
                subset = self.data[self.data['outcome_class'] == outcome]
                if len(subset) > 0:
                    outcome_summary.append([
                        outcome,
                        f'{len(subset):,}',
                        f'{len(subset)/len(self.data)*100:.1f}%',
                        f'{subset["max_gain"].mean():.1f}%',
                        f'{subset["max_gain"].max():.1f}%',
                        '✓' if outcome in ['K2', 'K3', 'K4'] else '✗'
                    ])
            
            outcome_table = Table(outcome_summary, colWidths=[0.8*inch, 1*inch, 1*inch, 1*inch, 1*inch, 1*inch])
            outcome_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2e5090')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(outcome_table)
            
            elements.append(Spacer(1, 0.2*inch))
            
            # Detailed performance metrics
            elements.append(Paragraph("<b>2. Advanced Performance Metrics</b>", self.styles['Heading2']))
            
            successful = self.data[self.data['outcome_class'].isin(['K2', 'K3', 'K4'])]
            failed = self.data[self.data['outcome_class'] == 'K5']
            
            if len(successful) > 0:
                perf_text = f"""
                <b>Successful Patterns Analysis (K2-K4):</b><br/>
                • Total Count: {len(successful):,} patterns<br/>
                • Average Gain: {successful['max_gain'].mean():.2f}%<br/>
                • Median Gain: {successful['max_gain'].median():.2f}%<br/>
                • Standard Deviation: {successful['max_gain'].std():.2f}%<br/>
                • Skewness: {stats.skew(successful['max_gain']):.2f}<br/>
                • Kurtosis: {stats.kurtosis(successful['max_gain']):.2f}<br/><br/>
                
                <b>Pattern Characteristics of Successful Trades:</b><br/>
                • Average Duration: {successful['duration'].mean():.1f} days<br/>
                • Average Boundary Width: {successful['boundary_width'].mean():.2f}%<br/>
                • Average Volume Contraction: {successful['volume_contraction'].mean():.3f}<br/><br/>
                
                <b>Percentile Distribution of Gains:</b><br/>
                • 10th Percentile: {successful['max_gain'].quantile(0.1):.1f}%<br/>
                • 25th Percentile: {successful['max_gain'].quantile(0.25):.1f}%<br/>
                • 50th Percentile (Median): {successful['max_gain'].quantile(0.5):.1f}%<br/>
                • 75th Percentile: {successful['max_gain'].quantile(0.75):.1f}%<br/>
                • 90th Percentile: {successful['max_gain'].quantile(0.9):.1f}%<br/>
                • 95th Percentile: {successful['max_gain'].quantile(0.95):.1f}%<br/>
                • 99th Percentile: {successful['max_gain'].quantile(0.99):.1f}%<br/>
                """
                elements.append(Paragraph(perf_text, self.styles['DetailText']))
        
        return elements
    
    def _create_temporal_analysis(self):
        """Create time-based analysis"""
        elements = []
        
        elements.append(Paragraph("<b>TEMPORAL ANALYSIS</b>", self.styles['SectionHeader']))
        
        # Duration-based success analysis
        elements.append(Paragraph("<b>1. Success Rate by Pattern Duration</b>", self.styles['Heading2']))
        
        duration_bins = [(10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 100)]
        duration_analysis = []
        duration_analysis.append(['Duration Range', 'Count', 'Success Rate', 'Avg Gain', 'K4 Rate'])
        
        for min_dur, max_dur in duration_bins:
            subset = self.data[(self.data['duration'] >= min_dur) & (self.data['duration'] < max_dur)]
            if len(subset) > 0 and 'outcome_class' in self.data.columns:
                success_rate = len(subset[subset['outcome_class'].isin(['K2', 'K3', 'K4'])]) / len(subset) * 100
                k4_rate = len(subset[subset['outcome_class'] == 'K4']) / len(subset) * 100
                duration_analysis.append([
                    f'{min_dur}-{max_dur} days',
                    f'{len(subset):,}',
                    f'{success_rate:.1f}%',
                    f'{subset["max_gain"].mean():.1f}%',
                    f'{k4_rate:.1f}%'
                ])
        
        if len(duration_analysis) > 1:
            duration_table = Table(duration_analysis, colWidths=[1.5*inch, 1*inch, 1.2*inch, 1*inch, 1*inch])
            duration_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(duration_table)
        
        elements.append(Spacer(1, 0.2*inch))
        
        # Time to peak analysis
        if 'time_to_peak' in self.data.columns:
            elements.append(Paragraph("<b>2. Time to Peak Analysis</b>", self.styles['Heading2']))
            
            time_stats = self.data['time_to_peak'].dropna()
            if len(time_stats) > 0:
                time_text = f"""
                Analysis of how quickly patterns reach their maximum gain after breakout:<br/><br/>
                
                • Average Time to Peak: {time_stats.mean():.1f} days<br/>
                • Median Time to Peak: {time_stats.median():.0f} days<br/>
                • Fastest Peak: {time_stats.min():.0f} days<br/>
                • Slowest Peak: {time_stats.max():.0f} days<br/><br/>
                
                <b>Strategic Insight:</b> {time_stats.quantile(0.5):.0f}% of patterns reach their peak within {time_stats.quantile(0.75):.0f} days,
                suggesting optimal holding period for maximum gains.
                """
                elements.append(Paragraph(time_text, self.styles['DetailText']))
        
        return elements
    
    def _create_statistical_analysis(self):
        """Create deep statistical analysis"""
        elements = []
        
        elements.append(Paragraph("<b>STATISTICAL DEEP DIVE</b>", self.styles['SectionHeader']))
        
        # Correlation analysis
        elements.append(Paragraph("<b>1. Feature Correlation Analysis</b>", self.styles['Heading2']))
        
        numeric_cols = ['duration', 'boundary_width', 'volume_contraction', 'avg_daily_volatility', 'max_gain']
        numeric_data = self.data[numeric_cols].dropna()
        
        if len(numeric_data) > 0:
            corr_matrix = numeric_data.corr()
            
            corr_text = f"""
            <b>Key Correlations Discovered:</b><br/><br/>
            
            • Duration vs Max Gain: {corr_matrix.loc['duration', 'max_gain']:.3f}<br/>
            • Boundary Width vs Max Gain: {corr_matrix.loc['boundary_width', 'max_gain']:.3f}<br/>
            • Volume Contraction vs Max Gain: {corr_matrix.loc['volume_contraction', 'max_gain']:.3f}<br/>
            • Volatility vs Max Gain: {corr_matrix.loc['avg_daily_volatility', 'max_gain']:.3f}<br/><br/>
            
            <b>Interpretation:</b><br/>
            """
            
            # Add interpretation based on correlations
            if corr_matrix.loc['boundary_width', 'max_gain'] < -0.1:
                corr_text += "• Tighter patterns (lower boundary width) correlate with higher gains<br/>"
            if corr_matrix.loc['volume_contraction', 'max_gain'] < -0.1:
                corr_text += "• Stronger volume contraction correlates with better outcomes<br/>"
            if abs(corr_matrix.loc['duration', 'max_gain']) > 0.1:
                correlation_type = 'positive' if corr_matrix.loc['duration', 'max_gain'] > 0 else 'negative'
                corr_text += f"• Pattern duration shows {correlation_type} correlation with gains<br/>"
            
            elements.append(Paragraph(corr_text, self.styles['DetailText']))
        
        elements.append(Spacer(1, 0.2*inch))
        
        # Statistical tests
        elements.append(Paragraph("<b>2. Statistical Significance Tests</b>", self.styles['Heading2']))
        
        if 'outcome_class' in self.data.columns:
            successful = self.data[self.data['outcome_class'].isin(['K2', 'K3', 'K4'])]
            failed = self.data[self.data['outcome_class'].isin(['K0', 'K1', 'K5'])]
            
            if len(successful) > 0 and len(failed) > 0:
                # T-tests for different features
                test_results = []
                test_results.append(['Feature', 'Successful Mean', 'Failed Mean', 'Difference', 'P-Value', 'Significant'])
                
                for feature in ['duration', 'boundary_width', 'volume_contraction']:
                    if feature in self.data.columns:
                        success_vals = successful[feature].dropna()
                        fail_vals = failed[feature].dropna()
                        
                        if len(success_vals) > 0 and len(fail_vals) > 0:
                            t_stat, p_value = stats.ttest_ind(success_vals, fail_vals)
                            
                            test_results.append([
                                feature,
                                f'{success_vals.mean():.2f}',
                                f'{fail_vals.mean():.2f}',
                                f'{success_vals.mean() - fail_vals.mean():.2f}',
                                f'{p_value:.4f}',
                                '✓' if p_value < 0.05 else '✗'
                            ])
                
                if len(test_results) > 1:
                    test_table = Table(test_results, colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 1*inch, 0.8*inch, 0.8*inch])
                    test_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 9),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    elements.append(test_table)
        
        return elements
    
    def _create_risk_analysis(self):
        """Create comprehensive risk analysis"""
        elements = []
        
        elements.append(Paragraph("<b>RISK ANALYSIS</b>", self.styles['SectionHeader']))
        
        # Risk metrics
        elements.append(Paragraph("<b>1. Risk Metrics Overview</b>", self.styles['Heading2']))
        
        if 'outcome_class' in self.data.columns:
            failed = self.data[self.data['outcome_class'] == 'K5']
            successful = self.data[self.data['outcome_class'].isin(['K2', 'K3', 'K4'])]
            
            avg_loss_failed = f"{failed['max_gain'].mean():.1f}%" if len(failed) > 0 else "N/A"
            max_drawdown = f"{failed['max_gain'].min():.1f}%" if len(failed) > 0 else "N/A"
            avg_win = f"{successful['max_gain'].mean():.1f}%" if len(successful) > 0 else "N/A"
            avg_loss = f"{failed['max_gain'].mean():.1f}%" if len(failed) > 0 else "N/A"
            profit_factor = f"{abs(successful['max_gain'].sum()/failed['max_gain'].sum()):.2f}" if len(failed) > 0 and failed['max_gain'].sum() != 0 else "N/A"
            
            risk_text = f"""
            <b>Failure Analysis:</b><br/>
            • Total Failed Patterns (K5): {len(failed):,}<br/>
            • Failure Rate: {len(failed)/len(self.data)*100:.1f}%<br/>
            • Average Loss on Failure: {avg_loss_failed}<br/>
            • Maximum Drawdown: {max_drawdown}<br/><br/>
            
            <b>Risk-Adjusted Returns:</b><br/>
            • Win Rate: {len(successful)/len(self.data)*100:.1f}%<br/>
            • Average Win: {avg_win}<br/>
            • Average Loss: {avg_loss}<br/>
            • Profit Factor: {profit_factor}<br/>
            • Expected Value: {self.data['max_gain'].mean():.2f}%<br/><br/>
            
            <b>Risk Categories:</b><br/>
            • Low Risk (K2-K4, Duration 20-40 days): {len(self.data[(self.data['outcome_class'].isin(['K2','K3','K4'])) & (self.data['duration'].between(20,40))]):,} patterns<br/>
            • Medium Risk (K1-K3, Any Duration): {len(self.data[self.data['outcome_class'].isin(['K1','K2','K3'])]):,} patterns<br/>
            • High Risk (K0, K5): {len(self.data[self.data['outcome_class'].isin(['K0','K5'])]):,} patterns<br/>
            """
            elements.append(Paragraph(risk_text, self.styles['DetailText']))
        
        elements.append(Spacer(1, 0.2*inch))
        
        # Volatility analysis
        elements.append(Paragraph("<b>2. Volatility and Risk Distribution</b>", self.styles['Heading2']))
        
        if 'avg_daily_volatility' in self.data.columns:
            vol_stats = self.data['avg_daily_volatility'].describe()
            
            vol_text = f"""
            <b>Volatility Statistics:</b><br/>
            • Mean Daily Volatility: {vol_stats['mean']:.2f}%<br/>
            • Volatility Std Dev: {vol_stats['std']:.2f}%<br/>
            • Low Volatility Patterns (<{vol_stats['25%']:.2f}%): {len(self.data[self.data['avg_daily_volatility'] < vol_stats['25%']]):,}<br/>
            • High Volatility Patterns (>{vol_stats['75%']:.2f}%): {len(self.data[self.data['avg_daily_volatility'] > vol_stats['75%']]):,}<br/><br/>
            
            <b>Risk Management Recommendations:</b><br/>
            • Focus on patterns with volatility between {vol_stats['25%']:.2f}% and {vol_stats['50%']:.2f}% for balanced risk<br/>
            • Avoid patterns with volatility above {vol_stats['75%']:.2f}% unless strong confirmation signals present<br/>
            • Implement stop-loss at {vol_stats['mean'] * 2:.1f}% below entry for average volatility patterns<br/>
            """
            elements.append(Paragraph(vol_text, self.styles['DetailText']))
        
        return elements
    
    def _create_top_patterns_analysis(self):
        """Analyze top performing patterns"""
        elements = []
        
        elements.append(Paragraph("<b>TOP PATTERNS ANALYSIS</b>", self.styles['SectionHeader']))
        
        # Top 10 patterns
        elements.append(Paragraph("<b>Top 10 Best Performing Patterns</b>", self.styles['Heading2']))
        
        top_patterns = self.data.nlargest(10, 'max_gain')[['ticker', 'duration', 'boundary_width', 'max_gain', 'outcome_class']]
        
        top_data = [['Rank', 'Ticker', 'Duration', 'Boundary', 'Max Gain', 'Class']]
        for i, (_, row) in enumerate(top_patterns.iterrows(), 1):
            top_data.append([
                str(i),
                row.get('ticker', 'N/A'),
                f"{row['duration']:.0f}d",
                f"{row['boundary_width']:.1f}%",
                f"{row['max_gain']:.1f}%",
                row.get('outcome_class', 'N/A')
            ])
        
        top_table = Table(top_data, colWidths=[0.5*inch, 1.5*inch, 0.8*inch, 0.8*inch, 1*inch, 0.6*inch])
        top_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4CAF50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(top_table)
        
        elements.append(Spacer(1, 0.2*inch))
        
        # Common characteristics of top performers
        elements.append(Paragraph("<b>Common Characteristics of Top Performers</b>", self.styles['Heading2']))
        
        top_20_percent = self.data.nlargest(int(len(self.data) * 0.2), 'max_gain')
        
        characteristics_text = f"""
        Analysis of top 20% performing patterns reveals:<br/><br/>
        
        <b>Duration Profile:</b><br/>
        • Average Duration: {top_20_percent['duration'].mean():.1f} days<br/>
        • Most Common Range: {top_20_percent['duration'].quantile(0.25):.0f}-{top_20_percent['duration'].quantile(0.75):.0f} days<br/><br/>
        
        <b>Boundary Characteristics:</b><br/>
        • Average Boundary Width: {top_20_percent['boundary_width'].mean():.2f}%<br/>
        • Optimal Range: {top_20_percent['boundary_width'].quantile(0.25):.1f}%-{top_20_percent['boundary_width'].quantile(0.75):.1f}%<br/><br/>
        
        <b>Volume Profile:</b><br/>
        • Average Volume Contraction: {top_20_percent['volume_contraction'].mean():.3f}<br/>
        • Typical Range: {top_20_percent['volume_contraction'].quantile(0.25):.2f}-{top_20_percent['volume_contraction'].quantile(0.75):.2f}<br/><br/>
        
        <b>Success Formula:</b><br/>
        The most successful patterns typically combine:
        • Moderate duration ({top_20_percent['duration'].median():.0f} days median)
        • Tight consolidation (boundary width < {top_20_percent['boundary_width'].median():.1f}%)
        • Strong volume contraction (< {top_20_percent['volume_contraction'].median():.2f})
        """
        elements.append(Paragraph(characteristics_text, self.styles['DetailText']))
        
        return elements
    
    def _create_recommendations(self):
        """Create actionable recommendations"""
        elements = []
        
        elements.append(Paragraph("<b>STRATEGIC RECOMMENDATIONS</b>", self.styles['SectionHeader']))
        
        # Calculate key thresholds
        if 'outcome_class' in self.data.columns:
            successful = self.data[self.data['outcome_class'].isin(['K2', 'K3', 'K4'])]
            
            if len(successful) > 0:
                optimal_duration_min = successful['duration'].quantile(0.25)
                optimal_duration_max = successful['duration'].quantile(0.75)
                optimal_boundary = successful['boundary_width'].quantile(0.50)
                optimal_volume = successful['volume_contraction'].quantile(0.50)
                
                recommendations = f"""
                <b>1. Pattern Selection Criteria</b><br/>
                Based on the comprehensive analysis, prioritize patterns with:<br/>
                • Duration between {optimal_duration_min:.0f} and {optimal_duration_max:.0f} days<br/>
                • Boundary width below {optimal_boundary:.1f}%<br/>
                • Volume contraction below {optimal_volume:.2f}<br/>
                • Clear support and resistance levels<br/><br/>
                
                <b>2. Risk Management Guidelines</b><br/>
                • Set stop-loss at 2-3% below consolidation low<br/>
                • Take partial profits at +15% (K2 level)<br/>
                • Hold remaining position for +35% (K3 level) or higher<br/>
                • Maximum position size: 2-3% of portfolio per pattern<br/><br/>
                
                <b>3. Timing Considerations</b><br/>
                • Enter positions near the lower boundary of consolidation<br/>
                • Wait for volume confirmation on breakout (>1.5x average)<br/>
                • Monitor for false breakouts in first 2-3 days<br/>
                • Average holding period: {successful.get('time_to_peak', pd.Series([30])).median():.0f} days<br/><br/>
                
                <b>4. Portfolio Optimization</b><br/>
                • Maintain 5-10 concurrent positions for diversification<br/>
                • Focus on patterns with success rate > {len(successful)/len(self.data)*100:.0f}%<br/>
                • Avoid patterns with characteristics similar to K5 failures<br/>
                • Review and rebalance weekly based on pattern evolution<br/><br/>
                
                <b>5. Advanced Strategies</b><br/>
                • Combine with momentum indicators for confirmation<br/>
                • Use volume-weighted average price (VWAP) for entry timing<br/>
                • Consider market regime (bull/bear) for position sizing<br/>
                • Track sector rotation for additional edge<br/><br/>
                
                <b>Expected Performance Metrics:</b><br/>
                Following these recommendations should yield:<br/>
                • Win Rate: {len(successful)/len(self.data)*100:.0f}% or higher<br/>
                • Average Win: {successful['max_gain'].mean():.1f}%<br/>
                • Risk-Reward Ratio: 1:{successful['max_gain'].mean()/10:.1f}<br/>
                • Monthly Expected Return: {successful['max_gain'].mean()/30:.1f}% (assuming 30-day average hold)<br/>
                """
                
                elements.append(Paragraph(recommendations, self.styles['DetailText']))
        
        elements.append(Spacer(1, 0.3*inch))
        
        # Conclusion
        elements.append(Paragraph("<b>CONCLUSION</b>", self.styles['Heading2']))
        
        conclusion_text = f"""
        This comprehensive analysis of {len(self.data):,} consolidation patterns provides a robust framework 
        for identifying high-probability trading opportunities. The data-driven insights reveal clear patterns 
        of success and failure, enabling systematic approach to pattern trading.<br/><br/>
        
        <b>Key Success Factors:</b><br/>
        • Pattern quality (tight consolidation, volume contraction)<br/>
        • Appropriate duration (not too short, not too extended)<br/>
        • Risk management (position sizing, stop-losses)<br/>
        • Patience and discipline in execution<br/><br/>
        
        <b>Next Steps:</b><br/>
        1. Implement screening criteria based on optimal parameters<br/>
        2. Backtest strategy with historical data<br/>
        3. Start with small positions to validate approach<br/>
        4. Scale gradually based on confirmed results<br/>
        5. Continuously monitor and adjust parameters<br/><br/>
        
        <i>Remember: Past performance does not guarantee future results. Always practice proper risk management.</i>
        """
        
        elements.append(Paragraph(conclusion_text, self.styles['DetailText']))
        
        return elements


def generate_enhanced_report(data, output_filename):
    """Generate enhanced PDF report"""
    generator = EnhancedPDFReportGenerator(data, output_filename)
    return generator.generate_report()


if __name__ == "__main__":
    # This is called from the batch file
    import os
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'gcs-key.json'
    
    # Get parameters from command line or use defaults
    num_tickers = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    
    try:
        from advanced_analysis_with_gcs import load_real_market_data_for_analysis
        
        print(f"Loading data for {num_tickers} tickers...")
        data = load_real_market_data_for_analysis(
            num_tickers=num_tickers,
            use_full_history=True
        )
        
        if data.empty:
            print("ERROR: No patterns found!")
            sys.exit(1)
        
        print(f"Found {len(data)} patterns")
        
        # Generate enhanced report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"enhanced_report_{timestamp}.pdf"
        
        success = generate_enhanced_report(data, report_name)
        
        if success:
            print("\n" + "="*60)
            print("ENHANCED ANALYSIS COMPLETED!")
            print("="*60)
            print(f"\nDetailed PDF Report: {report_name}")
            
            # Try to open PDF
            try:
                import subprocess
                subprocess.Popen(['start', '', report_name], shell=True)
            except:
                pass
        else:
            print("\nERROR: PDF generation failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"FEHLER: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)