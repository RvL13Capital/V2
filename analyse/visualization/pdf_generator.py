"""
Unified PDF Report Generator for AIv3 System
Consolidates all PDF generation functionality
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import tempfile
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class UnifiedPDFGenerator:
    """Unified PDF report generator for all analysis types"""

    def __init__(self, output_dir: str = "./reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.styles = self._setup_styles()
        self.temp_dir = tempfile.mkdtemp()

    def _setup_styles(self) -> Dict:
        """Setup PDF styles"""
        styles = getSampleStyleSheet()

        # Custom styles
        styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))

        styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor('#2ca02c'),
            spaceAfter=12,
            spaceBefore=12
        ))

        styles.add(ParagraphStyle(
            name='SubHeader',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#ff7f0e'),
            spaceAfter=6
        ))

        styles.add(ParagraphStyle(
            name='Metric',
            parent=styles['Normal'],
            fontSize=11,
            leftIndent=20
        ))

        return styles

    def generate_pattern_report(
        self,
        patterns: pd.DataFrame,
        title: str = "AIv3 Pattern Analysis Report",
        filename: Optional[str] = None
    ) -> str:
        """Generate comprehensive pattern analysis report"""
        if filename is None:
            filename = f"pattern_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        filepath = self.output_dir / filename
        doc = SimpleDocTemplate(str(filepath), pagesize=letter)
        story = []

        # Title
        story.append(Paragraph(title, self.styles['CustomTitle']))
        story.append(Spacer(1, 0.3 * inch))

        # Executive Summary
        story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        summary_data = self._create_summary_table(patterns)
        story.append(summary_data)
        story.append(Spacer(1, 0.2 * inch))

        # Pattern Distribution Charts
        story.append(Paragraph("Pattern Analysis", self.styles['SectionHeader']))

        # Outcome distribution chart
        outcome_chart = self._create_outcome_distribution_chart(patterns)
        if outcome_chart:
            story.append(Image(outcome_chart, width=6 * inch, height=4 * inch))
            story.append(Spacer(1, 0.2 * inch))

        # Duration analysis chart
        duration_chart = self._create_duration_analysis_chart(patterns)
        if duration_chart:
            story.append(Image(duration_chart, width=6 * inch, height=3 * inch))

        story.append(PageBreak())

        # Top Patterns Table
        story.append(Paragraph("Top Performing Patterns", self.styles['SectionHeader']))
        top_patterns_table = self._create_top_patterns_table(patterns)
        story.append(top_patterns_table)

        story.append(PageBreak())

        # Statistical Analysis
        story.append(Paragraph("Statistical Analysis", self.styles['SectionHeader']))
        stats_table = self._create_statistics_table(patterns)
        story.append(stats_table)

        # Build PDF
        doc.build(story)
        logger.info(f"PDF report generated: {filepath}")
        return str(filepath)

    def _create_summary_table(self, patterns: pd.DataFrame) -> Table:
        """Create summary statistics table"""
        total_patterns = len(patterns)
        explosive = len(patterns[patterns['outcome_class'] == 'K4_EXPLOSIVE']) if 'outcome_class' in patterns else 0
        strong = len(patterns[patterns['outcome_class'] == 'K3_STRONG']) if 'outcome_class' in patterns else 0
        failed = len(patterns[patterns['outcome_class'] == 'K5_FAILED']) if 'outcome_class' in patterns else 0

        avg_duration = patterns['duration_days'].mean() if 'duration_days' in patterns else 0
        avg_gain = patterns['outcome_max_gain'].mean() if 'outcome_max_gain' in patterns else 0

        data = [
            ['Metric', 'Value'],
            ['Total Patterns', f"{total_patterns:,}"],
            ['Explosive Patterns (>75%)', f"{explosive:,} ({explosive/total_patterns*100:.1f}%)"],
            ['Strong Patterns (35-75%)', f"{strong:,} ({strong/total_patterns*100:.1f}%)"],
            ['Failed Patterns', f"{failed:,} ({failed/total_patterns*100:.1f}%)"],
            ['Average Duration', f"{avg_duration:.1f} days"],
            ['Average Max Gain', f"{avg_gain:.1f}%"],
        ]

        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))

        return table

    def _create_outcome_distribution_chart(self, patterns: pd.DataFrame) -> Optional[str]:
        """Create outcome distribution pie chart"""
        if 'outcome_class' not in patterns.columns:
            return None

        outcome_counts = patterns['outcome_class'].value_counts()

        fig = go.Figure(data=[go.Pie(
            labels=outcome_counts.index,
            values=outcome_counts.values,
            hole=0.3,
            marker=dict(colors=px.colors.qualitative.Set2)
        )])

        fig.update_layout(
            title="Pattern Outcome Distribution",
            showlegend=True,
            width=600,
            height=400
        )

        # Save to temp file
        chart_path = Path(self.temp_dir) / "outcome_distribution.png"
        fig.write_image(str(chart_path))
        return str(chart_path)

    def _create_duration_analysis_chart(self, patterns: pd.DataFrame) -> Optional[str]:
        """Create duration vs outcome analysis chart"""
        if 'duration_days' not in patterns.columns or 'outcome_max_gain' not in patterns.columns:
            return None

        fig = go.Figure()

        # Scatter plot
        fig.add_trace(go.Scatter(
            x=patterns['duration_days'],
            y=patterns['outcome_max_gain'],
            mode='markers',
            marker=dict(
                size=8,
                color=patterns['outcome_max_gain'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Max Gain %")
            ),
            text=patterns['ticker'] if 'ticker' in patterns else None,
            hovertemplate='Duration: %{x} days<br>Max Gain: %{y:.1f}%<extra></extra>'
        ))

        # Add trend line
        z = np.polyfit(patterns['duration_days'].fillna(0), patterns['outcome_max_gain'].fillna(0), 1)
        p = np.poly1d(z)
        x_trend = np.linspace(patterns['duration_days'].min(), patterns['duration_days'].max(), 100)

        fig.add_trace(go.Scatter(
            x=x_trend,
            y=p(x_trend),
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Trend'
        ))

        fig.update_layout(
            title="Pattern Duration vs Maximum Gain",
            xaxis_title="Duration (days)",
            yaxis_title="Maximum Gain (%)",
            width=600,
            height=300,
            showlegend=False
        )

        chart_path = Path(self.temp_dir) / "duration_analysis.png"
        fig.write_image(str(chart_path))
        return str(chart_path)

    def _create_top_patterns_table(self, patterns: pd.DataFrame, top_n: int = 10) -> Table:
        """Create table of top performing patterns"""
        if 'outcome_max_gain' not in patterns.columns:
            return Table([['No pattern data available']])

        top_patterns = patterns.nlargest(top_n, 'outcome_max_gain')

        data = [['Ticker', 'Start Date', 'Duration', 'Max Gain', 'Outcome']]

        for _, row in top_patterns.iterrows():
            data.append([
                row.get('ticker', 'N/A'),
                row.get('pattern_start_date', 'N/A').strftime('%Y-%m-%d') if pd.notna(
                    row.get('pattern_start_date')) else 'N/A',
                f"{row.get('duration_days', 0):.0f}d",
                f"{row.get('outcome_max_gain', 0):.1f}%",
                row.get('outcome_class', 'N/A')
            ])

        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))

        return table

    def _create_statistics_table(self, patterns: pd.DataFrame) -> Table:
        """Create detailed statistics table"""
        stats_data = [
            ['Metric', 'Mean', 'Median', 'Std Dev', 'Min', 'Max']
        ]

        metrics = ['duration_days', 'outcome_max_gain', 'avg_bbw', 'avg_volume_ratio', 'price_range_pct']

        for metric in metrics:
            if metric in patterns.columns:
                col_data = patterns[metric].dropna()
                stats_data.append([
                    metric.replace('_', ' ').title(),
                    f"{col_data.mean():.2f}",
                    f"{col_data.median():.2f}",
                    f"{col_data.std():.2f}",
                    f"{col_data.min():.2f}",
                    f"{col_data.max():.2f}"
                ])

        table = Table(stats_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))

        return table

    def generate_backtest_report(
        self,
        backtest_results: Dict[str, Any],
        filename: Optional[str] = None
    ) -> str:
        """Generate backtest report"""
        if filename is None:
            filename = f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        # Implementation for backtest reports
        # (Similar structure to pattern report but with backtest-specific metrics)
        pass

    def generate_analysis_report(
        self,
        analysis_data: Dict[str, Any],
        report_type: str = "comprehensive",
        filename: Optional[str] = None
    ) -> str:
        """Generate general analysis report"""
        if filename is None:
            filename = f"{report_type}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        # Implementation for various analysis report types
        pass