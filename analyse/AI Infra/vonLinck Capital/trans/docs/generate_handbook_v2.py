#!/usr/bin/env python3
"""
TRANS System Handbook Generator V2 - Elite Edition
====================================================
Professional hedge fund quality documentation featuring:
- Sophisticated muted color palette
- Precise alignment and generous whitespace
- Elegant data visualizations
- Premium typography hierarchy
- Clean geometric design language

Design Philosophy: Understated elegance, data-driven clarity,
institutional-grade presentation quality.

Author: TRANS Development Team
Date: 2026-01-12
Version: 2.0 Elite
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional
import math

# PDF Generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm, mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, ListFlowable, ListItem, KeepTogether,
    Flowable, HRFlowable, Frame, PageTemplate, BaseDocTemplate
)
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Circle, Polygon, Wedge
from reportlab.graphics import renderPDF
from reportlab.pdfgen import canvas
from reportlab.graphics.charts.barcharts import VerticalBarChart, HorizontalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.widgets.markers import makeMarker


# =============================================================================
# ELITE COLOR PALETTE - Muted, Sophisticated
# =============================================================================

class Colors:
    """Sophisticated hedge fund color palette - muted, professional"""

    # Primary - Deep charcoal navy (not harsh black)
    PRIMARY = colors.Color(0.122, 0.137, 0.165)        # #1f2329 - Rich charcoal
    PRIMARY_LIGHT = colors.Color(0.180, 0.200, 0.235)  # #2e3340 - Lighter charcoal
    PRIMARY_DARK = colors.Color(0.075, 0.086, 0.106)   # #13161b - Near black

    # Accent - Muted gold/bronze (understated luxury)
    ACCENT = colors.Color(0.725, 0.612, 0.463)         # #b99c76 - Muted bronze
    ACCENT_LIGHT = colors.Color(0.847, 0.784, 0.694)   # #d8c8b1 - Light sand
    ACCENT_DARK = colors.Color(0.545, 0.447, 0.318)    # #8b7251 - Dark bronze

    # Success - Muted teal/green
    SUCCESS = colors.Color(0.204, 0.478, 0.412)        # #347a69 - Forest teal
    SUCCESS_LIGHT = colors.Color(0.878, 0.929, 0.906)  # #e0ede7 - Pale teal

    # Caution - Muted amber
    CAUTION = colors.Color(0.780, 0.588, 0.278)        # #c79647 - Muted gold
    CAUTION_LIGHT = colors.Color(0.969, 0.941, 0.894)  # #f7f0e4 - Cream

    # Risk - Muted burgundy (not harsh red)
    RISK = colors.Color(0.612, 0.259, 0.243)           # #9c423e - Burgundy
    RISK_LIGHT = colors.Color(0.957, 0.914, 0.910)     # #f4e9e8 - Pale rose

    # Neutrals - Warm grays
    WHITE = colors.Color(0.992, 0.988, 0.984)          # #fdfcfb - Warm white
    PAPER = colors.Color(0.976, 0.969, 0.957)          # #f9f7f4 - Paper
    GRAY_100 = colors.Color(0.945, 0.937, 0.922)       # #f1efeb - Light gray
    GRAY_200 = colors.Color(0.878, 0.867, 0.847)       # #e0ddd8 - Border gray
    GRAY_300 = colors.Color(0.753, 0.737, 0.710)       # #c0bcb5 - Mid gray
    GRAY_400 = colors.Color(0.565, 0.545, 0.518)       # #908b84 - Text gray
    GRAY_500 = colors.Color(0.420, 0.400, 0.376)       # #6b6660 - Dark gray
    GRAY_600 = colors.Color(0.298, 0.282, 0.263)       # #4c4843 - Charcoal
    TEXT = colors.Color(0.200, 0.188, 0.173)           # #33302c - Body text

    # Data visualization palette (muted, distinguishable)
    VIZ_1 = colors.Color(0.329, 0.431, 0.525)          # #546e86 - Steel blue
    VIZ_2 = colors.Color(0.545, 0.447, 0.318)          # #8b7251 - Bronze
    VIZ_3 = colors.Color(0.420, 0.533, 0.463)          # #6b8876 - Sage
    VIZ_4 = colors.Color(0.612, 0.478, 0.431)          # #9c7a6e - Dusty rose
    VIZ_5 = colors.Color(0.478, 0.459, 0.545)          # #7a758b - Lavender gray


# =============================================================================
# PROFESSIONAL STYLES
# =============================================================================

def get_elite_styles():
    """Create elite hedge fund paragraph styles"""
    styles = getSampleStyleSheet()

    # Override body text for warm, readable typography
    styles['BodyText'].fontSize = 10
    styles['BodyText'].leading = 15
    styles['BodyText'].textColor = Colors.TEXT
    styles['BodyText'].spaceAfter = 10
    styles['BodyText'].alignment = TA_JUSTIFY
    styles['BodyText'].fontName = 'Helvetica'

    # Main title - Large, elegant
    styles.add(ParagraphStyle(
        name='MainTitle',
        parent=styles['Heading1'],
        fontSize=42,
        leading=48,
        textColor=Colors.PRIMARY,
        spaceAfter=8,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
        tracking=2
    ))

    # Subtitle - Light, understated
    styles.add(ParagraphStyle(
        name='Subtitle',
        parent=styles['Normal'],
        fontSize=14,
        leading=18,
        textColor=Colors.GRAY_400,
        spaceAfter=40,
        alignment=TA_CENTER,
        fontName='Helvetica'
    ))

    # Chapter - Clean, authoritative
    styles.add(ParagraphStyle(
        name='Chapter',
        parent=styles['Heading1'],
        fontSize=24,
        leading=30,
        textColor=Colors.PRIMARY,
        spaceBefore=0,
        spaceAfter=20,
        fontName='Helvetica-Bold'
    ))

    # Section - Clear hierarchy
    styles.add(ParagraphStyle(
        name='Section',
        parent=styles['Heading2'],
        fontSize=14,
        leading=18,
        textColor=Colors.PRIMARY_LIGHT,
        spaceBefore=25,
        spaceAfter=12,
        fontName='Helvetica-Bold'
    ))

    # Subsection
    styles.add(ParagraphStyle(
        name='Subsection',
        parent=styles['Heading3'],
        fontSize=11,
        leading=14,
        textColor=Colors.GRAY_500,
        spaceBefore=15,
        spaceAfter=8,
        fontName='Helvetica-Bold'
    ))

    # Insight callout
    styles.add(ParagraphStyle(
        name='Insight',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=Colors.PRIMARY,
        spaceBefore=12,
        spaceAfter=12,
        leftIndent=15,
        rightIndent=15,
        fontName='Helvetica'
    ))

    # Metric display
    styles.add(ParagraphStyle(
        name='MetricLarge',
        parent=styles['Normal'],
        fontSize=32,
        leading=36,
        textColor=Colors.PRIMARY,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    ))

    # Metric label
    styles.add(ParagraphStyle(
        name='MetricLabel',
        parent=styles['Normal'],
        fontSize=9,
        leading=11,
        textColor=Colors.GRAY_400,
        alignment=TA_CENTER,
        fontName='Helvetica',
        spaceBefore=2
    ))

    # Caption
    styles.add(ParagraphStyle(
        name='Caption',
        parent=styles['Normal'],
        fontSize=8,
        leading=10,
        textColor=Colors.GRAY_400,
        alignment=TA_CENTER,
        fontName='Helvetica-Oblique',
        spaceBefore=8
    ))

    # Table header
    styles.add(ParagraphStyle(
        name='TableHeader',
        parent=styles['Normal'],
        fontSize=9,
        leading=11,
        textColor=Colors.WHITE,
        fontName='Helvetica-Bold'
    ))

    return styles


# =============================================================================
# ELEGANT FLOWABLES
# =============================================================================

class ElegantRule(Flowable):
    """Subtle horizontal rule with optional accent"""
    def __init__(self, width=480, thickness=0.5, color=None, accent_width=60):
        Flowable.__init__(self)
        self.rule_width = width
        self.thickness = thickness
        self.color = color or Colors.GRAY_200
        self.accent_width = accent_width

    def wrap(self, availWidth, availHeight):
        return (self.rule_width, self.thickness + 20)

    def draw(self):
        c = self.canv
        y = 10

        # Main line
        c.setStrokeColor(self.color)
        c.setLineWidth(self.thickness)
        c.line(0, y, self.rule_width, y)

        # Accent mark in center
        if self.accent_width > 0:
            center = self.rule_width / 2
            c.setStrokeColor(Colors.ACCENT)
            c.setLineWidth(self.thickness * 2)
            c.line(center - self.accent_width/2, y, center + self.accent_width/2, y)


class EliteMetricCard(Flowable):
    """Premium metric display card"""
    def __init__(self, value: str, label: str, sublabel: str = "",
                 width: int = 110, height: int = 85):
        Flowable.__init__(self)
        self.value = value
        self.label = label
        self.sublabel = sublabel
        self.card_width = width
        self.card_height = height

    def wrap(self, availWidth, availHeight):
        return (self.card_width, self.card_height)

    def draw(self):
        c = self.canv
        w, h = self.card_width, self.card_height

        # Subtle background
        c.setFillColor(Colors.WHITE)
        c.setStrokeColor(Colors.GRAY_200)
        c.setLineWidth(0.5)
        c.roundRect(0, 0, w, h, 4, fill=1, stroke=1)

        # Top accent line
        c.setStrokeColor(Colors.ACCENT)
        c.setLineWidth(2)
        c.line(0, h - 1, w, h - 1)

        # Value
        c.setFillColor(Colors.PRIMARY)
        c.setFont("Helvetica-Bold", 24)
        c.drawCentredString(w/2, h - 38, self.value)

        # Label
        c.setFillColor(Colors.GRAY_500)
        c.setFont("Helvetica", 8)
        c.drawCentredString(w/2, h - 52, self.label)

        # Sublabel
        if self.sublabel:
            c.setFillColor(Colors.GRAY_400)
            c.setFont("Helvetica", 7)
            c.drawCentredString(w/2, h - 65, self.sublabel)


class ExecutiveSummaryBanner(Flowable):
    """Clean executive summary banner"""
    def __init__(self, width=480, height=140):
        Flowable.__init__(self)
        self.width = width
        self.height = height

    def wrap(self, availWidth, availHeight):
        return (self.width, self.height)

    def draw(self):
        c = self.canv
        w, h = self.width, self.height

        # Background
        c.setFillColor(Colors.PRIMARY)
        c.roundRect(0, 0, w, h, 6, fill=1, stroke=0)

        # Subtle pattern overlay (diagonal lines)
        c.setStrokeColor(colors.Color(1, 1, 1, alpha=0.03))
        c.setLineWidth(0.5)
        for i in range(-20, 50):
            c.line(i * 15, 0, i * 15 + h, h)

        # Header section
        c.setFillColor(Colors.WHITE)
        c.setFont("Helvetica", 10)
        c.drawString(25, h - 25, "PRODUCTION MODEL")

        c.setFont("Helvetica-Bold", 16)
        c.drawString(25, h - 45, "eu_model.pt")

        c.setFillColor(Colors.ACCENT_LIGHT)
        c.setFont("Helvetica", 9)
        c.drawString(25, h - 62, "V18 Architecture  |  Coil-Aware Focal Loss  |  13 Context Features")

        # Metrics row
        metrics = [
            ("32.1%", "Top 15% Precision"),
            ("1.67x", "Lift vs Random"),
            ("13.6%", "K2 Recall"),
            ("2,416", "Clean Patterns"),
        ]

        card_w = 95
        card_h = 55
        start_x = 25
        start_y = 12
        gap = (w - 50 - 4 * card_w) / 3

        for i, (value, label) in enumerate(metrics):
            x = start_x + i * (card_w + gap)

            # Card background
            c.setFillColor(colors.Color(1, 1, 1, alpha=0.1))
            c.roundRect(x, start_y, card_w, card_h, 3, fill=1, stroke=0)

            # Value
            c.setFillColor(Colors.WHITE)
            c.setFont("Helvetica-Bold", 20)
            c.drawCentredString(x + card_w/2, start_y + card_h - 25, value)

            # Label
            c.setFillColor(Colors.ACCENT_LIGHT)
            c.setFont("Helvetica", 7)
            c.drawCentredString(x + card_w/2, start_y + card_h - 40, label)


class PipelineDiagram(Flowable):
    """Clean, aligned pipeline architecture diagram"""
    def __init__(self, width=480, height=280):
        Flowable.__init__(self)
        self.width = width
        self.height = height

    def wrap(self, availWidth, availHeight):
        return (self.width, self.height)

    def draw(self):
        c = self.canv
        w, h = self.width, self.height

        # Title
        c.setFillColor(Colors.PRIMARY)
        c.setFont("Helvetica-Bold", 12)
        c.drawCentredString(w/2, h - 15, "Pipeline Architecture")

        c.setFillColor(Colors.GRAY_400)
        c.setFont("Helvetica", 8)
        c.drawCentredString(w/2, h - 28, "Five-stage temporal sequence processing pipeline")

        # Pipeline boxes - perfectly aligned
        stages = [
            ("00_detect", "Pattern\nDetection", Colors.VIZ_1),
            ("00b_label", "Outcome\nLabeling", Colors.VIZ_2),
            ("01_generate", "Sequence\nGeneration", Colors.VIZ_3),
            ("02_train", "Model\nTraining", Colors.VIZ_4),
            ("03_predict", "Inference", Colors.VIZ_5),
        ]

        box_w = 75
        box_h = 50
        total_width = len(stages) * box_w + (len(stages) - 1) * 20
        start_x = (w - total_width) / 2
        box_y = h - 100

        for i, (script, label, color) in enumerate(stages):
            x = start_x + i * (box_w + 20)

            # Box
            c.setFillColor(color)
            c.roundRect(x, box_y, box_w, box_h, 4, fill=1, stroke=0)

            # Label
            c.setFillColor(Colors.WHITE)
            c.setFont("Helvetica-Bold", 8)
            lines = label.split('\n')
            for j, line in enumerate(lines):
                c.drawCentredString(x + box_w/2, box_y + box_h - 18 - j * 10, line)

            # Script name
            c.setFillColor(Colors.GRAY_400)
            c.setFont("Courier", 6)
            c.drawCentredString(x + box_w/2, box_y - 10, script + ".py")

            # Arrow
            if i < len(stages) - 1:
                arrow_x = x + box_w + 3
                arrow_y = box_y + box_h/2

                c.setStrokeColor(Colors.GRAY_300)
                c.setLineWidth(1)
                c.line(arrow_x, arrow_y, arrow_x + 14, arrow_y)

                # Arrowhead
                c.setFillColor(Colors.GRAY_300)
                path = c.beginPath()
                path.moveTo(arrow_x + 14, arrow_y)
                path.lineTo(arrow_x + 10, arrow_y + 3)
                path.lineTo(arrow_x + 10, arrow_y - 3)
                path.close()
                c.drawPath(path, fill=1, stroke=0)

        # Data flow labels
        data_labels = ["OHLCV", "Candidates", "Labeled", "Sequences", "Predictions"]
        c.setFillColor(Colors.GRAY_400)
        c.setFont("Helvetica", 6)
        for i, label in enumerate(data_labels):
            x = start_x + i * (box_w + 20) + box_w/2
            if i < len(data_labels) - 1:
                x += 10 + box_w/2
            c.drawCentredString(x - (10 + box_w/2 if i < len(data_labels) - 1 else 0), box_y - 25, label)

        # Key innovations - clean card layout
        card_y = box_y - 130
        card_w = 145
        card_h = 70
        card_gap = (w - 3 * card_w) / 4

        innovations = [
            ("Two-Registry System", "Temporal Integrity",
             ["outcome_class = NULL", "100-day labeling rule"], Colors.RISK),
            ("Operation Clean Slate", "Data Quality",
             ["NMS: 81k → 2.4k", "Physics + Robust Scaling"], Colors.SUCCESS),
            ("Coil-Aware Focal Loss", "Training Innovation",
             ["Gradient amplification", "32.1% Top-15 precision"], Colors.CAUTION),
        ]

        for i, (title, subtitle, points, accent) in enumerate(innovations):
            x = card_gap + i * (card_w + card_gap)

            # Card background
            c.setFillColor(Colors.PAPER)
            c.setStrokeColor(Colors.GRAY_200)
            c.setLineWidth(0.5)
            c.roundRect(x, card_y, card_w, card_h, 4, fill=1, stroke=1)

            # Accent line
            c.setFillColor(accent)
            c.rect(x, card_y + card_h - 3, card_w, 3, fill=1, stroke=0)

            # Title
            c.setFillColor(Colors.PRIMARY)
            c.setFont("Helvetica-Bold", 8)
            c.drawString(x + 10, card_y + card_h - 18, title)

            # Subtitle
            c.setFillColor(Colors.GRAY_400)
            c.setFont("Helvetica", 6)
            c.drawString(x + 10, card_y + card_h - 28, subtitle)

            # Points
            c.setFillColor(Colors.GRAY_500)
            c.setFont("Helvetica", 7)
            for j, point in enumerate(points):
                c.drawString(x + 12, card_y + card_h - 43 - j * 10, "• " + point)


class NeuralArchitectureDiagram(Flowable):
    """Sophisticated neural network architecture visualization"""
    def __init__(self, width=480, height=320):
        Flowable.__init__(self)
        self.width = width
        self.height = height

    def wrap(self, availWidth, availHeight):
        return (self.width, self.height)

    def draw(self):
        c = self.canv
        w, h = self.width, self.height

        # Title
        c.setFillColor(Colors.PRIMARY)
        c.setFont("Helvetica-Bold", 12)
        c.drawCentredString(w/2, h - 15, "TemporalHybridModel V18")

        c.setFillColor(Colors.GRAY_400)
        c.setFont("Helvetica", 8)
        c.drawCentredString(w/2, h - 28, "Dual-branch architecture with context-query attention")

        # Central layout
        center_x = w / 2

        # Input layer
        input_y = h - 70

        # Temporal input
        c.setFillColor(Colors.VIZ_1)
        c.roundRect(center_x - 180, input_y, 120, 35, 4, fill=1, stroke=0)
        c.setFillColor(Colors.WHITE)
        c.setFont("Helvetica-Bold", 9)
        c.drawCentredString(center_x - 120, input_y + 22, "Temporal Input")
        c.setFont("Helvetica", 7)
        c.drawCentredString(center_x - 120, input_y + 8, "(batch, 20, 14)")

        # Context input
        c.setFillColor(Colors.VIZ_3)
        c.roundRect(center_x + 60, input_y, 120, 35, 4, fill=1, stroke=0)
        c.setFillColor(Colors.WHITE)
        c.setFont("Helvetica-Bold", 9)
        c.drawCentredString(center_x + 120, input_y + 22, "Context Input")
        c.setFont("Helvetica", 7)
        c.drawCentredString(center_x + 120, input_y + 8, "(batch, 13)")

        # Branch labels
        c.setFillColor(Colors.PRIMARY)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(center_x - 180, input_y - 18, "Branch A: Temporal Processing")
        c.drawString(center_x + 60, input_y - 18, "Branch B: Context Processing")

        # Processing layers - Branch A
        layer_y = input_y - 80

        # LSTM
        c.setFillColor(Colors.VIZ_1)
        c.setStrokeColor(colors.Color(Colors.VIZ_1.red * 0.8, Colors.VIZ_1.green * 0.8, Colors.VIZ_1.blue * 0.8))
        c.setLineWidth(1)
        c.roundRect(center_x - 200, layer_y, 70, 40, 3, fill=1, stroke=1)
        c.setFillColor(Colors.WHITE)
        c.setFont("Helvetica-Bold", 8)
        c.drawCentredString(center_x - 165, layer_y + 25, "LSTM")
        c.setFont("Helvetica", 6)
        c.drawCentredString(center_x - 165, layer_y + 10, "h=32, L=2")

        # CNN
        c.setFillColor(Colors.VIZ_2)
        c.roundRect(center_x - 120, layer_y, 70, 40, 3, fill=1, stroke=0)
        c.setFillColor(Colors.WHITE)
        c.setFont("Helvetica-Bold", 8)
        c.drawCentredString(center_x - 85, layer_y + 25, "CNN")
        c.setFont("Helvetica", 6)
        c.drawCentredString(center_x - 85, layer_y + 10, "k=[3,5]")

        # Self-Attention
        c.setFillColor(Colors.VIZ_4)
        c.roundRect(center_x - 160, layer_y - 50, 90, 35, 3, fill=1, stroke=0)
        c.setFillColor(Colors.WHITE)
        c.setFont("Helvetica-Bold", 8)
        c.drawCentredString(center_x - 115, layer_y - 30, "Self-Attention")
        c.setFont("Helvetica", 6)
        c.drawCentredString(center_x - 115, layer_y - 42, "heads=4")

        # Branch B layers
        # GRN
        c.setFillColor(Colors.VIZ_3)
        c.roundRect(center_x + 80, layer_y, 80, 40, 3, fill=1, stroke=0)
        c.setFillColor(Colors.WHITE)
        c.setFont("Helvetica-Bold", 8)
        c.drawCentredString(center_x + 120, layer_y + 25, "GRN")
        c.setFont("Helvetica", 6)
        c.drawCentredString(center_x + 120, layer_y + 10, "13 → 32 dim")

        # Context-Query Attention
        c.setFillColor(Colors.ACCENT)
        c.roundRect(center_x + 60, layer_y - 50, 120, 35, 3, fill=1, stroke=0)
        c.setFillColor(Colors.WHITE)
        c.setFont("Helvetica-Bold", 8)
        c.drawCentredString(center_x + 120, layer_y - 30, "Context-Query Attn")
        c.setFont("Helvetica", 6)
        c.drawCentredString(center_x + 120, layer_y - 42, "Q=ctx, K/V=temp")

        # Fusion layer
        fusion_y = layer_y - 110
        c.setFillColor(Colors.PRIMARY)
        c.roundRect(center_x - 80, fusion_y, 160, 40, 4, fill=1, stroke=0)
        c.setFillColor(Colors.WHITE)
        c.setFont("Helvetica-Bold", 10)
        c.drawCentredString(center_x, fusion_y + 25, "Fusion Layer")
        c.setFont("Helvetica", 7)
        c.drawCentredString(center_x, fusion_y + 10, "352 → 256 → 64")

        # Output
        output_y = fusion_y - 55
        c.setFillColor(Colors.ACCENT)
        c.roundRect(center_x - 60, output_y, 120, 35, 4, fill=1, stroke=0)
        c.setFillColor(Colors.WHITE)
        c.setFont("Helvetica-Bold", 10)
        c.drawCentredString(center_x, output_y + 22, "Output")
        c.setFont("Helvetica", 8)
        c.drawCentredString(center_x, output_y + 8, "3 Classes")

        # Class indicators
        class_y = output_y - 25
        classes = [
            ("Danger", Colors.RISK),
            ("Noise", Colors.GRAY_400),
            ("Target", Colors.SUCCESS),
        ]

        for i, (name, color) in enumerate(classes):
            x = center_x - 50 + i * 50
            c.setFillColor(color)
            c.circle(x, class_y, 5, fill=1, stroke=0)
            c.setFont("Helvetica", 6)
            c.setFillColor(Colors.GRAY_500)
            c.drawCentredString(x, class_y - 12, name)

        # Connection lines (simplified, elegant)
        c.setStrokeColor(Colors.GRAY_300)
        c.setLineWidth(0.5)

        # Vertical connectors
        c.line(center_x - 120, input_y, center_x - 120, layer_y + 40)
        c.line(center_x + 120, input_y, center_x + 120, layer_y + 40)
        c.line(center_x, fusion_y, center_x, output_y + 35)


class ContextFeaturesDiagram(Flowable):
    """Elegant context features visualization"""
    def __init__(self, width=480, height=320):
        Flowable.__init__(self)
        self.width = width
        self.height = height

    def wrap(self, availWidth, availHeight):
        return (self.width, self.height)

    def draw(self):
        c = self.canv
        w, h = self.width, self.height

        # Title
        c.setFillColor(Colors.PRIMARY)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(0, h - 15, "Context Features")

        c.setFillColor(Colors.GRAY_400)
        c.setFont("Helvetica", 9)
        c.drawString(0, h - 30, "13 features processed by GRN branch")

        # Two-column layout
        col_w = 230
        col_gap = 20

        # Original features (left column)
        left_x = 0
        section_y = h - 60

        c.setFillColor(Colors.VIZ_1)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(left_x, section_y, "Market Context (8)")

        original = [
            ("0", "float_turnover", "Accumulation activity"),
            ("1", "trend_position", "200-SMA position"),
            ("2", "base_duration", "Consolidation days"),
            ("3", "relative_volume", "Vol vs 50-day avg"),
            ("4", "distance_to_high", "% below 52w high"),
            ("5", "log_float", "Log shares outstanding"),
            ("6", "log_dollar_volume", "Log daily $ volume"),
            ("7", "relative_strength_spy", "RS vs SPY 90-day"),
        ]

        row_y = section_y - 20
        c.setFont("Helvetica", 8)

        for idx, name, desc in original:
            # Index
            c.setFillColor(Colors.GRAY_400)
            c.drawString(left_x, row_y, idx)

            # Name
            c.setFillColor(Colors.PRIMARY)
            c.setFont("Helvetica-Bold", 8)
            c.drawString(left_x + 20, row_y, name)

            # Description
            c.setFillColor(Colors.GRAY_500)
            c.setFont("Helvetica", 7)
            c.drawString(left_x + 120, row_y, desc)

            row_y -= 16

        # Coil features (right column) - highlighted
        right_x = col_w + col_gap
        section_y = h - 60

        # Highlight box
        c.setFillColor(Colors.SUCCESS_LIGHT)
        c.roundRect(right_x - 10, section_y - 130, col_w + 20, 145, 4, fill=1, stroke=0)

        c.setFillColor(Colors.SUCCESS)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(right_x, section_y, "NEW: Coil Features (5)")

        c.setFillColor(Colors.GRAY_500)
        c.setFont("Helvetica", 7)
        c.drawString(right_x, section_y - 12, "Bias-free pattern state at detection")

        coil = [
            ("8", "price_position_at_end", "(close-lower)/width"),
            ("9", "distance_to_danger", "(close-stop)/close"),
            ("10", "bbw_slope_5d", "BBW regression slope"),
            ("11", "vol_trend_5d", "Volume trend slope"),
            ("12", "coil_intensity", "Composite tension"),
        ]

        row_y = section_y - 35

        for idx, name, desc in coil:
            # Index
            c.setFillColor(Colors.SUCCESS)
            c.setFont("Helvetica-Bold", 8)
            c.drawString(right_x, row_y, idx)

            # Name
            c.setFillColor(Colors.PRIMARY)
            c.setFont("Helvetica-Bold", 8)
            c.drawString(right_x + 20, row_y, name)

            # Formula/Description
            c.setFillColor(Colors.GRAY_500)
            c.setFont("Courier", 6)
            c.drawString(right_x + 130, row_y, desc)

            row_y -= 18

        # Key insight box at bottom
        insight_y = h - 240
        c.setFillColor(Colors.CAUTION_LIGHT)
        c.setStrokeColor(Colors.ACCENT)
        c.setLineWidth(1)
        c.roundRect(0, insight_y, w, 50, 4, fill=1, stroke=1)

        c.setFillColor(Colors.ACCENT_DARK)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(15, insight_y + 32, "Key Insight")

        c.setFillColor(Colors.TEXT)
        c.setFont("Helvetica", 8)
        c.drawString(15, insight_y + 18, "price_position_at_end < 0.4 shows 5.5x better K2 hit rate than >= 0.6")
        c.drawString(15, insight_y + 6, "The coil features capture this predictive signal without any look-ahead bias.")


class PrecisionComparisonChart(Flowable):
    """Clean precision comparison visualization"""
    def __init__(self, width=480, height=200):
        Flowable.__init__(self)
        self.width = width
        self.height = height

    def wrap(self, availWidth, availHeight):
        return (self.width, self.height)

    def draw(self):
        c = self.canv
        w, h = self.width, self.height

        # Title
        c.setFillColor(Colors.PRIMARY)
        c.setFont("Helvetica-Bold", 11)
        c.drawCentredString(w/2, h - 15, "Model Comparison: Precision @ Top K%")

        # Chart area
        chart_x = 60
        chart_y = 35
        chart_w = w - 100
        chart_h = h - 70

        # Background
        c.setFillColor(Colors.PAPER)
        c.rect(chart_x, chart_y, chart_w, chart_h, fill=1, stroke=0)

        # Grid lines
        c.setStrokeColor(Colors.GRAY_200)
        c.setLineWidth(0.5)
        for i in range(5):
            y = chart_y + i * chart_h / 4
            c.line(chart_x, y, chart_x + chart_w, y)

            # Y-axis labels
            c.setFillColor(Colors.GRAY_400)
            c.setFont("Helvetica", 7)
            c.drawRightString(chart_x - 8, y - 3, f"{int(i * 12.5)}%")

        # Data
        percentiles = [5, 10, 15, 20]
        baseline = [42.2, 37.4, 30.7, 27.3]
        coil_focal = [22.2, 30.8, 32.1, 29.0]

        # Plot lines
        def get_point(i, val):
            x = chart_x + 30 + i * (chart_w - 60) / 3
            y = chart_y + (val / 50) * chart_h
            return x, y

        # Baseline line
        c.setStrokeColor(Colors.GRAY_400)
        c.setLineWidth(1.5)
        points = [get_point(i, v) for i, v in enumerate(baseline)]
        for i in range(len(points) - 1):
            c.line(points[i][0], points[i][1], points[i+1][0], points[i+1][1])

        # Coil focal line
        c.setStrokeColor(Colors.SUCCESS)
        c.setLineWidth(2)
        points = [get_point(i, v) for i, v in enumerate(coil_focal)]
        for i in range(len(points) - 1):
            c.line(points[i][0], points[i][1], points[i+1][0], points[i+1][1])

        # Data points
        for i, v in enumerate(baseline):
            x, y = get_point(i, v)
            c.setFillColor(Colors.WHITE)
            c.setStrokeColor(Colors.GRAY_400)
            c.setLineWidth(1)
            c.circle(x, y, 4, fill=1, stroke=1)

        for i, v in enumerate(coil_focal):
            x, y = get_point(i, v)
            c.setFillColor(Colors.SUCCESS)
            c.circle(x, y, 5, fill=1, stroke=0)

        # X-axis labels
        c.setFillColor(Colors.GRAY_500)
        c.setFont("Helvetica", 8)
        for i, pct in enumerate(percentiles):
            x = chart_x + 30 + i * (chart_w - 60) / 3
            c.drawCentredString(x, chart_y - 15, f"Top {pct}%")

        # Legend
        legend_x = chart_x + chart_w - 130
        legend_y = chart_y + chart_h - 15

        c.setFillColor(Colors.GRAY_400)
        c.circle(legend_x, legend_y, 4, fill=1, stroke=0)
        c.setFont("Helvetica", 7)
        c.drawString(legend_x + 10, legend_y - 3, "Baseline (no coil)")

        c.setFillColor(Colors.SUCCESS)
        c.circle(legend_x, legend_y - 15, 4, fill=1, stroke=0)
        c.setFont("Helvetica-Bold", 7)
        c.drawString(legend_x + 10, legend_y - 18, "Coil Focal Loss")

        # Highlight winner at Top 15%
        x15, y15 = get_point(2, coil_focal[2])
        c.setStrokeColor(Colors.ACCENT)
        c.setLineWidth(1)
        c.circle(x15, y15, 10, fill=0, stroke=1)

        c.setFillColor(Colors.ACCENT_DARK)
        c.setFont("Helvetica-Bold", 7)
        c.drawString(x15 + 12, y15 + 3, "32.1%")


class CoilFocalLossExplainer(Flowable):
    """Clean explanation of Coil-Aware Focal Loss"""
    def __init__(self, width=480, height=250):
        Flowable.__init__(self)
        self.width = width
        self.height = height

    def wrap(self, availWidth, availHeight):
        return (self.width, self.height)

    def draw(self):
        c = self.canv
        w, h = self.width, self.height

        # Title
        c.setFillColor(Colors.PRIMARY)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(0, h - 15, "Coil-Aware Focal Loss")

        c.setFillColor(Colors.GRAY_400)
        c.setFont("Helvetica", 8)
        c.drawString(0, h - 28, "Solving the K2 learning problem through gradient amplification")

        # Problem box
        box_y = h - 90
        box_h = 55

        c.setFillColor(Colors.RISK_LIGHT)
        c.roundRect(0, box_y, 225, box_h, 4, fill=1, stroke=0)

        c.setFillColor(Colors.RISK)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(12, box_y + box_h - 15, "Problem")

        c.setFillColor(Colors.TEXT)
        c.setFont("Helvetica", 8)
        c.drawString(12, box_y + box_h - 30, "Standard loss: K2 predictions = 0.04%")
        c.drawString(12, box_y + box_h - 42, "Model avoids K2 due to precision penalty")

        # Solution box
        c.setFillColor(Colors.SUCCESS_LIGHT)
        c.roundRect(245, box_y, 235, box_h, 4, fill=1, stroke=0)

        c.setFillColor(Colors.SUCCESS)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(257, box_y + box_h - 15, "Solution")

        c.setFillColor(Colors.TEXT)
        c.setFont("Helvetica", 8)
        c.drawString(257, box_y + box_h - 30, "Coil Focal Loss: K2 predictions = 6.4%")
        c.drawString(257, box_y + box_h - 42, "Amplify gradients for high-coil K2 patterns")

        # Formula
        formula_y = box_y - 50
        c.setFillColor(Colors.GRAY_100)
        c.roundRect(50, formula_y, 380, 35, 4, fill=1, stroke=0)

        c.setFillColor(Colors.PRIMARY)
        c.setFont("Courier-Bold", 10)
        c.drawCentredString(w/2, formula_y + 20, "loss_K2 = focal_loss × (1 + coil_intensity × 3.0)")

        c.setFillColor(Colors.GRAY_500)
        c.setFont("Helvetica", 7)
        c.drawCentredString(w/2, formula_y + 6, "coil_intensity = (1 - price_pos) × (1 - bbw_pctl) × vol_dryup")

        # Results comparison bars
        bar_y = formula_y - 90

        c.setFillColor(Colors.PRIMARY)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(0, bar_y + 65, "Results Comparison")

        models = [
            ("Baseline", 0.2, 30.7, Colors.GRAY_400),
            ("Coil Focal", 13.6, 32.1, Colors.SUCCESS),
            ("ASL + Coil", 22.6, 26.3, Colors.CAUTION),
        ]

        bar_w = 130
        bar_h = 8
        max_recall = 25
        max_precision = 35

        c.setFont("Helvetica", 7)
        c.setFillColor(Colors.GRAY_500)
        c.drawString(0, bar_y + 45, "K2 Recall")
        c.drawString(260, bar_y + 45, "Top 15% Precision")

        for i, (name, recall, precision, color) in enumerate(models):
            y = bar_y + 30 - i * 20

            # Model name
            c.setFillColor(Colors.GRAY_500)
            c.setFont("Helvetica", 7)
            c.drawString(0, y, name)

            # Recall bar
            recall_w = (recall / max_recall) * bar_w
            c.setFillColor(Colors.GRAY_200)
            c.rect(60, y - 2, bar_w, bar_h, fill=1, stroke=0)
            c.setFillColor(color)
            c.rect(60, y - 2, recall_w, bar_h, fill=1, stroke=0)
            c.setFillColor(Colors.GRAY_600)
            c.setFont("Helvetica", 6)
            c.drawString(60 + bar_w + 5, y, f"{recall}%")

            # Precision bar
            precision_w = (precision / max_precision) * bar_w
            c.setFillColor(Colors.GRAY_200)
            c.rect(320, y - 2, bar_w, bar_h, fill=1, stroke=0)
            c.setFillColor(color)
            c.rect(320, y - 2, precision_w, bar_h, fill=1, stroke=0)
            c.setFillColor(Colors.GRAY_600)
            c.drawString(320 + bar_w + 5, y, f"{precision}%")


class CodeBlock(Flowable):
    """Professional code block"""
    def __init__(self, code: str, title: str = "", width=460):
        Flowable.__init__(self)
        self.code = code
        self.title = title
        self.block_width = width

    def wrap(self, availWidth, availHeight):
        self.width = min(self.block_width, availWidth)
        lines = self.code.count('\n') + 1
        self.height = lines * 11 + 25 + (18 if self.title else 0)
        return (self.width, self.height)

    def draw(self):
        c = self.canv
        w, h = self.width, self.height

        # Background
        c.setFillColor(Colors.GRAY_600)
        c.roundRect(0, 0, w, h, 4, fill=1, stroke=0)

        # Title bar
        if self.title:
            c.setFillColor(Colors.GRAY_500)
            c.roundRect(0, h - 18, w, 18, 4, fill=1, stroke=0)
            c.rect(0, h - 18, w, 12, fill=1, stroke=0)

            c.setFillColor(Colors.GRAY_300)
            c.setFont("Helvetica", 7)
            c.drawString(10, h - 13, self.title)

        # Code
        c.setFillColor(Colors.GRAY_100)
        c.setFont("Courier", 7)

        y = h - (28 if self.title else 15)
        for line in self.code.split('\n'):
            if y > 8:
                # Simple highlighting
                if line.strip().startswith('#'):
                    c.setFillColor(Colors.GRAY_400)
                elif 'def ' in line or 'class ' in line:
                    c.setFillColor(Colors.VIZ_3)
                else:
                    c.setFillColor(Colors.GRAY_100)
                c.drawString(12, y, line[:75])
                y -= 11


# =============================================================================
# CHAPTER GENERATORS
# =============================================================================

def generate_cover_page(styles):
    """Generate elegant cover page"""
    content = []

    content.append(Spacer(1, 2*inch))

    # Main title
    content.append(Paragraph("TRANS", styles['MainTitle']))

    content.append(Spacer(1, 0.1*inch))

    content.append(Paragraph(
        "Temporal Sequence Architecture",
        styles['Subtitle']
    ))

    content.append(ElegantRule(width=200, thickness=1, accent_width=40))

    content.append(Spacer(1, 0.3*inch))

    content.append(Paragraph(
        "System Handbook",
        ParagraphStyle(
            'CoverSub',
            fontSize=16,
            textColor=Colors.GRAY_500,
            alignment=TA_CENTER,
            fontName='Helvetica',
            spaceAfter=60
        )
    ))

    content.append(Spacer(1, 0.5*inch))

    # Version card
    version_data = [
        ["Model", "eu_model.pt (V18 + Coil Focal Loss)"],
        ["Precision", "32.1% @ Top 15%"],
        ["Features", "14 temporal × 20 steps + 13 context"],
        ["Date", datetime.now().strftime("%B %d, %Y")],
    ]

    version_table = Table(version_data, colWidths=[100, 250])
    version_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TEXTCOLOR', (0, 0), (0, -1), Colors.GRAY_400),
        ('TEXTCOLOR', (1, 0), (1, -1), Colors.PRIMARY),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (0, -1), 15),
    ]))

    content.append(version_table)

    content.append(PageBreak())
    return content


def generate_executive_summary(styles):
    """Generate executive summary"""
    content = []

    content.append(Paragraph("Executive Summary", styles['Chapter']))
    content.append(ElegantRule(width=460, accent_width=50))
    content.append(Spacer(1, 0.25*inch))

    content.append(ExecutiveSummaryBanner())

    content.append(Spacer(1, 0.3*inch))

    content.append(Paragraph(
        "TRANS is a production-ready consolidation pattern detection system designed to identify "
        "micro and small-cap equities poised for significant upward moves. The system employs "
        "a hybrid neural architecture combining LSTM, CNN, and attention mechanisms with a "
        "novel Coil-Aware Focal Loss function that specifically addresses the challenge of "
        "learning rare but valuable target patterns.",
        styles['BodyText']
    ))

    content.append(Spacer(1, 0.2*inch))

    content.append(Paragraph("Key Innovations", styles['Section']))

    innovations = [
        ("<b>Two-Registry System</b> — Separates pattern detection from outcome labeling, "
         "guaranteeing strict temporal integrity with no look-ahead bias."),

        ("<b>13-Feature Context Branch</b> — Gated Residual Network processes 8 market context "
         "features plus 5 new coil features capturing pattern state at detection time."),

        ("<b>Coil-Aware Focal Loss</b> — Custom loss function amplifying gradients for K2 "
         "(Target) patterns with high coil intensity, solving the critical K2 learning problem."),

        ("<b>Operation Clean Slate</b> — NMS filter reduces 81,512 overlapping patterns to "
         "2,416 unique consolidation events, eliminating train/val/test leakage."),
    ]

    for item in innovations:
        content.append(Paragraph("• " + item, styles['BodyText']))

    content.append(Spacer(1, 0.2*inch))

    content.append(Paragraph("Performance Metrics", styles['Section']))

    metrics_data = [
        ['Metric', 'Value', 'Benchmark', 'Status'],
        ['Top 15% Precision', '32.1%', '> 30%', '✓ Pass'],
        ['Lift vs Random', '1.67x', '> 1.5x', '✓ Pass'],
        ['K2 Recall', '13.6%', '> 10%', '✓ Pass'],
        ['Clean Patterns', '2,416', 'N/A', '—'],
    ]

    metrics_table = Table(metrics_data, colWidths=[130, 90, 90, 80])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), Colors.PRIMARY),
        ('TEXTCOLOR', (0, 0), (-1, 0), Colors.WHITE),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, Colors.GRAY_200),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [Colors.WHITE, Colors.PAPER]),
        ('TEXTCOLOR', (3, 1), (3, -1), Colors.SUCCESS),
    ]))

    content.append(metrics_table)

    content.append(PageBreak())
    return content


def generate_architecture_chapter(styles):
    """Generate architecture chapter"""
    content = []

    content.append(Paragraph("Pipeline Architecture", styles['Chapter']))
    content.append(ElegantRule(width=460, accent_width=50))
    content.append(Spacer(1, 0.2*inch))

    content.append(Paragraph(
        "The TRANS pipeline processes market data through five distinct stages, each implemented "
        "as a separate module for maintainability and debugging. The architecture enforces strict "
        "temporal ordering to prevent any form of look-ahead bias.",
        styles['BodyText']
    ))

    content.append(Spacer(1, 0.15*inch))
    content.append(PipelineDiagram())
    content.append(Spacer(1, 0.2*inch))

    content.append(Paragraph("Two-Registry System", styles['Section']))

    content.append(Paragraph(
        "The two-registry architecture is fundamental to temporal integrity. Pattern detection "
        "and outcome labeling are completely decoupled processes:",
        styles['BodyText']
    ))

    registry_data = [
        ['Registry', 'Script', 'Output', 'Guarantee'],
        ['Candidate', '00_detect_patterns.py', 'candidates.parquet', 'outcome_class = NULL'],
        ['Outcome', '00b_label_outcomes.py', 'labeled.parquet', 'Labels only after 100 days'],
    ]

    registry_table = Table(registry_data, colWidths=[80, 140, 120, 140])
    registry_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), Colors.PRIMARY),
        ('TEXTCOLOR', (0, 0), (-1, 0), Colors.WHITE),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, Colors.GRAY_200),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [Colors.WHITE, Colors.PAPER]),
    ]))

    content.append(registry_table)

    content.append(PageBreak())
    return content


def generate_neural_network_chapter(styles):
    """Generate neural network chapter"""
    content = []

    content.append(Paragraph("Neural Network Architecture", styles['Chapter']))
    content.append(ElegantRule(width=460, accent_width=50))
    content.append(Spacer(1, 0.2*inch))

    content.append(Paragraph(
        "TemporalHybridModel V18 combines four processing branches: LSTM for sequence evolution, "
        "CNN for multi-scale pattern detection, Self-Attention for geometric structure analysis, "
        "and Context-Query Attention for regime-guided search.",
        styles['BodyText']
    ))

    content.append(Spacer(1, 0.15*inch))
    content.append(NeuralArchitectureDiagram())
    content.append(Spacer(1, 0.2*inch))

    content.append(Paragraph("Architecture Components", styles['Section']))

    arch_data = [
        ['Component', 'Configuration', 'Output', 'Purpose'],
        ['LSTM', 'hidden=32, layers=2', '32 dim', 'Sequence evolution'],
        ['CNN', 'kernels=[3,5], ch=[32,64]', '96 dim', 'Multi-scale patterns'],
        ['Self-Attention', 'heads=4', '96 dim', 'Geometric structure'],
        ['GRN', '13 → 32 dim', '32 dim', 'Context processing'],
        ['Context-Query Attn', 'Q=ctx, K/V=temp', '96 dim', 'Regime-guided search'],
        ['Fusion', '352 → 256 → 64', '64 dim', 'Feature combination'],
        ['Output', 'Linear(64, 3)', '3 classes', 'Classification'],
    ]

    arch_table = Table(arch_data, colWidths=[100, 120, 60, 140])
    arch_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), Colors.PRIMARY),
        ('TEXTCOLOR', (0, 0), (-1, 0), Colors.WHITE),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, Colors.GRAY_200),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [Colors.WHITE, Colors.PAPER]),
    ]))

    content.append(arch_table)

    content.append(PageBreak())
    return content


def generate_features_chapter(styles):
    """Generate features chapter"""
    content = []

    content.append(Paragraph("Feature Engineering", styles['Chapter']))
    content.append(ElegantRule(width=460, accent_width=50))
    content.append(Spacer(1, 0.2*inch))

    content.append(Paragraph(
        "The model processes two types of features: 14 temporal features computed for each of "
        "20 timesteps, and 13 context features computed once per pattern. The context branch "
        "includes 5 new coil features introduced in January 2026.",
        styles['BodyText']
    ))

    content.append(Spacer(1, 0.15*inch))
    content.append(ContextFeaturesDiagram())
    content.append(Spacer(1, 0.25*inch))

    content.append(Paragraph("Temporal Features (14)", styles['Section']))

    temporal_data = [
        ['Index', 'Feature', 'Normalization', 'Description'],
        ['0-3', 'open, high, low, close', 'Relativized to day 0', 'OHLC prices'],
        ['4', 'volume', 'Log ratio to day 0', 'Trading volume'],
        ['5-7', 'bbw_20, adx, vol_ratio', 'Z-score (global)', 'Technical indicators'],
        ['8-11', 'vol_dryup, var, nes, lpf', 'Robust (median/IQR)', 'Composite scores'],
        ['12-13', 'upper, lower boundary', 'Relativized to day 0', 'Pattern bounds'],
    ]

    temporal_table = Table(temporal_data, colWidths=[50, 130, 120, 140])
    temporal_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), Colors.VIZ_1),
        ('TEXTCOLOR', (0, 0), (-1, 0), Colors.WHITE),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, Colors.GRAY_200),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [Colors.WHITE, Colors.PAPER]),
    ]))

    content.append(temporal_table)

    content.append(PageBreak())
    return content


def generate_coil_loss_chapter(styles):
    """Generate coil focal loss chapter"""
    content = []

    content.append(Paragraph("Coil-Aware Focal Loss", styles['Chapter']))
    content.append(ElegantRule(width=460, accent_width=50))
    content.append(Spacer(1, 0.2*inch))

    content.append(Paragraph(
        "The Coil-Aware Focal Loss is the key innovation enabling effective K2 (Target) pattern "
        "learning. Standard loss functions caused the model to avoid K2 predictions entirely due "
        "to precision penalties. This custom loss amplifies gradients for high-coil K2 patterns.",
        styles['BodyText']
    ))

    content.append(Spacer(1, 0.15*inch))
    content.append(CoilFocalLossExplainer())
    content.append(Spacer(1, 0.25*inch))

    content.append(Paragraph("Implementation", styles['Section']))

    content.append(CodeBlock(
        "class CoilAwareFocalLoss(nn.Module):\n"
        "    def __init__(self, gamma=2.0, coil_weight=3.0):\n"
        "        self.gamma = gamma\n"
        "        self.coil_weight = coil_weight\n"
        "    \n"
        "    def forward(self, inputs, targets, coil_intensity):\n"
        "        ce = F.cross_entropy(inputs, targets, reduction='none')\n"
        "        pt = torch.exp(-ce)\n"
        "        focal = (1 - pt) ** self.gamma * ce\n"
        "        \n"
        "        # Amplify gradients for high-coil K2 patterns\n"
        "        is_k2 = (targets == 2)\n"
        "        boost = 1.0 + coil_intensity * self.coil_weight\n"
        "        focal[is_k2] *= boost[is_k2]\n"
        "        \n"
        "        return focal.mean()",
        title="losses/coil_focal_loss.py"
    ))

    content.append(Spacer(1, 0.2*inch))
    content.append(PrecisionComparisonChart())

    content.append(PageBreak())
    return content


def generate_cli_chapter(styles):
    """Generate CLI reference chapter"""
    content = []

    content.append(Paragraph("CLI Reference", styles['Chapter']))
    content.append(ElegantRule(width=460, accent_width=50))
    content.append(Spacer(1, 0.2*inch))

    content.append(Paragraph("Training Pipeline", styles['Section']))

    content.append(CodeBlock(
        "# Generate sequences with 13 context features\n"
        "python pipeline/01_generate_sequences.py \\\n"
        "    --input output/detected_patterns.parquet \\\n"
        "    --apply-nms --apply-physics-filter\n"
        "\n"
        "# Train with Coil-Aware Focal Loss\n"
        "python pipeline/02_train_temporal.py \\\n"
        "    --sequences output/sequences/*.h5 \\\n"
        "    --use-coil-focal --coil-strength-weight 3.0 \\\n"
        "    --epochs 100 --train-cutoff 2024-01-01\n"
        "\n"
        "# Generate predictions\n"
        "python pipeline/03_predict_temporal.py \\\n"
        "    --model output/models/eu_model.pt",
        title="Training Commands"
    ))

    content.append(Spacer(1, 0.2*inch))

    content.append(Paragraph("Key Flags", styles['Section']))

    flags_data = [
        ['Flag', 'Default', 'Description'],
        ['--use-coil-focal', 'Off', 'Enable Coil-Aware Focal Loss (recommended)'],
        ['--coil-strength-weight', '3.0', 'Coil intensity multiplier'],
        ['--apply-nms', 'Off', 'Enable NMS de-duplication'],
        ['--apply-physics-filter', 'Off', 'Remove untradeable patterns'],
        ['--train-cutoff', 'None', 'Date for train/val split'],
        ['--compile', 'Off', 'torch.compile for speedup'],
    ]

    flags_table = Table(flags_data, colWidths=[130, 70, 260])
    flags_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), Colors.PRIMARY),
        ('TEXTCOLOR', (0, 0), (-1, 0), Colors.WHITE),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (0, -1), 'Courier'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, Colors.GRAY_200),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [Colors.WHITE, Colors.PAPER]),
    ]))

    content.append(flags_table)

    content.append(PageBreak())
    return content


# =============================================================================
# MAIN BUILDER
# =============================================================================

def build_elite_handbook(output_path: str = "TRANS_Handbook_Elite.pdf"):
    """Build the elite handbook"""

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=0.75*inch,
        rightMargin=0.75*inch,
        topMargin=0.7*inch,
        bottomMargin=0.7*inch
    )

    styles = get_elite_styles()

    content = []
    content.extend(generate_cover_page(styles))
    content.extend(generate_executive_summary(styles))
    content.extend(generate_architecture_chapter(styles))
    content.extend(generate_neural_network_chapter(styles))
    content.extend(generate_features_chapter(styles))
    content.extend(generate_coil_loss_chapter(styles))
    content.extend(generate_cli_chapter(styles))

    doc.build(content)
    print(f"Elite handbook generated: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate TRANS Elite Handbook")
    parser.add_argument("--output", "-o", default="docs/TRANS_Handbook_Elite.pdf")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    build_elite_handbook(args.output)
