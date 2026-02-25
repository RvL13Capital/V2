#!/usr/bin/env python3
"""
TRAnS System Documentation Generator
Generates comprehensive PDF documentation of the system.
"""

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, ListFlowable, ListItem, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime
from pathlib import Path

def create_pdf():
    """Generate the TRAnS system documentation PDF."""

    output_path = Path("output/reports/TRAnS_System_Documentation.pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )

    # Styles
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#1a365d')
    )

    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Heading1'],
        fontSize=18,
        spaceBefore=20,
        spaceAfter=12,
        textColor=colors.HexColor('#2c5282')
    )

    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Heading2'],
        fontSize=14,
        spaceBefore=15,
        spaceAfter=8,
        textColor=colors.HexColor('#2b6cb0')
    )

    h3_style = ParagraphStyle(
        'H3',
        parent=styles['Heading3'],
        fontSize=12,
        spaceBefore=10,
        spaceAfter=6,
        textColor=colors.HexColor('#3182ce')
    )

    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=8,
        alignment=TA_JUSTIFY,
        leading=14
    )

    code_style = ParagraphStyle(
        'Code',
        parent=styles['Code'],
        fontSize=8,
        backColor=colors.HexColor('#f7fafc'),
        borderColor=colors.HexColor('#e2e8f0'),
        borderWidth=1,
        borderPadding=5,
        leftIndent=10,
        spaceAfter=10
    )

    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontSize=10,
        leftIndent=20,
        spaceAfter=4
    )

    # Build content
    content = []

    # ==========================================================================
    # TITLE PAGE
    # ==========================================================================
    content.append(Spacer(1, 2*inch))
    content.append(Paragraph("TRAnS", title_style))
    content.append(Paragraph("Temporal Retail Analysis System", ParagraphStyle(
        'Subtitle', parent=styles['Heading2'], fontSize=16, alignment=TA_CENTER,
        textColor=colors.HexColor('#4a5568')
    )))
    content.append(Spacer(1, 0.5*inch))
    content.append(Paragraph("Complete System Documentation", ParagraphStyle(
        'Subtitle2', parent=styles['Normal'], fontSize=12, alignment=TA_CENTER,
        textColor=colors.HexColor('#718096')
    )))
    content.append(Spacer(1, 1*inch))

    # Summary box
    summary_data = [
        ['Parameter', 'Value'],
        ['System Purpose', 'Micro/Small-Cap Consolidation Breakout Detection'],
        ['Account Size Target', '$10,000 - $100,000'],
        ['Risk Per Trade', '$250 (Fixed R-Unit)'],
        ['Max Position', '$5,000 or 4% ADV'],
        ['Current Performance', 'EU: 3.0x Lift (Top-15% Precision)'],
        ['Model Status', 'Production Ready (Jan 31, 2026)'],
        ['Data Pipeline', 'V22 Dynamic Window Labeling'],
    ]

    summary_table = Table(summary_data, colWidths=[2.5*inch, 3.5*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5282')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f7fafc')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
    ]))
    content.append(summary_table)

    content.append(Spacer(1, 1*inch))
    content.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                            ParagraphStyle('Date', parent=styles['Normal'],
                                          fontSize=9, alignment=TA_CENTER,
                                          textColor=colors.HexColor('#a0aec0'))))
    content.append(PageBreak())

    # ==========================================================================
    # TABLE OF CONTENTS
    # ==========================================================================
    content.append(Paragraph("Table of Contents", h1_style))

    toc_items = [
        "1. Executive Summary",
        "2. System Architecture Overview",
        "3. Pipeline Flow - Step by Step",
        "   3.1 Step 0: Pattern Detection",
        "   3.2 Step 0b: Outcome Labeling",
        "   3.3 Step 1: Sequence Generation",
        "   3.4 Step 2: Model Training",
        "   3.5 Step 3: Prediction & Execution",
        "4. Core Components Deep Dive",
        "   4.1 Pattern Scanner",
        "   4.2 R-Multiple Labeling Framework",
        "   4.3 Feature Engineering",
        "   4.4 Neural Network Architecture",
        "   4.5 Loss Functions",
        "5. Data Integrity & Safeguards",
        "6. Production Configuration",
        "7. Key Files Reference",
        "8. Appendix: Code Structure",
    ]

    for item in toc_items:
        indent = 20 if item.startswith("   ") else 0
        content.append(Paragraph(item, ParagraphStyle(
            'TOC', parent=styles['Normal'], fontSize=10, leftIndent=indent, spaceAfter=4
        )))

    content.append(PageBreak())

    # ==========================================================================
    # 1. EXECUTIVE SUMMARY
    # ==========================================================================
    content.append(Paragraph("1. Executive Summary", h1_style))

    content.append(Paragraph("""
    <b>TRAnS (Temporal Retail Analysis System)</b> is a production-grade machine learning system
    designed to identify consolidation patterns in micro/small-cap stocks that have high probability
    of significant upward breakouts. The system targets 40%+ gains by detecting "coiling spring"
    patterns where price consolidation builds potential energy before explosive moves.
    """, body_style))

    content.append(Paragraph("<b>Core Philosophy</b>", h3_style))
    content.append(Paragraph("""
    The system is built on three fundamental principles:
    """, body_style))

    principles = [
        "<b>Structural Risk Framework:</b> Uses R-multiples (risk multiples) instead of percentage gains to normalize outcomes across different price levels and volatilities.",
        "<b>Temporal Integrity:</b> Strict separation between pattern detection and outcome labeling prevents look-ahead bias - the most common cause of ML trading system failures.",
        "<b>Retail-Focused Execution:</b> Designed for accounts of $10K-$100K with position sizing that respects liquidity constraints (4% of ADV maximum).",
    ]

    for p in principles:
        content.append(Paragraph(f"• {p}", bullet_style))

    content.append(Spacer(1, 0.2*inch))
    content.append(Paragraph("<b>Current Performance (Jan 31, 2026)</b>", h3_style))

    perf_data = [
        ['Metric', 'Value', 'Interpretation'],
        ['Top-15% Lift', '3.0x', 'Model ranks patterns 3x better than random'],
        ['Top-15% Target Rate', '9.1%', 'vs 3.0% baseline - significant alpha'],
        ['Training Data', '6,240 patterns', 'After 73.6% deduplication'],
        ['Date Range', '2010-2025', '15 years of EU market data'],
        ['Model Architecture', 'V18 (LSTM+CNN)', 'Temporal + structural features'],
    ]

    perf_table = Table(perf_data, colWidths=[1.8*inch, 1.5*inch, 2.7*inch])
    perf_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#48bb78')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0fff4')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#9ae6b4')),
    ]))
    content.append(perf_table)

    content.append(Spacer(1, 0.2*inch))
    content.append(Paragraph("<b>Key Innovation: The Coiling Spring Model</b>", h3_style))
    content.append(Paragraph("""
    Consolidation patterns represent potential energy building in a stock. When volatility contracts
    (tight Bollinger Bands), volume dries up, and price coils into a tight range, the stock is
    accumulating energy. The system detects these patterns and predicts which ones will "spring"
    into significant moves vs. which will fail or drift sideways.
    """, body_style))

    content.append(PageBreak())

    # ==========================================================================
    # 2. SYSTEM ARCHITECTURE OVERVIEW
    # ==========================================================================
    content.append(Paragraph("2. System Architecture Overview", h1_style))

    content.append(Paragraph("""
    The TRAnS system follows a <b>four-step temporal pipeline</b> with strict separation of concerns.
    Each step operates independently with well-defined inputs and outputs, ensuring reproducibility
    and preventing data leakage.
    """, body_style))

    content.append(Paragraph("<b>Pipeline Architecture Diagram</b>", h3_style))

    # ASCII-style pipeline diagram as table
    pipeline_diagram = """
    ┌─────────────────────────────────────────────────────────────┐
    │                    RAW OHLCV DATA                           │
    │              (GCS Bucket / Local Cache)                     │
    └─────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  STEP 0: PATTERN DETECTION (00_detect_patterns.py)         │
    │  • ConsolidationPatternScanner scans for tight ranges      │
    │  • Two-phase state machine (Qualification → Active)        │
    │  • Output: candidate_patterns.parquet (NO labels)          │
    └─────────────────────────┬───────────────────────────────────┘
                              │
                              │ [40+ day ripening delay]
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  STEP 0b: OUTCOME LABELING (00b_label_outcomes.py)         │
    │  • R-multiple framework (structural risk measurement)      │
    │  • Dynamic window: 10-60 days based on volatility          │
    │  • Volume confirmation: 3 consecutive days required        │
    │  • Output: labeled_patterns.parquet (with outcome_class)   │
    └─────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  STEP 1: SEQUENCE GENERATION (01_generate_sequences.py)    │
    │  • Convert patterns to 20×10 temporal tensors              │
    │  • NMS filter: Remove 71% overlapping patterns             │
    │  • Physics filter: Remove untradeable (<$50k liquidity)    │
    │  • Output: sequences.npy + metadata.parquet + context      │
    └─────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  STEP 2: MODEL TRAINING (02_train_temporal.py)             │
    │  • HybridFeatureNetwork (LSTM + CNN architecture)          │
    │  • Coil-Aware Focal Loss (amplifies learning on coils)     │
    │  • Temporal split (pattern-level, no leakage)              │
    │  • Output: best_model.pt (production checkpoint)           │
    └─────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  STEP 3: PREDICTION & EXECUTION (03_predict_temporal.py)   │
    │  • Expected Value (EV) calculation from probabilities      │
    │  • Signal generation (STRONG/GOOD/MODERATE/WEAK/AVOID)     │
    │  • Position sizing with risk & liquidity limits            │
    │  • Output: nightly_orders.csv (execution plan)             │
    └─────────────────────────────────────────────────────────────┘
    """

    content.append(Paragraph(pipeline_diagram.replace('\n', '<br/>'), code_style))

    content.append(Paragraph("<b>Orchestration</b>", h3_style))
    content.append(Paragraph("""
    The <b>main_pipeline.py</b> file orchestrates all steps with region-aware configuration
    (EU/US/GLOBAL), artifact validation between steps, and subprocess isolation for memory
    efficiency. It supports both full pipeline execution and individual step runs.
    """, body_style))

    content.append(Paragraph("<b>Standard Execution Command</b>", h3_style))
    cmd = """python main_pipeline.py --region EU --full-pipeline \\
    --apply-nms --apply-physics-filter --mode training \\
    --use-coil-focal --epochs 100 --disable-trinity"""
    content.append(Paragraph(cmd, code_style))

    content.append(PageBreak())

    # ==========================================================================
    # 3. PIPELINE FLOW - STEP BY STEP
    # ==========================================================================
    content.append(Paragraph("3. Pipeline Flow - Step by Step", h1_style))

    # STEP 0
    content.append(Paragraph("3.1 Step 0: Pattern Detection", h2_style))
    content.append(Paragraph("<b>File:</b> pipeline/00_detect_patterns.py", body_style))
    content.append(Paragraph("<b>Core Class:</b> ConsolidationPatternScanner (core/pattern_scanner.py)", body_style))

    content.append(Spacer(1, 0.1*inch))
    content.append(Paragraph("<b>Purpose</b>", h3_style))
    content.append(Paragraph("""
    Scans raw OHLCV data to identify consolidation patterns - periods where a stock trades in a
    tight range with low volume, suggesting accumulation or distribution before a significant move.
    """, body_style))

    content.append(Paragraph("<b>Two-Phase State Machine</b>", h3_style))

    phase_data = [
        ['Phase', 'Duration', 'Criteria', 'Purpose'],
        ['Qualification', 'Days 1-10',
         'BBW < 30th %ile\nADX < 32\nVolume < 35% avg\nRange < 65% avg',
         'Identify volatility contraction'],
        ['Active', 'Day 10+',
         'Monitor until:\n• Breakout (close > upper)\n• Failure (close < lower)\n• Timeout (150 days)',
         'Track pattern resolution'],
    ]

    phase_table = Table(phase_data, colWidths=[1*inch, 0.9*inch, 2*inch, 2*inch])
    phase_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4299e1')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ebf8ff')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#90cdf4')),
    ]))
    content.append(phase_table)

    content.append(Spacer(1, 0.1*inch))
    content.append(Paragraph("<b>Qualification Criteria Explained</b>", h3_style))

    criteria_items = [
        "<b>BBW (Bollinger Band Width) < 30th percentile:</b> Measures volatility contraction. When BBW is in the lowest 30% of its historical range, the stock is trading in an unusually tight range - the 'coil' is forming.",
        "<b>ADX < 32:</b> Average Directional Index measures trend strength. Low ADX indicates the stock is not trending strongly, suggesting consolidation rather than a directional move.",
        "<b>Volume < 35% of 20-day average:</b> Suppressed volume indicates lack of participation - either accumulation by informed buyers or simply disinterest. Both can precede significant moves.",
        "<b>Daily Range < 65% of 20-day average:</b> Tight daily ranges (High-Low) confirm the coiling pattern geometrically.",
    ]

    for item in criteria_items:
        content.append(Paragraph(f"• {item}", bullet_style))

    content.append(Spacer(1, 0.1*inch))
    content.append(Paragraph("<b>Output Schema</b>", h3_style))

    output_cols = [
        ['Column', 'Type', 'Description'],
        ['pattern_id', 'string', 'Unique identifier (ticker_date format)'],
        ['ticker', 'string', 'Stock symbol'],
        ['start_date', 'date', 'Qualification phase start'],
        ['end_date', 'date', 'Qualification complete (trigger point)'],
        ['upper_boundary', 'float', 'Resistance level'],
        ['lower_boundary', 'float', 'Support level'],
        ['box_width', 'float', 'Price range (upper - lower)'],
        ['days_in_pattern', 'int', 'Pattern duration'],
    ]

    output_table = Table(output_cols, colWidths=[1.3*inch, 0.8*inch, 3.8*inch])
    output_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#718096')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f7fafc')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
    ]))
    content.append(output_table)

    content.append(Paragraph("""
    <b>Critical Note:</b> The output contains NO outcome labels at this stage. This is intentional -
    separating detection from labeling prevents look-ahead bias.
    """, body_style))

    content.append(PageBreak())

    # STEP 0b
    content.append(Paragraph("3.2 Step 0b: Outcome Labeling", h2_style))
    content.append(Paragraph("<b>File:</b> pipeline/00b_label_outcomes.py", body_style))

    content.append(Paragraph("<b>Purpose</b>", h3_style))
    content.append(Paragraph("""
    Applies the <b>Structural Risk Labeling Framework</b> to classify pattern outcomes using
    R-multiples (risk multiples) rather than percentage gains. This normalizes outcomes across
    different price levels and volatilities.
    """, body_style))

    content.append(Paragraph("<b>R-Multiple Framework</b>", h3_style))

    rmult_code = """
    Technical_Floor = Pattern_Lower_Boundary  # Chart support level
    Trigger_Price = Pattern_Upper_Boundary + $0.01  # Breakout confirmation
    R_Dollar = Trigger_Price - Technical_Floor  # Structural risk per share
    Entry_Price = MAX(Trigger_Price, Open[t+1])  # Gap-adjusted entry

    Example:
        Upper = $10.00, Lower = $9.00
        Trigger = $10.01
        R_Dollar = $10.01 - $9.00 = $1.01
        If stock hits $13.04, that's +3R (target achieved)
        If stock drops to $8.99, that's stop-out (danger)
    """
    content.append(Paragraph(rmult_code, code_style))

    content.append(Paragraph("<b>Label Classes</b>", h3_style))

    label_data = [
        ['Class', 'Name', 'Condition', 'Strategic Value', 'Interpretation'],
        ['0', 'DANGER', 'Close < Technical Floor', '-2.0', 'Pattern failed, stop hit'],
        ['1', 'NOISE', 'Neither target nor stop', '-0.1', 'Pattern drifted, opportunity cost'],
        ['2', 'TARGET', '+3R with volume confirm', '+5.0', 'Successful breakout'],
    ]

    label_table = Table(label_data, colWidths=[0.5*inch, 0.7*inch, 1.5*inch, 1*inch, 2.2*inch])
    label_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#805ad5')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('BACKGROUND', (0, 1), (0, 1), colors.HexColor('#fed7d7')),
        ('BACKGROUND', (0, 2), (0, 2), colors.HexColor('#fefcbf')),
        ('BACKGROUND', (0, 3), (0, 3), colors.HexColor('#c6f6d5')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#b794f4')),
    ]))
    content.append(label_table)

    content.append(Spacer(1, 0.1*inch))
    content.append(Paragraph("<b>Volume Confirmation (Critical - Jan 2026 Fix)</b>", h3_style))
    content.append(Paragraph("""
    To qualify as TARGET, a pattern must show <b>sustained buying pressure</b>, not just a
    price spike. This prevents false positives from dormant stocks gapping on zero volume.
    """, body_style))

    vol_req = [
        "Volume > 2x 20-day average (relative threshold)",
        "Dollar volume > $50,000/day (absolute threshold)",
        "<b>3 consecutive days</b> meeting BOTH conditions",
    ]
    for v in vol_req:
        content.append(Paragraph(f"• {v}", bullet_style))

    content.append(Spacer(1, 0.1*inch))
    content.append(Paragraph("<b>Dynamic Labeling Window (V22)</b>", h3_style))
    content.append(Paragraph("""
    Instead of a fixed 40-day outcome window, V22 adapts to stock volatility:
    """, body_style))

    window_code = """
    volatility_proxy = risk_width_pct (or box_width / price)
    dynamic_window = clip(1 / volatility_proxy, MIN=10, MAX=60)

    Examples:
        10% volatility → 10-day window (high-vol stocks resolve fast)
        5% volatility  → 20-day window
        2% volatility  → 50-day window (low-vol needs more time)
    """
    content.append(Paragraph(window_code, code_style))

    content.append(PageBreak())

    # STEP 1
    content.append(Paragraph("3.3 Step 1: Sequence Generation", h2_style))
    content.append(Paragraph("<b>File:</b> pipeline/01_generate_sequences.py", body_style))

    content.append(Paragraph("<b>Purpose</b>", h3_style))
    content.append(Paragraph("""
    Converts labeled patterns into fixed-size tensors suitable for neural network training.
    Each pattern becomes a <b>20×10 tensor</b> (20 timesteps × 10 features per timestep).
    """, body_style))

    content.append(Paragraph("<b>10 Temporal Features (Per Timestep)</b>", h3_style))

    feat_data = [
        ['Index', 'Feature', 'Description', 'Purpose'],
        ['0', 'open', 'Opening price (normalized)', 'Daily price action'],
        ['1', 'high', 'Daily high (normalized)', 'Intraday range'],
        ['2', 'low', 'Daily low (normalized)', 'Intraday range'],
        ['3', 'close', 'Closing price (normalized)', 'Settlement price'],
        ['4', 'volume', 'Trading volume (log-normalized)', 'Participation level'],
        ['5', 'bbw_20', 'Bollinger Band Width', 'Volatility contraction'],
        ['6', 'adx', 'Average Directional Index', 'Trend strength'],
        ['7', 'volume_ratio_20', 'Volume / 20-day MA', 'Relative activity'],
        ['8', 'upper_slope', 'Rolling slope of highs', 'Pattern geometry'],
        ['9', 'lower_slope', 'Rolling slope of lows', 'Pattern geometry'],
    ]

    feat_table = Table(feat_data, colWidths=[0.5*inch, 1.2*inch, 2*inch, 2.2*inch])
    feat_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#38a169')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0fff4')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#9ae6b4')),
    ]))
    content.append(feat_table)

    content.append(Spacer(1, 0.1*inch))
    content.append(Paragraph("<b>Filtering Pipeline</b>", h3_style))
    content.append(Paragraph("""
    Raw patterns go through multiple filters to ensure data quality and tradeability:
    """, body_style))

    filter_data = [
        ['Filter', 'Removal Rate', 'Purpose'],
        ['NMS (Non-Max Suppression)', '71-73%', 'Remove overlapping patterns from same event'],
        ['Physics Filter', 'Variable', 'Remove untradeable: <$50k liquidity, >0.5R gap'],
        ['Deduplication', '~0.6%', 'Remove cross-ticker duplicates (ETF overlap)'],
    ]

    filter_table = Table(filter_data, colWidths=[1.8*inch, 1.2*inch, 2.9*inch])
    filter_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ed8936')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#fffaf0')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#fbd38d')),
    ]))
    content.append(filter_table)

    content.append(Spacer(1, 0.1*inch))
    content.append(Paragraph("<b>Log-Diff Transformation (Dormant Stock Fix)</b>", h3_style))
    content.append(Paragraph("""
    Volume ratios can explode on dormant stocks (e.g., vol_20d / vol_60d when vol_60d ≈ 0).
    The log-diff transformation handles this gracefully:
    """, body_style))

    logdiff_code = """
    log_diff(a, b) = log1p(a) - log1p(b)

    • Handles zeros: log1p(0) = 0
    • No division by zero
    • Output naturally bounded (~-10 to +10)

    Example:
        vol_20d=0, vol_60d=100   → -4.6 (dormant)
        vol_20d=500, vol_60d=100 → +1.6 (awakening)
        vol_20d=5000, vol_60d=100 → +3.9 (volume shock)
    """
    content.append(Paragraph(logdiff_code, code_style))

    content.append(PageBreak())

    # STEP 2
    content.append(Paragraph("3.4 Step 2: Model Training", h2_style))
    content.append(Paragraph("<b>File:</b> pipeline/02_train_temporal.py", body_style))
    content.append(Paragraph("<b>Model:</b> models/temporal_hybrid_unified.py", body_style))

    content.append(Paragraph("<b>HybridFeatureNetwork Architecture</b>", h3_style))
    content.append(Paragraph("""
    The model combines temporal pattern recognition (LSTM + CNN) with static context features
    through a multi-branch architecture:
    """, body_style))

    arch_diagram = """
    Temporal Input (20×10)              Context Input (24 features)
           │                                      │
           ▼                                      ▼
    ┌─────────────┐                      ┌─────────────┐
    │  BRANCH A   │                      │  BRANCH B   │
    │  LSTM (2L)  │                      │    GRN      │
    │  + CNN      │                      │  (Gated    │
    │  (dynamics) │                      │  Residual) │
    └──────┬──────┘                      └──────┬─────┘
           │                                    │
           │            ┌──────────┐           │
           └───────────►│  Concat  │◄──────────┘
                        │  + Dense │
                        └────┬─────┘
                             │
                        ┌────▼────┐
                        │ Softmax │
                        │ (3-way) │
                        └─────────┘
                             │
                          [D, N, T]
    """
    content.append(Paragraph(arch_diagram.replace('\n', '<br/>'), code_style))

    content.append(Paragraph("<b>Key Components</b>", h3_style))

    comp_items = [
        "<b>2-Layer LSTM:</b> Captures temporal dependencies in the 20-day sequence. Hidden state encodes the 'narrative' of how the pattern evolved.",
        "<b>Dual CNN (conv_3 + conv_5):</b> Parallel convolutions with kernel sizes 3 and 5 capture local price structures at different scales.",
        "<b>Gated Residual Network (GRN):</b> Processes 24 static context features with gating mechanism to select relevant market context.",
        "<b>Fusion Layer:</b> Concatenates LSTM output, CNN features, and context embedding before final classification.",
    ]
    for item in comp_items:
        content.append(Paragraph(f"• {item}", bullet_style))

    content.append(Spacer(1, 0.1*inch))
    content.append(Paragraph("<b>Coil-Aware Focal Loss</b>", h3_style))
    content.append(Paragraph("""
    Standard cross-entropy doesn't account for pattern quality. Coil-Aware Focal Loss amplifies
    learning on high-quality "coiled" patterns:
    """, body_style))

    loss_code = """
    Standard Focal Loss: FL(pt) = -(1-pt)^gamma * log(pt)

    Coil Boost (for Target class with high coil_intensity):
        boost = 1.0 + (coil_intensity * coil_weight)
        loss = FL(pt) * boost

    Example: coil_intensity=0.8, coil_weight=3.0
             → boost = 1.0 + 0.8 * 3.0 = 3.4× loss amplification
             → Model learns harder from high-quality coil patterns
    """
    content.append(Paragraph(loss_code, code_style))

    content.append(Spacer(1, 0.1*inch))
    content.append(Paragraph("<b>Temporal Split (Critical for Integrity)</b>", h3_style))
    content.append(Paragraph("""
    The training uses <b>pattern-level temporal splitting</b> to prevent data leakage:
    """, body_style))

    split_data = [
        ['Split', 'Date Range', 'Patterns', 'Purpose'],
        ['Train', '< 2024-01-01', '~85%', 'Model learning'],
        ['Validation', '2024-01-01 to 2024-07-01', '~5%', 'Hyperparameter tuning'],
        ['Test', '>= 2024-07-01', '~10%', 'Final evaluation'],
    ]

    split_table = Table(split_data, colWidths=[1*inch, 2*inch, 0.8*inch, 2.1*inch])
    split_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e53e3e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#fff5f5')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#fc8181')),
    ]))
    content.append(split_table)

    content.append(Paragraph("""
    <b>Why --disable-trinity is REQUIRED:</b> Trinity mode groups overlapping patterns into clusters
    and splits by cluster. This caused a bug where 99.7% of data went to training and only 0.1% to
    validation. Pattern-level splitting ensures proper temporal separation.
    """, body_style))

    content.append(PageBreak())

    # STEP 3
    content.append(Paragraph("3.5 Step 3: Prediction & Execution", h2_style))
    content.append(Paragraph("<b>File:</b> pipeline/03_predict_temporal.py", body_style))

    content.append(Paragraph("<b>Expected Value (EV) Calculation</b>", h3_style))
    content.append(Paragraph("""
    The model outputs probabilities for each class. Expected Value combines these with strategic
    values to produce a single actionable score:
    """, body_style))

    ev_code = """
    EV = P(Danger) × (-2.0) + P(Noise) × (-0.1) + P(Target) × (+5.0)

    Example:
        P(D)=0.25, P(N)=0.60, P(T)=0.15
        EV = 0.25×(-2.0) + 0.60×(-0.1) + 0.15×(5.0)
           = -0.50 + -0.06 + 0.75
           = +0.19 (WEAK signal)
    """
    content.append(Paragraph(ev_code, code_style))

    content.append(Paragraph("<b>Signal Thresholds</b>", h3_style))

    signal_data = [
        ['Signal', 'EV Threshold', 'Action', 'Confidence'],
        ['STRONG', '>= 5.0', 'High priority trade', 'Very high'],
        ['GOOD', '>= 3.0', 'Standard trade', 'High'],
        ['MODERATE', '>= 1.0', 'Consider with caution', 'Medium'],
        ['WEAK', '>= 0.0', 'Low conviction', 'Low'],
        ['AVOID', '< 0.0', 'Do not trade', 'Negative EV'],
    ]

    signal_table = Table(signal_data, colWidths=[1*inch, 1*inch, 1.8*inch, 2.1*inch])
    signal_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2b6cb0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#c6f6d5')),
        ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#c6f6d5')),
        ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#fefcbf')),
        ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor('#fefcbf')),
        ('BACKGROUND', (0, 5), (-1, 5), colors.HexColor('#fed7d7')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#90cdf4')),
    ]))
    content.append(signal_table)

    content.append(Spacer(1, 0.1*inch))
    content.append(Paragraph("<b>Position Sizing</b>", h3_style))

    sizing_code = """
    Risk-Based Sizing:
        risk_per_share = trigger_price - lower_boundary
        shares_from_risk = $250 / risk_per_share

    Clamping:
        1. Never exceed $5,000 per position
        2. Never exceed 4% of average daily volume

    Final shares = min(shares_from_risk, shares_from_cap, shares_from_liquidity)
    """
    content.append(Paragraph(sizing_code, code_style))

    content.append(Paragraph("<b>Execution Strategy: Buy Stop Limit</b>", h3_style))

    exec_items = [
        "<b>Stop Price:</b> Upper boundary + $0.01 (breakout trigger)",
        "<b>Limit Price:</b> Stop + 0.5×R (caps slippage)",
        "<b>Stop Loss:</b> Lower boundary (technical floor)",
        "<b>Gap Protection:</b> If Open > Limit, order canceled (don't chase gaps)",
    ]
    for item in exec_items:
        content.append(Paragraph(f"• {item}", bullet_style))

    content.append(Paragraph("""
    <b>Exit Plan (Justice Tiers):</b> Sell 50% at +3R, 25% at +5R, hold 25% with trailing stop
    for +10R runners. Weighted average: 5.25R if all targets hit.
    """, body_style))

    content.append(PageBreak())

    # ==========================================================================
    # 4. CORE COMPONENTS DEEP DIVE
    # ==========================================================================
    content.append(Paragraph("4. Core Components Deep Dive", h1_style))

    content.append(Paragraph("4.1 Pattern Scanner", h2_style))
    content.append(Paragraph("<b>File:</b> core/pattern_scanner.py", body_style))

    content.append(Paragraph("""
    The ConsolidationPatternScanner implements a finite state machine that tracks each ticker
    through qualification, active monitoring, and resolution phases. Key methods:
    """, body_style))

    scanner_methods = [
        "<b>scan_ticker(df, ticker):</b> Main entry point. Iterates through daily bars, updating state machine.",
        "<b>check_qualification(row):</b> Evaluates if current bar meets BBW, ADX, volume, and range criteria.",
        "<b>update_boundaries(row):</b> Adjusts upper/lower boundaries as pattern evolves.",
        "<b>check_breakout(row):</b> Detects if close > upper (with confirmation rules).",
        "<b>check_failure(row):</b> Detects if close < lower (pattern invalidated).",
    ]
    for m in scanner_methods:
        content.append(Paragraph(f"• {m}", bullet_style))

    content.append(Spacer(1, 0.15*inch))
    content.append(Paragraph("4.2 R-Multiple Labeling Framework", h2_style))
    content.append(Paragraph("""
    The labeling system uses <b>path-dependent outcome classification</b>. Unlike simple
    percentage-based labels, R-multiples account for the structural risk of each pattern:
    """, body_style))

    content.append(Paragraph("""
    <b>Why R-Multiples Matter:</b> A 10% gain on a pattern with 2% risk is 5R (excellent).
    The same 10% gain on a pattern with 15% risk is only 0.67R (poor). R-multiples normalize
    for the inherent risk in each setup.
    """, body_style))

    content.append(Spacer(1, 0.15*inch))
    content.append(Paragraph("4.3 Feature Engineering", h2_style))
    content.append(Paragraph("<b>File:</b> config/feature_registry.py", body_style))

    content.append(Paragraph("""
    The FeatureRegistry centralizes all feature definitions with:
    """, body_style))

    registry_items = [
        "Feature name → index mapping (prevents hardcoded indices)",
        "Type classification (bounded, ratio, log_diff, slope)",
        "Valid ranges and default values",
        "Transformation requirements (log1p, clip, normalize)",
    ]
    for item in registry_items:
        content.append(Paragraph(f"• {item}", bullet_style))

    content.append(Spacer(1, 0.15*inch))
    content.append(Paragraph("4.4 Neural Network Architecture", h2_style))
    content.append(Paragraph("<b>File:</b> models/temporal_hybrid_unified.py", body_style))

    content.append(Paragraph("""
    The HybridFeatureNetwork supports multiple ablation modes for experimentation:
    """, body_style))

    mode_data = [
        ['Mode', 'LSTM', 'CNN', 'Context GRN', 'Use Case'],
        ['v18_full', 'Yes', 'Yes', 'Optional', 'Production (default)'],
        ['concat', 'No', 'Yes', 'Yes', 'Baseline comparison'],
        ['lstm', 'Yes', 'Yes', 'Yes', 'Full features'],
        ['cqa_only', 'No', 'Yes + CQA', 'Yes', 'Attention study'],
    ]

    mode_table = Table(mode_data, colWidths=[1*inch, 0.6*inch, 0.8*inch, 1*inch, 2.5*inch])
    mode_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#553c9a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#faf5ff')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#b794f4')),
    ]))
    content.append(mode_table)

    content.append(Spacer(1, 0.15*inch))
    content.append(Paragraph("4.5 Loss Functions", h2_style))
    content.append(Paragraph("<b>File:</b> losses/coil_focal_loss.py", body_style))

    content.append(Paragraph("""
    <b>CoilAwareFocalLoss</b> combines focal loss (for class imbalance) with coil-intensity
    weighting (for pattern quality). The loss function has three components:
    """, body_style))

    loss_items = [
        "<b>Focal modulation:</b> (1-pt)^gamma down-weights easy examples",
        "<b>Class weights:</b> Square root of inverse frequency (handles imbalance)",
        "<b>Coil boost:</b> Amplifies gradient for high-quality Target patterns",
    ]
    for item in loss_items:
        content.append(Paragraph(f"• {item}", bullet_style))

    content.append(PageBreak())

    # ==========================================================================
    # 5. DATA INTEGRITY & SAFEGUARDS
    # ==========================================================================
    content.append(Paragraph("5. Data Integrity & Safeguards", h1_style))

    content.append(Paragraph("""
    The system includes multiple safeguards to prevent common ML trading pitfalls:
    """, body_style))

    content.append(Paragraph("<b>The '74x Bug' Prevention</b>", h3_style))
    content.append(Paragraph("""
    An early version had a data loading bug that duplicated rows 74×, causing massive overfitting.
    The pipeline now auto-blocks if duplication ratio > 1.1×.
    """, body_style))

    content.append(Paragraph("<b>Automated Checks (utils/data_integrity.py)</b>", h3_style))

    check_data = [
        ['Check', 'Threshold', 'Action if Failed'],
        ['Duplication ratio', '> 1.1×', 'CRITICAL ERROR - pipeline stops'],
        ['Temporal leakage', 'val dates <= train max', 'CRITICAL ERROR - pipeline stops'],
        ['Feature leakage', 'outcome cols in features', 'CRITICAL ERROR - pipeline stops'],
        ['Statistical power', '< 100 Target events', 'WARNING - results unreliable'],
        ['Metadata alignment', 'HDF5/parquet mismatch', 'CRITICAL ERROR - pipeline stops'],
    ]

    check_table = Table(check_data, colWidths=[1.5*inch, 1.5*inch, 2.9*inch])
    check_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#c53030')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#fff5f5')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#fc8181')),
    ]))
    content.append(check_table)

    content.append(Spacer(1, 0.15*inch))
    content.append(Paragraph("<b>Look-Ahead Bias Prevention</b>", h3_style))

    lookahead_items = [
        "<b>40-day ripening delay:</b> Patterns are not labeled until 40+ days after detection",
        "<b>Event anchor restriction:</b> NMS only searches up to pattern.end_date",
        "<b>FUTURE_* columns:</b> Post-pattern metadata blocked from model input",
        "<b>11 dedicated tests:</b> tests/test_lookahead.py validates temporal integrity",
    ]
    for item in lookahead_items:
        content.append(Paragraph(f"• {item}", bullet_style))

    content.append(PageBreak())

    # ==========================================================================
    # 6. PRODUCTION CONFIGURATION
    # ==========================================================================
    content.append(Paragraph("6. Production Configuration", h1_style))

    content.append(Paragraph("<b>Required CLI Flags</b>", h3_style))

    flag_data = [
        ['Flag', 'Purpose', 'Why Required'],
        ['--apply-nms', 'Remove overlapping patterns', 'Prevents 71% redundant data'],
        ['--apply-physics-filter', 'Remove untradeable', '$50k liquidity floor'],
        ['--mode training', 'Preserve dormant stocks', 'Lottery tickets for model'],
        ['--use-coil-focal', 'Coil-aware loss', 'Better pattern learning'],
        ['--disable-trinity', 'Pattern-level splits', 'Prevents temporal leak'],
        ['--skip-market-cap-api', 'PIT market cap', 'EU APIs fail 99%'],
    ]

    flag_table = Table(flag_data, colWidths=[1.5*inch, 1.5*inch, 2.9*inch])
    flag_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2f855a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0fff4')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#68d391')),
    ]))
    content.append(flag_table)

    content.append(Spacer(1, 0.15*inch))
    content.append(Paragraph("<b>Key Constants</b>", h3_style))

    const_code = """
    # Labeling
    MIN_OUTCOME_WINDOW = 10      # High-vol stocks
    MAX_OUTCOME_WINDOW = 60      # Low-vol stocks
    TARGET_R_MULTIPLE = 3.0     # 3R move to qualify
    GAP_LIMIT_R = 0.5           # Max acceptable entry gap

    # Execution
    RISK_UNIT_DOLLARS = 250.0   # Fixed R per trade
    MAX_CAPITAL_PER_TRADE = 5000.0
    ADV_LIQUIDITY_PCT = 0.04    # 4% of daily volume
    MAX_DANGER_PROB = 0.25      # Danger filter threshold

    # Volume Confirmation
    VOLUME_MULTIPLIER_TARGET = 2.0  # 2× average
    VOLUME_SUSTAINED_DAYS = 3       # 3 consecutive days
    MIN_DOLLAR_VOLUME = 50000       # $50k floor
    """
    content.append(Paragraph(const_code, code_style))

    content.append(PageBreak())

    # ==========================================================================
    # 7. KEY FILES REFERENCE
    # ==========================================================================
    content.append(Paragraph("7. Key Files Reference", h1_style))

    files_data = [
        ['File', 'Purpose'],
        ['main_pipeline.py', 'Unified pipeline orchestrator'],
        ['pipeline/00_detect_patterns.py', 'Pattern detection step'],
        ['pipeline/00b_label_outcomes.py', 'Outcome labeling step'],
        ['pipeline/01_generate_sequences.py', 'Sequence generation step'],
        ['pipeline/02_train_temporal.py', 'Model training step'],
        ['pipeline/03_predict_temporal.py', 'Prediction & execution step'],
        ['models/temporal_hybrid_unified.py', 'Neural network architecture'],
        ['models/temporal_hybrid_v18.py', 'Legacy model (deprecated)'],
        ['core/pattern_scanner.py', 'Pattern detection state machine'],
        ['config/feature_registry.py', 'Feature definitions'],
        ['config/constants.py', 'System constants'],
        ['utils/data_integrity.py', 'Data validation checks'],
        ['utils/temporal_split.py', 'Train/val/test splitting'],
        ['losses/coil_focal_loss.py', 'Custom loss functions'],
        ['output/models/production_model.pt', 'Production model weights'],
    ]

    files_table = Table(files_data, colWidths=[2.8*inch, 3.1*inch])
    files_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a5568')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f7fafc')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#a0aec0')),
        ('FONTNAME', (0, 1), (0, -1), 'Courier'),
    ]))
    content.append(files_table)

    content.append(PageBreak())

    # ==========================================================================
    # 8. APPENDIX
    # ==========================================================================
    content.append(Paragraph("8. Appendix: Code Structure", h1_style))

    structure = """
    trans/
    ├── main_pipeline.py          # Entry point, orchestrates all steps
    ├── CLAUDE.md                 # System specification document
    │
    ├── pipeline/                 # Pipeline steps
    │   ├── 00_detect_patterns.py
    │   ├── 00b_label_outcomes.py
    │   ├── 01_generate_sequences.py
    │   ├── 02_train_temporal.py
    │   └── 03_predict_temporal.py
    │
    ├── models/                   # Neural network definitions
    │   ├── temporal_hybrid_unified.py  # Main model
    │   └── inference_wrapper.py        # Production inference
    │
    ├── core/                     # Core detection logic
    │   ├── pattern_scanner.py    # State machine scanner
    │   └── pattern_detector.py   # Detection utilities
    │
    ├── config/                   # Configuration
    │   ├── constants.py          # System constants
    │   ├── feature_registry.py   # Feature definitions
    │   └── context_features.py   # Context feature list
    │
    ├── utils/                    # Utilities
    │   ├── data_integrity.py     # Validation checks
    │   ├── temporal_split.py     # Data splitting
    │   ├── data_loader.py        # Data loading
    │   └── market_cap_fetcher.py # Market cap estimation
    │
    ├── losses/                   # Loss functions
    │   └── coil_focal_loss.py
    │
    ├── tests/                    # Test suite
    │   ├── test_lookahead.py     # Temporal integrity
    │   ├── test_data_integrity.py
    │   └── ...
    │
    ├── output/                   # Generated artifacts
    │   ├── models/               # Trained models
    │   ├── sequences/            # Generated sequences
    │   └── reports/              # Reports and documentation
    │
    └── docs/                     # Documentation
        ├── SYSTEM_OVERVIEW.md
        ├── ARCHITECTURE.md
        └── BUG_FIX_HISTORY.md
    """
    content.append(Paragraph(structure.replace('\n', '<br/>'), code_style))

    content.append(Spacer(1, 0.3*inch))
    content.append(Paragraph("— End of Document —", ParagraphStyle(
        'End', parent=styles['Normal'], fontSize=10, alignment=TA_CENTER,
        textColor=colors.HexColor('#a0aec0')
    )))

    # Build PDF
    doc.build(content)
    print(f"PDF generated: {output_path}")
    return str(output_path)

if __name__ == "__main__":
    create_pdf()
