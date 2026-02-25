#!/usr/bin/env python3
"""Generate comprehensive PDF documentation for TRANS model."""

from fpdf import FPDF
from datetime import datetime


class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 10)
        self.cell(0, 8, 'TRANS Model Documentation - Confidential', 0, 1, 'R')
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 14)
        self.set_fill_color(240, 240, 240)
        self.cell(0, 10, title, 0, 1, 'L', True)
        self.ln(4)

    def section_title(self, title):
        self.set_font('Helvetica', 'B', 11)
        self.cell(0, 8, title, 0, 1, 'L')
        self.ln(2)

    def body_text(self, text):
        self.set_font('Helvetica', '', 10)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def code_block(self, text):
        self.set_font('Courier', '', 8)
        self.set_fill_color(245, 245, 245)
        self.multi_cell(0, 4, text, 0, 'L', True)
        self.ln(2)

    def table_header(self, cols, widths):
        self.set_font('Helvetica', 'B', 9)
        self.set_fill_color(220, 220, 220)
        for i, col in enumerate(cols):
            self.cell(widths[i], 7, col, 1, 0, 'C', True)
        self.ln()

    def table_row(self, cols, widths):
        self.set_font('Helvetica', '', 9)
        for i, col in enumerate(cols):
            self.cell(widths[i], 6, str(col), 1, 0, 'C')
        self.ln()


def generate_pdf():
    # Create PDF
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=15)

    # =========================================================================
    # TITLE PAGE
    # =========================================================================
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 24)
    pdf.ln(60)
    pdf.cell(0, 15, 'TRANS Model Documentation', 0, 1, 'C')
    pdf.set_font('Helvetica', '', 14)
    pdf.cell(0, 10, 'Temporal Recognition and Analysis of Sleeper Patterns', 0, 1, 'C')
    pdf.ln(20)
    pdf.set_font('Helvetica', '', 12)
    pdf.cell(0, 8, 'Technical Specification for External Review', 0, 1, 'C')
    pdf.ln(30)
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 6, 'Version: 1.0', 0, 1, 'C')
    pdf.cell(0, 6, 'Date: January 17, 2026', 0, 1, 'C')
    pdf.cell(0, 6, 'Model: EU Ensemble v1 (5 models)', 0, 1, 'C')

    # =========================================================================
    # TABLE OF CONTENTS
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title('Table of Contents')
    toc = [
        '1. Executive Summary',
        '2. Business Problem',
        '3. Data Pipeline',
        '4. Feature Engineering',
        '5. Model Architecture',
        '6. Training Process',
        '7. Evaluation Metrics',
        '8. Production Usage',
        '9. Limitations and Risks',
        '10. Code Examples'
    ]
    for item in toc:
        pdf.body_text(item)

    # =========================================================================
    # 1. EXECUTIVE SUMMARY
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title('1. Executive Summary')

    pdf.section_title('1.1 What This System Does')
    pdf.body_text('TRANS (Temporal Recognition and Analysis of Sleeper Patterns) is a machine learning system that:')
    pdf.body_text('1. Detects consolidation patterns in stock price data')
    pdf.body_text('2. Classifies these patterns into three outcome categories')
    pdf.body_text('3. Predicts which patterns are likely to produce profitable trades')

    pdf.section_title('1.2 Key Results')
    pdf.table_header(['Metric', 'Value', 'Meaning'], [50, 30, 110])
    pdf.table_row(['Top-5% Target Rate', '49.1%', 'Half of best predictions are winners'], [50, 30, 110])
    pdf.table_row(['Top-15% Target Rate', '48.5%', 'Nearly half of high-conf preds win'], [50, 30, 110])
    pdf.table_row(['Base Rate (Random)', '29.0%', 'Without model, only 29% would win'], [50, 30, 110])
    pdf.table_row(['Lift', '1.67x', 'Model is 67% better than random'], [50, 30, 110])

    pdf.section_title('1.3 The Core Insight')
    pdf.body_text('Stocks often enter "consolidation" periods where price moves sideways in a tight range before making a significant move. This system identifies these patterns and predicts which ones will break out upward profitably.')

    # =========================================================================
    # 2. BUSINESS PROBLEM
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title('2. Business Problem')

    pdf.section_title('2.1 What is a Consolidation Pattern?')
    pdf.body_text('A consolidation pattern occurs when a stock:')
    pdf.body_text('- Trades in a narrow price range (the "box")')
    pdf.body_text('- Shows declining volatility (price swings get smaller)')
    pdf.body_text('- Has reduced trading volume (fewer shares traded)')
    pdf.body_text('- Lasts for multiple weeks')

    pdf.section_title('2.2 The Three Outcomes')
    pdf.table_header(['Class', 'Name', 'Definition', 'Strategic Value'], [20, 30, 80, 40])
    pdf.table_row(['0', 'Danger', 'Stop-loss hit (-2R)', '-1.0'], [20, 30, 80, 40])
    pdf.table_row(['1', 'Noise', 'Neither target nor stop', '-0.1'], [20, 30, 80, 40])
    pdf.table_row(['2', 'Target', 'Profit target hit (+5R)', '+5.0'], [20, 30, 80, 40])

    pdf.section_title('2.3 Risk-Reward Framework (R-Multiples)')
    pdf.body_text('R = Risk per trade = Distance from entry to stop-loss')
    pdf.code_block('Example:\n- Entry price: $10.00\n- Stop-loss: $9.50\n- R = $0.50\n\nTarget: +5R = $12.50 (25% gain)\nStop: -2R = $9.00 (10% loss)')

    # =========================================================================
    # 3. DATA PIPELINE
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title('3. Data Pipeline')

    pdf.section_title('3.1 Data Sources')
    pdf.table_header(['Data Type', 'Source', 'Update'], [50, 70, 50])
    pdf.table_row(['Price Data (OHLCV)', 'Yahoo Finance / Parquet', 'Daily'], [50, 70, 50])
    pdf.table_row(['Market Cap', 'Yahoo Finance API', 'Cached (7 days)'], [50, 70, 50])
    pdf.table_row(['SPY Benchmark', 'Yahoo Finance', 'Daily'], [50, 70, 50])

    pdf.section_title('3.2 Universe Selection')
    pdf.body_text('Geographic Focus: European equities (4,490 tickers)')
    pdf.body_text('Exchanges: London (.L), Frankfurt (.DE), Paris (.PA), Milan (.MI), Amsterdam (.AS), Swiss (.SW), Stockholm (.ST), and others.')
    pdf.body_text('Filtering: Must have market cap, 100+ days history, <50% zero-volume days, no extreme gaps (>100% single day).')

    pdf.section_title('3.3 Pipeline Steps')
    pdf.code_block('Step 1: Detect Patterns -> 555,229 candidates\nStep 2: Label Outcomes (wait 100 days) -> 525,473 labeled\nStep 3: Generate Sequences -> 7,611 after deduplication\nStep 4: Train Ensemble -> 5 models')

    pdf.section_title('3.4 Temporal Integrity (Preventing Look-Ahead Bias)')
    pdf.body_text('Critical: The model NEVER sees future information during training.')
    pdf.body_text('- Patterns detected without knowing outcomes')
    pdf.body_text('- Labels assigned only after 100 days pass')
    pdf.body_text('- Train/val/test split is strictly temporal (by date)')

    # =========================================================================
    # 4. FEATURE ENGINEERING
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title('4. Feature Engineering')

    pdf.section_title('4.1 Temporal Features (14 features x 20 days)')
    pdf.body_text('The model receives a 20-day "movie" of the consolidation pattern.')

    pdf.body_text('Price Features (Indices 0-3): Open, High, Low, Close')
    pdf.body_text('Formula: (Price_t / Close_0) - 1 (relativized to day 0)')

    pdf.body_text('Volume Feature (Index 4): log(Volume_t / Volume_0)')
    pdf.body_text('Interpretation: 0=same, +0.69=doubled, -0.69=halved')

    pdf.section_title('4.2 Technical Indicators (Indices 5-7)')
    pdf.table_header(['Index', 'Feature', 'Range', 'Purpose'], [20, 50, 40, 80])
    pdf.table_row(['5', 'BBW_20', '[0, 0.50]', 'Bollinger Band Width (volatility)'], [20, 50, 40, 80])
    pdf.table_row(['6', 'ADX', '[0, 100]', 'Trend strength indicator'], [20, 50, 40, 80])
    pdf.table_row(['7', 'Volume_Ratio', '[0, 20]', 'Current vs average volume'], [20, 50, 40, 80])

    pdf.section_title('4.3 Composite Scores (Indices 8-11)')
    pdf.table_header(['Index', 'Feature', 'Range', 'Purpose'], [20, 50, 40, 80])
    pdf.table_row(['8', 'Vol_Dryup', '[0, 7]', 'Volume contraction signal'], [20, 50, 40, 80])
    pdf.table_row(['9', 'VAR_Score', '[-1, 1]', 'Volume accumulation'], [20, 50, 40, 80])
    pdf.table_row(['10', 'NES_Score', '[-1, 1]', 'Energy concentration'], [20, 50, 40, 80])
    pdf.table_row(['11', 'LPF_Score', '[-1, 1]', 'Liquidity flow pressure'], [20, 50, 40, 80])

    pdf.section_title('4.4 Context Features (13 static values)')
    pdf.body_text('Market context around the pattern:')
    pdf.table_header(['Feature', 'Description'], [60, 130])
    pdf.table_row(['Float_Turnover', 'Trading activity vs market cap'], [60, 130])
    pdf.table_row(['Trend_Position', 'Price vs 200-day moving average'], [60, 130])
    pdf.table_row(['Base_Duration', 'How long consolidating'], [60, 130])
    pdf.table_row(['Relative_Volume', 'Recent vs historical volume'], [60, 130])
    pdf.table_row(['Distance_to_High', 'Gap to 52-week high'], [60, 130])
    pdf.table_row(['Price_Position', 'Where in box (0=bottom, 1=top)'], [60, 130])
    pdf.table_row(['Coil_Intensity', 'Combined coil quality score'], [60, 130])

    # =========================================================================
    # 5. MODEL ARCHITECTURE
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title('5. Model Architecture')

    pdf.section_title('5.1 High-Level Overview')
    pdf.body_text('The model has 4 parallel branches that process the input differently, then fuses their outputs for final classification.')

    pdf.section_title('5.2 Branch A: LSTM (32 dimensions)')
    pdf.body_text('Purpose: Capture sequential evolution over 20 days')
    pdf.body_text('Architecture: 2-layer LSTM with hidden_size=32')
    pdf.body_text('Learns: Trend evolution, pattern "narrative"')

    pdf.section_title('5.3 Branch B: CNN + Self-Attention (96 dimensions)')
    pdf.body_text('Purpose: Detect geometric "shape" of consolidation')
    pdf.body_text('Architecture: Two parallel 1D convolutions (kernel 3 and 5), then 4-head MultiheadAttention')
    pdf.body_text('Learns: Local patterns, which days are most important')

    pdf.section_title('5.4 Branch C: Context-Query Attention (96 dimensions)')
    pdf.body_text('Purpose: Use market context to decide what to look for')
    pdf.body_text('Architecture: Cross-attention where context queries the sequence')
    pdf.body_text('Learns: Context-dependent pattern interpretation')

    pdf.section_title('5.5 Branch D: GRN (32 dimensions)')
    pdf.body_text('Purpose: Process static context features')
    pdf.body_text('Architecture: Gated Residual Network with learnable gating')
    pdf.body_text('Learns: Which context features to emphasize')

    pdf.section_title('5.6 Fusion Network')
    pdf.body_text('Concatenates all branches: 32 + 96 + 96 + 32 = 256 dimensions')
    pdf.body_text('Classification head: Dense(256 -> 128 -> 64 -> 3)')
    pdf.body_text('Output: 3 probabilities [P(Danger), P(Noise), P(Target)]')

    pdf.section_title('5.7 Ensemble (5 Models)')
    pdf.table_header(['Model', 'Dropout', 'Val Accuracy'], [40, 50, 60])
    pdf.table_row(['0', '0.30', '52.62%'], [40, 50, 60])
    pdf.table_row(['1', '0.40', '52.92%'], [40, 50, 60])
    pdf.table_row(['2', '0.35', '52.62%'], [40, 50, 60])
    pdf.table_row(['3', '0.30', '53.23%'], [40, 50, 60])
    pdf.table_row(['4', '0.45', '52.62%'], [40, 50, 60])
    pdf.body_text('Ensemble averages predictions from all 5 models for robustness.')

    # =========================================================================
    # 6. TRAINING PROCESS
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title('6. Training Process')

    pdf.section_title('6.1 Dataset Split (Temporal)')
    pdf.body_text('Total: 7,611 sequences')
    pdf.body_text('Train: 70% = 5,327 (oldest patterns)')
    pdf.body_text('Val: 15% = 1,141 (middle period)')
    pdf.body_text('Test: 15% = 1,143 (most recent patterns)')

    pdf.section_title('6.2 Class Distribution')
    pdf.table_header(['Class', 'Name', 'Count', 'Percentage'], [30, 40, 40, 50])
    pdf.table_row(['0', 'Danger', '4,392', '57.7%'], [30, 40, 40, 50])
    pdf.table_row(['1', 'Noise', '1,225', '16.1%'], [30, 40, 40, 50])
    pdf.table_row(['2', 'Target', '1,994', '26.2%'], [30, 40, 40, 50])

    pdf.section_title('6.3 Loss Function: Coil-Aware Focal Loss')
    pdf.body_text('Focal Loss focuses on hard examples that the model gets wrong.')
    pdf.body_text('Coil-Aware modification: Patterns with high coil_intensity get boosted gradients.')
    pdf.body_text('Class Weights: Danger=5.0, Noise=1.0, Target=1.0')
    pdf.body_text('(Heavy penalty for predicting Target when actual is Danger)')

    pdf.section_title('6.4 Optimizer Settings')
    pdf.body_text('Optimizer: AdamW with lr=0.001, weight_decay=0.01')
    pdf.body_text('Batch size: 32, Max epochs: 200, Early stopping: 30 epochs patience')

    # =========================================================================
    # 7. EVALUATION METRICS
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title('7. Evaluation Metrics')

    pdf.section_title('7.1 Primary Metric: Top-K Target Rate')
    pdf.body_text('Sort patterns by Expected Value (EV), take top K%, measure Target rate.')
    pdf.body_text('EV = P(Danger)*(-1) + P(Noise)*(-0.1) + P(Target)*(+5)')

    pdf.table_header(['Percentile', 'N', 'Target Rate', 'Danger Rate', 'Avg EV'], [35, 25, 40, 40, 35])
    pdf.table_row(['Top 5%', '57', '49.1%', '22.8%', '1.97'], [35, 25, 40, 40, 35])
    pdf.table_row(['Top 10%', '114', '52.6%', '26.3%', '1.95'], [35, 25, 40, 40, 35])
    pdf.table_row(['Top 15%', '171', '48.5%', '31.0%', '1.94'], [35, 25, 40, 40, 35])
    pdf.table_row(['Top 20%', '228', '46.1%', '32.9%', '1.93'], [35, 25, 40, 40, 35])

    pdf.section_title('7.2 Ensemble vs Individual Performance')
    pdf.table_header(['Model', 'Top-15% Target Rate'], [80, 80])
    pdf.table_row(['Individual Average', '42.8%'], [80, 80])
    pdf.table_row(['Best Individual', '45.0%'], [80, 80])
    pdf.table_row(['ENSEMBLE', '48.5%'], [80, 80])
    pdf.body_text('Ensemble improves +5.7pp over average, +3.5pp over best individual.')

    pdf.section_title('7.3 Lift Analysis')
    pdf.body_text('Base rate (random): 26.2% Target')
    pdf.body_text('Top-5% model prediction: 49.1% Target')
    pdf.body_text('Lift = 49.1% / 26.2% = 1.87x better than random')

    # =========================================================================
    # 8. PRODUCTION USAGE
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title('8. Production Usage')

    pdf.section_title('8.1 Loading the Ensemble')
    pdf.code_block('import torch\ncheckpoint = torch.load("ensemble_checkpoint.pt")\nmodels = []\nfor state_dict in checkpoint["model_states"]:\n    model = HybridFeatureNetwork(...)\n    model.load_state_dict(state_dict)\n    models.append(model)')

    pdf.section_title('8.2 Making Predictions')
    pdf.code_block('# Get predictions from each model\nall_probs = [softmax(model(seq, ctx)) for model in models]\n\n# Ensemble average\nensemble_probs = mean(all_probs)\n\n# Calculate EV\nEV = probs[0]*(-1) + probs[1]*(-0.1) + probs[2]*(+5)')

    pdf.section_title('8.3 Signal Interpretation')
    pdf.table_header(['EV Range', 'Danger Prob', 'Recommendation'], [40, 50, 80])
    pdf.table_row(['> 2.0', '< 35%', 'STRONG_SIGNAL'], [40, 50, 80])
    pdf.table_row(['1.5 - 2.0', '< 40%', 'SIGNAL'], [40, 50, 80])
    pdf.table_row(['0.5 - 1.5', 'Any', 'HOLD (monitor)'], [40, 50, 80])
    pdf.table_row(['< 0.5', 'Any', 'AVOID'], [40, 50, 80])

    # =========================================================================
    # 9. LIMITATIONS AND RISKS
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title('9. Limitations and Risks')

    pdf.section_title('9.1 Known Limitations')
    pdf.body_text('1. Training Data Period: Model trained on 2020-2024 data. May not perform well in different market regimes.')
    pdf.body_text('2. Geographic Bias: Optimized for European equities. US market may behave differently.')
    pdf.body_text('3. Market Cap Focus: Best performance on small/micro caps. Large caps filtered out.')
    pdf.body_text('4. Early Stopping: Models stopped at epochs 0-3. May indicate underfitting.')

    pdf.section_title('9.2 Risks')
    pdf.body_text('1. Overfitting: Past consolidations may not predict future ones.')
    pdf.body_text('2. Liquidity: Small caps may have wide spreads, eroding profits.')
    pdf.body_text('3. Black Swan: Model assumes normal conditions. Crashes not represented.')
    pdf.body_text('4. Look-Ahead: Despite checks, subtle data leakage could exist.')

    pdf.section_title('9.3 Recommended Safeguards')
    pdf.body_text('- Never risk more than 1-2% per trade')
    pdf.body_text('- Trade multiple patterns for diversification')
    pdf.body_text('- Monitor live vs backtest performance')
    pdf.body_text('- Retrain every 90 days')

    # =========================================================================
    # 10. APPENDIX
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title('10. Appendix: Technical Details')

    pdf.section_title('10.1 File Locations')
    pdf.code_block('Ensemble Checkpoint: output/models/ensemble_checkpoint_20260117_221415.pt\nSequences: output/sequences/sequences_20260117_205136.h5\nLabeled Patterns: output/labeled_patterns.parquet')

    pdf.section_title('10.2 Model Parameters')
    pdf.body_text('Total trainable parameters: ~50,000')
    pdf.body_text('Input: Sequences (batch, 20, 14) + Context (batch, 13)')
    pdf.body_text('Output: 3 class probabilities')

    pdf.section_title('10.3 Reproducibility')
    pdf.body_text('Random seeds: 42, 43, 44, 45, 46 (one per ensemble member)')
    pdf.body_text('PyTorch version: 2.x')
    pdf.body_text('Training hardware: CPU/GPU compatible')

    pdf.section_title('10.4 Contact')
    pdf.body_text('For questions, review source code in /trans directory.')
    pdf.body_text('Run: python -m pytest tests/ for validation.')

    # Save PDF
    output_path = 'output/TRANS_Model_Documentation_20260117.pdf'
    pdf.output(output_path)
    print(f'PDF saved to: {output_path}')
    return output_path


if __name__ == '__main__':
    generate_pdf()
