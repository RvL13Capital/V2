"""
Safe PDF Generator Wrapper
Ensures PDF generation succeeds even if some plots fail
"""

import os
import sys
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from io import BytesIO
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def safe_generate_report(data, output_filename):
    """Generate PDF report with robust error handling"""
    
    print(f"\nGenerating comprehensive PDF report: {output_filename}")
    
    # Import the analyzers
    try:
        from advanced_analysis_with_gcs import (
            RobustnessSensitivityAnalyzer,
            PostBreakoutAnalyzer, 
            RegressionAnalyzer,
            ClusterAnalyzer,
            CorrelationHeatmapAnalyzer
        )
    except ImportError:
        print("Error: Could not import analysis modules")
        return False
    
    # Create document
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18,
    )
    
    styles = getSampleStyleSheet()
    story = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Title page
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph(
        "<b>AIv3 Consolidation Pattern Analysis Report</b>",
        styles['Title']
    ))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        styles['Normal']
    ))
    story.append(Paragraph(
        f"Patterns Analyzed: {len(data)}",
        styles['Normal']
    ))
    story.append(PageBreak())
    
    # 1. Robustness Analysis
    print("  Adding Robustness & Sensitivity Analysis...")
    try:
        analyzer = RobustnessSensitivityAnalyzer(data)
        story.append(Paragraph("<b>1. Robustness & Sensitivity Analysis</b>", styles['Heading1']))
        
        # Add results without plots
        boundary_results = analyzer.vary_boundary_width()
        story.append(Paragraph("<b>Boundary Width Sensitivity:</b>", styles['Normal']))
        for width, metrics in list(boundary_results.items())[:5]:
            text = f"Width ≤{width}%: Success Rate={metrics['success_rate']:.1%}, Count={metrics['count']}"
            story.append(Paragraph(text, styles['Normal']))
        
        story.append(PageBreak())
    except Exception as e:
        print(f"    Warning: Robustness analysis failed: {e}")
        story.append(Paragraph("1. Robustness Analysis: Error generating analysis", styles['Normal']))
        story.append(PageBreak())
    
    # 2. Post-Breakout Analysis
    print("  Adding Post-Breakout Phase Analysis...")
    try:
        analyzer = PostBreakoutAnalyzer(data)
        story.append(Paragraph("<b>2. Post-Breakout Phase Analysis</b>", styles['Heading1']))
        
        pullback_stats = analyzer.analyze_pullback_patterns()
        story.append(Paragraph(f"Average Pullback Depth: {pullback_stats['avg_pullback_depth']:.1%}", styles['Normal']))
        story.append(Paragraph(f"Recovery Rate: {pullback_stats['recovery_rate']:.1%}", styles['Normal']))
        
        story.append(PageBreak())
    except Exception as e:
        print(f"    Warning: Post-breakout analysis failed: {e}")
        story.append(Paragraph("2. Post-Breakout Analysis: Error generating analysis", styles['Normal']))
        story.append(PageBreak())
    
    # 3. Regression Analysis
    print("  Adding Regression Analysis...")
    try:
        analyzer = RegressionAnalyzer(data)
        story.append(Paragraph("<b>3. Regression Analysis</b>", styles['Heading1']))
        
        linear_results = analyzer.run_linear_regression()
        story.append(Paragraph(f"Linear R²: {linear_results['r_squared']:.3f}", styles['Normal']))
        
        logistic_results = analyzer.run_logistic_regression()
        story.append(Paragraph(f"Model Accuracy: {logistic_results.get('accuracy', 0):.1%}", styles['Normal']))
        
        story.append(PageBreak())
    except Exception as e:
        print(f"    Warning: Regression analysis failed: {e}")
        story.append(Paragraph("3. Regression Analysis: Error generating analysis", styles['Normal']))
        story.append(PageBreak())
    
    # 4. Cluster Analysis
    print("  Adding Cluster Analysis...")
    try:
        analyzer = ClusterAnalyzer(data)
        story.append(Paragraph("<b>4. Cluster Analysis</b>", styles['Heading1']))
        
        results = analyzer.find_optimal_clusters()
        story.append(Paragraph(f"Optimal Clusters: {results['optimal_k']}", styles['Normal']))
        story.append(Paragraph(f"Silhouette Score: {results['silhouette_score']:.3f}", styles['Normal']))
        
        story.append(PageBreak())
    except Exception as e:
        print(f"    Warning: Cluster analysis failed: {e}")
        story.append(Paragraph("4. Cluster Analysis: Error generating analysis", styles['Normal']))
        story.append(PageBreak())
    
    # 5. Correlation Analysis
    print("  Adding Correlation Heatmaps...")
    try:
        analyzer = CorrelationHeatmapAnalyzer(data)
        story.append(Paragraph("<b>5. Correlation Analysis</b>", styles['Heading1']))
        
        corr_matrix = analyzer.calculate_correlations()
        story.append(Paragraph("Correlation matrix calculated successfully", styles['Normal']))
        story.append(Paragraph(f"Features analyzed: {len(corr_matrix.columns)}", styles['Normal']))
        
    except Exception as e:
        print(f"    Warning: Correlation analysis failed: {e}")
        story.append(Paragraph("5. Correlation Analysis: Error generating analysis", styles['Normal']))
    
    # Build PDF
    try:
        doc.build(story)
        print(f"\nPDF report successfully generated: {output_filename}")
        
        # Clean up any temp files
        import glob
        for temp_file in glob.glob(f"temp_*_{timestamp}.png"):
            try:
                os.remove(temp_file)
            except:
                pass
        
        return True
    except Exception as e:
        print(f"\nError building PDF: {e}")
        return False


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
        
        # Generate report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"analyse_report_{timestamp}.pdf"
        
        success = safe_generate_report(data, report_name)
        
        if success:
            print("\n" + "="*60)
            print("ANALYSE ABGESCHLOSSEN!")
            print("="*60)
            
            # Summary statistics
            total = len(data)
            if 'outcome_class' in data.columns:
                success_patterns = len(data[data['outcome_class'].isin(['K2','K3','K4'])])
                k4 = len(data[data['outcome_class'] == 'K4'])
                
                print("\nErgebnisse:")
                print(f"  Analysierte Muster: {total}")
                success_rate = success_patterns/total*100 if total > 0 else 0
                k4_rate = k4/total*100 if total > 0 else 0
                print(f"  Erfolgsrate (K2-K4): {success_rate:.1f}%")
                print(f"  Exceptional Rate (K4): {k4_rate:.1f}%")
            
            print(f"\nPDF-Report: {report_name}")
            
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