"""
Quick Test: Verify Attention System Installation
Run this to check if all dependencies are installed correctly
"""

import sys
import os

# Set UTF-8 encoding for Windows console
if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Use ASCII-friendly symbols
CHECK = '[OK]'
CROSS = '[X]'
WARNING = '[!]'


def check_imports():
    """Check if all required packages can be imported"""

    print("="*80)
    print("ATTENTION SYSTEM - DEPENDENCY CHECK")
    print("="*80)
    print()

    results = {}

    # Check TensorFlow
    print("Checking TensorFlow...")
    try:
        import tensorflow as tf
        print(f"  {CHECK} TensorFlow {tf.__version__} installed")
        results['tensorflow'] = True
    except ImportError as e:
        print(f"  {CROSS}TensorFlow not found: {e}")
        print("    Install with: pip install tensorflow")
        results['tensorflow'] = False

    # Check Keras
    print("\nChecking Keras...")
    try:
        from tensorflow import keras
        print(f"  {CHECK} Keras available (bundled with TensorFlow)")
        results['keras'] = True
    except ImportError as e:
        print(f"  {CROSS}Keras not found: {e}")
        results['keras'] = False

    # Check NumPy
    print("\nChecking NumPy...")
    try:
        import numpy as np
        print(f"  {CHECK} NumPy {np.__version__} installed")
        results['numpy'] = True
    except ImportError as e:
        print(f"  {CROSS}NumPy not found: {e}")
        print("    Install with: pip install numpy")
        results['numpy'] = False

    # Check Pandas
    print("\nChecking Pandas...")
    try:
        import pandas as pd
        print(f"  {CHECK} Pandas {pd.__version__} installed")
        results['pandas'] = True
    except ImportError as e:
        print(f"  {CROSS}Pandas not found: {e}")
        print("    Install with: pip install pandas")
        results['pandas'] = False

    # Check Matplotlib
    print("\nChecking Matplotlib...")
    try:
        import matplotlib
        print(f"  {CHECK} Matplotlib {matplotlib.__version__} installed")
        results['matplotlib'] = True
    except ImportError as e:
        print(f"  {CROSS}Matplotlib not found: {e}")
        print("    Install with: pip install matplotlib")
        results['matplotlib'] = False

    # Check Seaborn
    print("\nChecking Seaborn...")
    try:
        import seaborn as sns
        print(f"  {CHECK} Seaborn {sns.__version__} installed")
        results['seaborn'] = True
    except ImportError as e:
        print(f"  {CROSS}Seaborn not found: {e}")
        print("    Install with: pip install seaborn")
        results['seaborn'] = False

    # Check Plotly
    print("\nChecking Plotly...")
    try:
        import plotly
        print(f"  {CHECK} Plotly {plotly.__version__} installed")
        results['plotly'] = True
    except ImportError as e:
        print(f"  {CROSS}Plotly not found (optional): {e}")
        print("    Install with: pip install plotly")
        results['plotly'] = False

    # Check Scikit-learn
    print("\nChecking Scikit-learn...")
    try:
        import sklearn
        print(f"  {CHECK} Scikit-learn {sklearn.__version__} installed")
        results['sklearn'] = True
    except ImportError as e:
        print(f"  {CROSS}Scikit-learn not found: {e}")
        print("    Install with: pip install scikit-learn")
        results['sklearn'] = False

    print("\n" + "="*80)

    # Summary
    required = ['tensorflow', 'keras', 'numpy', 'pandas', 'matplotlib', 'sklearn']
    optional = ['seaborn', 'plotly']

    required_ok = all(results.get(pkg, False) for pkg in required)
    optional_ok = all(results.get(pkg, False) for pkg in optional)

    print("SUMMARY")
    print("="*80)
    print(f"Required packages: {CHECK + ' ALL OK' if required_ok else CROSS + ' MISSING'}")
    print(f"Optional packages: {CHECK + ' ALL OK' if optional_ok else WARNING + ' SOME MISSING'}")
    print()

    if required_ok:
        print(f"{CHECK} All required packages installed!")
        print(f"{CHECK} You can run the attention system")
    else:
        print(f"{CROSS} Some required packages are missing")
        print(f"{CROSS} Install missing packages before running the system")
        print("\nQuick fix:")
        print("  pip install -r requirements_attention.txt")

    return required_ok


def test_attention_model():
    """Quick test of attention model creation"""

    print("\n" + "="*80)
    print("TESTING ATTENTION MODEL")
    print("="*80)
    print()

    try:
        from attention_lstm_model import LSTMAttentionModel, AttentionLayer

        print("Creating test model...")
        model = LSTMAttentionModel(
            window_size=10,  # Small for testing
            n_features=5,
            lstm_units=16
        )

        print("Building model...")
        model.build_model()

        print("\nModel Summary:")
        print(model.get_model_summary())

        # Test with random data
        import numpy as np

        X_test = np.random.randn(10, 10, 5)  # 10 samples, 10 days, 5 features
        y_test = np.random.randint(0, 2, 10)

        print("\nTesting prediction...")
        predictions = model.predict(X_test)
        print(f"  {CHECK} Predictions shape: {predictions.shape}")

        print("\nTesting attention extraction...")
        attention = model.get_attention_weights(X_test)
        print(f"  {CHECK} Attention shape: {attention.shape}")
        print(f"  {CHECK} Attention sum per sample: {attention.sum(axis=1)[:3]} (should be ~1.0)")

        print("\n" + "="*80)
        print(f"{CHECK} ATTENTION MODEL TEST PASSED!")
        print("="*80)

        return True

    except Exception as e:
        print(f"\n{CROSS} Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualizer():
    """Quick test of attention visualizer"""

    print("\n" + "="*80)
    print("TESTING ATTENTION VISUALIZER")
    print("="*80)
    print()

    try:
        from attention_visualizer import AttentionVisualizer
        import numpy as np

        print("Creating visualizer...")
        viz = AttentionVisualizer()

        # Create test attention weights
        attention = np.random.dirichlet(np.ones(30))  # 30 days, sums to 1

        print("Testing attention plot...")
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend

        fig = viz.plot_single_attention(
            attention,
            dates=[f"2024-03-{i+1:02d}" for i in range(30)],
            prediction=0.75,
            show=False
        )
        print(f"  {CHECK} Plot created successfully")

        print("\n" + "="*80)
        print(f"{CHECK} VISUALIZER TEST PASSED!")
        print("="*80)

        return True

    except Exception as e:
        print(f"\n{CROSS} Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""

    print("\n" + "#"*80)
    print("# ATTENTION SYSTEM - INSTALLATION TEST")
    print("#"*80)
    print()

    # Test 1: Dependencies
    deps_ok = check_imports()

    if not deps_ok:
        print("\n" + "="*80)
        print(f"{CROSS} DEPENDENCY CHECK FAILED")
        print(f"{CROSS} Please install missing packages before proceeding")
        print("="*80)
        sys.exit(1)

    # Test 2: Attention Model
    model_ok = test_attention_model()

    # Test 3: Visualizer
    viz_ok = test_visualizer()

    # Final summary
    print("\n" + "#"*80)
    print("# FINAL RESULTS")
    print("#"*80)
    print()
    print(f"Dependencies:       {CHECK + ' PASS' if deps_ok else CROSS + ' FAIL'}")
    print(f"Attention Model:    {CHECK + ' PASS' if model_ok else CROSS + ' FAIL'}")
    print(f"Visualizer:         {CHECK + ' PASS' if viz_ok else CROSS + ' FAIL'}")
    print()

    if all([deps_ok, model_ok, viz_ok]):
        print("="*80)
        print(f"{CHECK} ALL TESTS PASSED!")
        print(f"{CHECK} Attention system is ready to use")
        print("="*80)
        print("\nNext steps:")
        print("  1. Run example: python run_attention_analysis.py")
        print("  2. Read documentation: ATTENTION_README.md")
        print("  3. Explore attention visualizations in output/")
        return 0
    else:
        print("="*80)
        print(f"{CROSS} SOME TESTS FAILED")
        print(f"{CROSS} Please fix issues before using the system")
        print("="*80)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
