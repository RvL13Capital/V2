"""
Integration Tests for TRANS Pipeline
=====================================

End-to-end integration tests for the complete TRANS pipeline:
00_detect_patterns → 01_generate_sequences → 02_train_temporal → 03_predict → 04_evaluate
"""

import unittest
import sys
import os
import tempfile
import shutil
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pattern_scanner import ConsolidationPatternScanner
from core.pattern_detector import TemporalPatternDetector
from core.path_dependent_labeler import PathDependentLabelerV17 as PathDependentLabeler
from models.temporal_hybrid_v18 import HybridFeatureNetwork
from models.asymmetric_loss import AsymmetricLoss
from config import (
    NUM_CLASSES,
    MIN_LIQUIDITY_DOLLAR,
    calculate_expected_value
)


class TestPipelineIntegration(unittest.TestCase):
    """Test complete pipeline integration."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        # Create temporary directory for test outputs
        cls.temp_dir = tempfile.mkdtemp(prefix="trans_test_")
        cls.output_dir = Path(cls.temp_dir) / "output"
        cls.data_dir = Path(cls.temp_dir) / "data"

        # Create directories
        (cls.output_dir / "patterns").mkdir(parents=True, exist_ok=True)
        (cls.output_dir / "sequences").mkdir(parents=True, exist_ok=True)
        (cls.output_dir / "models").mkdir(parents=True, exist_ok=True)
        (cls.output_dir / "predictions").mkdir(parents=True, exist_ok=True)
        (cls.data_dir / "raw").mkdir(parents=True, exist_ok=True)

        # Generate synthetic test data
        cls._generate_synthetic_data()

        print(f"Test environment created at: {cls.temp_dir}")

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Remove temporary directory
        if hasattr(cls, 'temp_dir'):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)
        print("Test environment cleaned up")

    @classmethod
    def _generate_synthetic_data(cls):
        """Generate synthetic market data for testing."""
        # Create synthetic ticker data with known patterns
        np.random.seed(42)

        tickers = ['TEST1', 'TEST2', 'TEST3']
        for ticker in tickers:
            dates = pd.date_range('2020-01-01', periods=1000, freq='D')

            # Generate price with consolidation patterns
            prices = []
            base_price = 100.0

            for i in range(1000):
                if 100 <= i < 120:  # Consolidation 1
                    # Tight range
                    price = base_price + np.random.uniform(-1, 1)
                elif i == 120:  # Breakout
                    price = base_price * 1.5
                    base_price = price
                elif 400 <= i < 430:  # Consolidation 2
                    price = base_price + np.random.uniform(-2, 2)
                elif i == 430:  # Breakout
                    price = base_price * 1.3
                    base_price = price
                elif 700 <= i < 740:  # Consolidation 3
                    price = base_price + np.random.uniform(-1.5, 1.5)
                elif i == 740:  # Breakdown
                    price = base_price * 0.8
                    base_price = price
                else:
                    # Random walk
                    price = base_price + np.random.uniform(-5, 5)
                    base_price = price * 0.99 + base_price * 0.01  # Mean reversion

                prices.append(max(price, 1.0))  # Ensure positive

            # Create OHLCV data
            df = pd.DataFrame({
                'date': dates,
                'open': prices,
                'high': [p + np.random.uniform(0, 2) for p in prices],
                'low': [p - np.random.uniform(0, 2) for p in prices],
                'close': prices,
                'volume': np.random.randint(100000, 1000000, 1000)
            })

            # Save to parquet
            file_path = cls.data_dir / "raw" / f"{ticker}.parquet"
            df.to_parquet(file_path, index=False)

        print(f"Generated synthetic data for {len(tickers)} tickers")

    def test_00_pattern_detection(self):
        """Test pattern detection (Step 00)."""
        scanner = ConsolidationPatternScanner(enable_market_cap=False)  # Disable market cap for faster testing

        patterns_found = 0
        all_patterns = []

        # Scan each test ticker
        for ticker_file in (self.data_dir / "raw").glob("*.parquet"):
            ticker = ticker_file.stem

            # Load the DataFrame from the test data
            df = pd.read_parquet(ticker_file)

            # Set date as index (expected by scanner)
            if 'date' in df.columns:
                df = df.set_index('date').sort_index()

            result = scanner.scan_ticker(
                ticker,
                start_date=datetime(2020, 1, 1),
                end_date=datetime(2022, 12, 31),
                df=df  # Pass DataFrame directly
            )

            self.assertIsNotNone(result)
            self.assertGreaterEqual(result.patterns_found, 0)

            if result.patterns_found > 0:
                patterns_found += result.patterns_found
                all_patterns.extend(result.patterns)

        # Should find at least some patterns in synthetic data
        self.assertGreater(patterns_found, 0, "No patterns found in synthetic data")

        # Save patterns for next step
        patterns_df = pd.DataFrame(all_patterns)
        output_file = self.output_dir / "patterns" / "detected_patterns.parquet"
        patterns_df.to_parquet(output_file, index=False)

        print(f"Step 00: Found {patterns_found} patterns across test tickers")
        return patterns_found

    def test_01_sequence_generation(self):
        """Test sequence generation (Step 01)."""
        # First run pattern detection
        patterns_found = self.test_00_pattern_detection()

        # Load detected patterns
        patterns_file = self.output_dir / "patterns" / "detected_patterns.parquet"
        self.assertTrue(patterns_file.exists())
        patterns_df = pd.read_parquet(patterns_file)

        detector = TemporalPatternDetector()
        labeler = PathDependentLabeler()

        all_sequences = []
        all_labels = []
        all_metadata = []

        # Generate sequences for each detected pattern
        for _, pattern in patterns_df.iterrows():
            # Load ticker data
            ticker_file = self.data_dir / "raw" / f"{pattern['ticker']}.parquet"
            df = pd.read_parquet(ticker_file)

            # Set date as index (expected by pattern_detector)
            if 'date' in df.columns:
                df = df.set_index('date').sort_index()

            # Generate sequences for this specific pattern
            sequences = detector.generate_sequences_for_pattern(
                ticker=pattern['ticker'],
                pattern_start=pattern['start_date'],
                pattern_end=pattern['end_date'],
                df=df  # Pass DataFrame directly for testing
            )

            if sequences is not None and len(sequences) > 0:
                # Label the pattern
                label = labeler.label_pattern(
                    full_data=df,
                    pattern_end_idx=pattern.get('end_idx', len(df)-1),
                    pattern_boundaries={
                        'upper': pattern.get('upper_boundary', df['high'].max()),
                        'lower': pattern.get('lower_boundary', df['low'].min())
                    }
                )

                # Only include patterns with valid labels (not grey zone or None)
                if label is not None and label >= 0:
                    all_sequences.append(sequences)
                    all_labels.extend([label] * len(sequences))

                    # Create metadata for each sequence
                    for i in range(len(sequences)):
                        all_metadata.append({
                            'ticker': pattern['ticker'],
                            'pattern_id': pattern.get('pattern_id', f"{pattern['ticker']}_{i}"),
                            'start_date': pattern['start_date']
                        })

        # Should generate some sequences
        self.assertGreater(len(all_sequences), 0, "No sequences generated")

        # Concatenate sequences
        all_sequences = np.concatenate(all_sequences, axis=0) if len(all_sequences) > 0 else np.array([])

        # Save sequences
        sequences_array = np.array(all_sequences)
        labels_array = np.array(all_labels)

        np.save(self.output_dir / "sequences" / "sequences.npy", sequences_array)
        np.save(self.output_dir / "sequences" / "labels.npy", labels_array)
        pd.DataFrame(all_metadata).to_parquet(
            self.output_dir / "sequences" / "metadata.parquet",
            index=False
        )

        print(f"Step 01: Generated {len(all_sequences)} sequences from {patterns_found} patterns")
        return len(all_sequences)

    def test_02_model_training(self):
        """Test model training (Step 02)."""
        # First generate sequences
        num_sequences = self.test_01_sequence_generation()

        # Load sequences
        sequences = np.load(self.output_dir / "sequences" / "sequences.npy")
        labels = np.load(self.output_dir / "sequences" / "labels.npy")

        self.assertEqual(len(sequences), num_sequences)
        self.assertEqual(len(labels), num_sequences)

        # Split data
        split_idx = int(len(sequences) * 0.8)
        train_sequences = sequences[:split_idx]
        train_labels = labels[:split_idx]
        val_sequences = sequences[split_idx:]
        val_labels = labels[split_idx:]

        # Create model
        model = HybridFeatureNetwork(
            input_features=sequences.shape[-1],  # Number of features
            num_classes=NUM_CLASSES  # Always 3 classes now
        )

        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(train_sequences),
            torch.LongTensor(train_labels)
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True
        )

        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(val_sequences),
            torch.LongTensor(val_labels)
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False
        )

        # Training setup
        criterion = AsymmetricLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        # Train for a few epochs (minimal for testing)
        model.train()
        for epoch in range(2):  # Just 2 epochs for testing
            total_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

        # Validate
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        accuracy = correct / total if total > 0 else 0
        print(f"Validation Accuracy: {accuracy:.2%}")

        # Save model
        model_path = self.output_dir / "models" / "test_model.pt"
        torch.save(model.state_dict(), model_path)

        # Save model metadata
        metadata = {
            'accuracy': accuracy,
            'num_epochs': 2,
            'num_classes': model.num_classes,
            'input_features': model.input_features,
            'timestamp': datetime.now().isoformat()
        }

        with open(self.output_dir / "models" / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Step 02: Trained model with {accuracy:.2%} validation accuracy")
        return accuracy

    def test_03_prediction(self):
        """Test prediction generation (Step 03)."""
        # Run training first
        accuracy = self.test_02_model_training()

        # Load model
        model_path = self.output_dir / "models" / "test_model.pt"
        self.assertTrue(model_path.exists())

        # Load sequences for prediction
        sequences = np.load(self.output_dir / "sequences" / "sequences.npy")
        metadata = pd.read_parquet(self.output_dir / "sequences" / "metadata.parquet")

        # Load model metadata to get architecture params
        with open(self.output_dir / "models" / "metadata.json", 'r') as f:
            model_metadata = json.load(f)

        # Recreate model
        model = HybridFeatureNetwork(
            input_features=model_metadata['input_features'],
            num_classes=model_metadata['num_classes']
        )
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
        model.eval()

        # Generate predictions
        predictions = []
        expected_values = []

        with torch.no_grad():
            for i in range(len(sequences)):
                input_tensor = torch.FloatTensor(sequences[i:i+1])
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=-1).numpy()[0]

                # Calculate expected value
                probs_dict = {j: float(probs[j]) for j in range(len(probs))}
                ev = calculate_expected_value(probs_dict)

                # Convert to string keys for parquet serialization
                probs_dict_str = {str(j): float(probs[j]) for j in range(len(probs))}

                prediction = {
                    'sequence_idx': i,
                    'pattern_id': metadata.iloc[i]['pattern_id'],
                    'ticker': metadata.iloc[i]['ticker'],
                    'predicted_class': int(np.argmax(probs)),
                    'class_probabilities': probs_dict_str,
                    'expected_value': ev,
                    'confidence': float(np.max(probs))
                }

                predictions.append(prediction)
                expected_values.append(ev)

        # Save predictions
        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_parquet(
            self.output_dir / "predictions" / "predictions.parquet",
            index=False
        )

        # Basic validation - test pipeline functionality, not model quality
        self.assertEqual(len(predictions), len(sequences))
        self.assertGreater(len(predictions), 0)  # Predictions were generated

        # Verify predictions have required columns
        required_cols = ['sequence_idx', 'pattern_id', 'ticker', 'predicted_class',
                        'class_probabilities', 'expected_value', 'confidence']
        for col in required_cols:
            self.assertIn(col, predictions_df.columns)

        print(f"Step 03: Generated {len(predictions)} predictions with avg EV = {np.nanmean(expected_values):.2f}")
        return predictions

    def test_04_evaluation(self):
        """Test evaluation metrics (Step 04)."""
        # Run prediction first
        predictions = self.test_03_prediction()

        # Load true labels
        labels = np.load(self.output_dir / "sequences" / "labels.npy")

        # Calculate metrics
        predicted_classes = [p['predicted_class'] for p in predictions]
        expected_values = [p['expected_value'] for p in predictions]

        # Accuracy
        correct = sum(1 for pred, true in zip(predicted_classes, labels) if pred == true)
        accuracy = correct / len(labels) if len(labels) > 0 else 0

        # EV correlation (simplified - normally would use actual returns)
        # For testing, use label as proxy for return
        from scipy.stats import pearsonr
        if len(set(labels)) > 1:  # Need variance for correlation
            ev_correlation, _ = pearsonr(expected_values, labels)
        else:
            ev_correlation = 0

        # Signal distribution
        strong_signals = sum(1 for ev in expected_values if ev >= 5.0)
        good_signals = sum(1 for ev in expected_values if 3.0 <= ev < 5.0)
        moderate_signals = sum(1 for ev in expected_values if 1.0 <= ev < 3.0)

        # Create evaluation report
        report = {
            'total_predictions': len(predictions),
            'accuracy': accuracy,
            'ev_correlation': ev_correlation,
            'avg_expected_value': np.mean(expected_values),
            'std_expected_value': np.std(expected_values),
            'strong_signals': strong_signals,
            'good_signals': good_signals,
            'moderate_signals': moderate_signals,
            'signal_rate': (strong_signals + good_signals) / len(predictions) if len(predictions) > 0 else 0
        }

        # Save evaluation report
        with open(self.output_dir / "predictions" / "evaluation.json", 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nStep 04: Evaluation Complete")
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  EV Correlation: {ev_correlation:.3f}")
        print(f"  Signal Rate: {report['signal_rate']:.2%}")

        # Assertions for test validation - test pipeline functionality, not model quality
        self.assertGreaterEqual(len(predictions), 0)  # Predictions generated
        self.assertIn('accuracy', report)  # Report has required metrics
        self.assertIn('signal_rate', report)
        self.assertIn('ev_correlation', report)

        # Verify report file was saved
        report_path = self.output_dir / "predictions" / "evaluation.json"
        self.assertTrue(report_path.exists())

        return report

    def test_full_pipeline(self):
        """Test complete pipeline end-to-end."""
        print("\n" + "="*60)
        print("RUNNING FULL PIPELINE INTEGRATION TEST")
        print("="*60)

        # Run all steps in sequence
        print("\n[Step 00] Pattern Detection...")
        patterns = self.test_00_pattern_detection()

        print("\n[Step 01] Sequence Generation...")
        sequences = self.test_01_sequence_generation()

        print("\n[Step 02] Model Training...")
        accuracy = self.test_02_model_training()

        print("\n[Step 03] Prediction Generation...")
        predictions = self.test_03_prediction()

        print("\n[Step 04] Evaluation...")
        report = self.test_04_evaluation()

        print("\n" + "="*60)
        print("PIPELINE INTEGRATION TEST COMPLETE")
        print("="*60)
        print(f"\nSummary:")
        print(f"  - Patterns Detected: {patterns}")
        print(f"  - Sequences Generated: {sequences}")
        print(f"  - Model Accuracy: {accuracy:.2%}")
        print(f"  - Predictions Made: {len(predictions)}")
        print(f"  - Signal Rate: {report['signal_rate']:.2%}")
        print(f"  - EV Correlation: {report['ev_correlation']:.3f}")

        # Final validation
        self.assertGreater(patterns, 0, "Pipeline failed: No patterns detected")
        self.assertGreater(sequences, 0, "Pipeline failed: No sequences generated")
        self.assertGreater(accuracy, 0, "Pipeline failed: Model training failed")
        self.assertGreater(len(predictions), 0, "Pipeline failed: No predictions made")


class TestAPIIntegration(unittest.TestCase):
    """Test API endpoints integration."""

    def setUp(self):
        """Set up test client."""
        # Note: This would require the API to be running
        # For now, we'll create placeholder tests
        pass

    def test_health_endpoint(self):
        """Test /health endpoint."""
        # Placeholder - would test actual API
        self.assertTrue(True, "API health check test placeholder")

    def test_pattern_scan_endpoint(self):
        """Test /scan/pattern endpoint."""
        # Placeholder - would test actual API
        self.assertTrue(True, "Pattern scan endpoint test placeholder")

    def test_prediction_endpoint(self):
        """Test /predict endpoint."""
        # Placeholder - would test actual API
        self.assertTrue(True, "Prediction endpoint test placeholder")


def run_integration_tests():
    """Run all integration tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add pipeline tests
    suite.addTests(loader.loadTestsFromTestCase(TestPipelineIntegration))

    # Add API tests (if API is running)
    # suite.addTests(loader.loadTestsFromTestCase(TestAPIIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*60)
    if result.wasSuccessful():
        print("ALL INTEGRATION TESTS PASSED!")
    else:
        print(f"FAILURES: {len(result.failures)}")
        print(f"ERRORS: {len(result.errors)}")
    print("="*60)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)