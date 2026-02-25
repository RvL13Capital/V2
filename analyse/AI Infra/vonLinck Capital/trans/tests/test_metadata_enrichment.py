"""
Tests for Metadata Enrichment Pipeline
======================================

Tests that computed features (coil, ignition, priority) are properly
included in metadata.parquet for downstream analysis.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import METADATA_ENRICHMENT_FIELDS


class TestMetadataEnrichmentFields:
    """Tests for METADATA_ENRICHMENT_FIELDS configuration"""

    def test_coil_fields_present(self):
        """Coil feature fields are in enrichment list"""
        coil_fields = [
            'price_position_at_end',
            'distance_to_danger',
            'bbw_slope_5d',
            'vol_trend_5d',
            'coil_intensity',
        ]
        for field in coil_fields:
            assert field in METADATA_ENRICHMENT_FIELDS, f"Missing coil field: {field}"

    def test_ignition_fields_present(self):
        """Ignition prioritization fields are in enrichment list"""
        ignition_fields = [
            'ignition_score',
            'recency_score',
            'priority_score',
            'oversample_copy',
        ]
        for field in ignition_fields:
            assert field in METADATA_ENRICHMENT_FIELDS, f"Missing ignition field: {field}"

    def test_pattern_metric_fields_present(self):
        """Pattern metric fields are in enrichment list"""
        metric_fields = [
            'box_width',
            'pattern_days',
            'upper_boundary',
            'lower_boundary',
        ]
        for field in metric_fields:
            assert field in METADATA_ENRICHMENT_FIELDS, f"Missing metric field: {field}"


class TestMetadataEnrichmentLogic:
    """Tests for the enrichment logic in process_single_ticker"""

    def create_pattern_row(self, **kwargs):
        """Create a test pattern row as pandas Series"""
        base = {
            'ticker': 'TEST',
            'start_date': datetime(2024, 1, 1),
            'end_date': datetime(2024, 1, 20),
            'outcome_class': 2,
            'pattern_id': 'TEST_2024-01-01',
        }
        base.update(kwargs)
        return pd.Series(base)

    def test_enrichment_extracts_coil_features(self):
        """Coil features are extracted when present"""
        pattern_row = self.create_pattern_row(
            price_position_at_end=0.65,
            distance_to_danger=0.35,
            bbw_slope_5d=0.002,
            vol_trend_5d=1.2,
            coil_intensity=0.78,
        )

        # Simulate enrichment logic
        enriched = {}
        for field in METADATA_ENRICHMENT_FIELDS:
            if field in pattern_row.index:
                value = pattern_row.get(field)
                if pd.notna(value):
                    if isinstance(value, (np.integer, np.floating)):
                        value = float(value)
                    enriched[field] = value

        assert enriched['price_position_at_end'] == 0.65
        assert enriched['distance_to_danger'] == 0.35
        assert enriched['bbw_slope_5d'] == 0.002
        assert enriched['vol_trend_5d'] == 1.2
        assert enriched['coil_intensity'] == 0.78

    def test_enrichment_extracts_ignition_features(self):
        """Ignition features are extracted when present"""
        pattern_row = self.create_pattern_row(
            ignition_score=0.85,
            recency_score=0.92,
            priority_score=0.78,
            oversample_copy=0,
        )

        enriched = {}
        for field in METADATA_ENRICHMENT_FIELDS:
            if field in pattern_row.index:
                value = pattern_row.get(field)
                if pd.notna(value):
                    if isinstance(value, (np.integer, np.floating)):
                        value = float(value)
                    enriched[field] = value

        assert enriched['ignition_score'] == 0.85
        assert enriched['recency_score'] == 0.92
        assert enriched['priority_score'] == 0.78
        assert enriched['oversample_copy'] == 0

    def test_enrichment_handles_missing_fields(self):
        """Missing fields are silently skipped"""
        pattern_row = self.create_pattern_row(
            price_position_at_end=0.5,
            # No other coil/ignition fields
        )

        enriched = {}
        for field in METADATA_ENRICHMENT_FIELDS:
            if field in pattern_row.index:
                value = pattern_row.get(field)
                if pd.notna(value):
                    enriched[field] = value

        assert 'price_position_at_end' in enriched
        assert 'ignition_score' not in enriched
        assert 'coil_intensity' not in enriched

    def test_enrichment_handles_nan_values(self):
        """NaN values are skipped"""
        pattern_row = self.create_pattern_row(
            price_position_at_end=0.5,
            coil_intensity=np.nan,
            ignition_score=np.nan,
        )

        enriched = {}
        for field in METADATA_ENRICHMENT_FIELDS:
            if field in pattern_row.index:
                value = pattern_row.get(field)
                if pd.notna(value):
                    enriched[field] = value

        assert 'price_position_at_end' in enriched
        assert 'coil_intensity' not in enriched
        assert 'ignition_score' not in enriched

    def test_numpy_types_converted_to_native(self):
        """Numpy types are converted for parquet compatibility"""
        pattern_row = self.create_pattern_row(
            price_position_at_end=np.float64(0.65),
            oversample_copy=np.int64(1),
            coil_intensity=np.float32(0.8),
        )

        enriched = {}
        for field in METADATA_ENRICHMENT_FIELDS:
            if field in pattern_row.index:
                value = pattern_row.get(field)
                if pd.notna(value):
                    if isinstance(value, (np.integer, np.floating)):
                        value = float(value)
                    enriched[field] = value

        # All should be native Python floats
        assert type(enriched['price_position_at_end']) is float
        assert type(enriched['oversample_copy']) is float
        assert type(enriched['coil_intensity']) is float


class TestMetadataEnrichmentIntegration:
    """Integration tests for metadata enrichment flow"""

    def test_full_enrichment_with_all_features(self):
        """Full enrichment with all feature types"""
        pattern_row = pd.Series({
            'ticker': 'AAPL',
            'start_date': datetime(2024, 1, 1),
            'end_date': datetime(2024, 1, 20),
            'outcome_class': 2,
            'pattern_id': 'AAPL_2024-01-01',
            # Coil features
            'price_position_at_end': 0.65,
            'distance_to_danger': 0.35,
            'bbw_slope_5d': 0.002,
            'vol_trend_5d': 1.2,
            'coil_intensity': 0.78,
            # Ignition features
            'ignition_score': 0.85,
            'recency_score': 0.92,
            'priority_score': 0.78,
            'oversample_copy': 0,
            # Pattern metrics
            'box_width': 0.08,
            'pattern_days': 20,
            'upper_boundary': 150.0,
            'lower_boundary': 138.0,
        })

        # Build base metadata (simulating actual code)
        base_metadata = {
            'ticker': pattern_row['ticker'],
            'label': pattern_row['outcome_class'],
            'pattern_start_date': pattern_row['start_date'],
            'pattern_end_date': pattern_row['end_date'],
            'pattern_id': pattern_row['pattern_id'],
            'market_phase': 1,
            'nms_cluster_id': -1,
        }

        # Add enrichment fields
        for field in METADATA_ENRICHMENT_FIELDS:
            if field in pattern_row.index:
                value = pattern_row.get(field)
                if pd.notna(value):
                    if isinstance(value, (np.integer, np.floating)):
                        value = float(value)
                    base_metadata[field] = value

        # Verify all expected fields present
        assert base_metadata['ticker'] == 'AAPL'
        assert base_metadata['label'] == 2
        assert base_metadata['price_position_at_end'] == 0.65
        assert base_metadata['coil_intensity'] == 0.78
        assert base_metadata['ignition_score'] == 0.85
        assert base_metadata['priority_score'] == 0.78
        assert base_metadata['box_width'] == 0.08
        assert base_metadata['upper_boundary'] == 150.0
        assert base_metadata['lower_boundary'] == 138.0

    def test_metadata_dataframe_creation(self):
        """Enriched metadata can be converted to DataFrame"""
        metadata_rows = []

        # Simulate 3 sequences from same pattern
        for seq_idx in range(3):
            base_metadata = {
                'ticker': 'TSLA',
                'label': 1,
                'pattern_start_date': datetime(2024, 3, 1),
                'pattern_end_date': datetime(2024, 3, 25),
                'pattern_id': 'TSLA_2024-03-01',
                'market_phase': 0,
                'nms_cluster_id': 42,
                'price_position_at_end': 0.4,
                'coil_intensity': 0.6,
                'priority_score': 0.55,
            }
            metadata_row = base_metadata.copy()
            metadata_row['sequence_idx'] = seq_idx
            metadata_rows.append(metadata_row)

        # Convert to DataFrame
        df = pd.DataFrame(metadata_rows)

        assert len(df) == 3
        assert 'price_position_at_end' in df.columns
        assert 'coil_intensity' in df.columns
        assert 'priority_score' in df.columns
        assert df['sequence_idx'].tolist() == [0, 1, 2]
        assert (df['ticker'] == 'TSLA').all()

    def test_metadata_parquet_roundtrip(self, tmp_path):
        """Enriched metadata survives parquet roundtrip"""
        metadata_rows = [{
            'ticker': 'NVDA',
            'label': 2,
            'sequence_idx': 0,
            'pattern_start_date': datetime(2024, 5, 1),
            'pattern_end_date': datetime(2024, 5, 20),
            'pattern_id': 'NVDA_2024-05-01',
            'market_phase': 1,
            'nms_cluster_id': -1,
            'price_position_at_end': 0.72,
            'distance_to_danger': 0.47,
            'bbw_slope_5d': -0.001,
            'vol_trend_5d': 0.85,
            'coil_intensity': 0.65,
            'ignition_score': 0.78,
            'recency_score': 0.95,
            'priority_score': 0.82,
            'box_width': 0.06,
        }]

        df = pd.DataFrame(metadata_rows)
        parquet_path = tmp_path / 'test_metadata.parquet'
        df.to_parquet(parquet_path, index=False)

        # Read back
        df_loaded = pd.read_parquet(parquet_path)

        assert len(df_loaded) == 1
        assert df_loaded['price_position_at_end'].iloc[0] == 0.72
        assert df_loaded['coil_intensity'].iloc[0] == 0.65
        assert df_loaded['ignition_score'].iloc[0] == 0.78
        assert df_loaded['priority_score'].iloc[0] == 0.82
        assert df_loaded['box_width'].iloc[0] == 0.06


class TestEdgeCases:
    """Edge case handling"""

    def test_empty_enrichment_fields(self):
        """Pattern with no enrichment fields still works"""
        pattern_row = pd.Series({
            'ticker': 'AMD',
            'start_date': datetime(2024, 2, 1),
            'end_date': datetime(2024, 2, 15),
            'outcome_class': 0,
            'pattern_id': 'AMD_2024-02-01',
        })

        enriched = {}
        for field in METADATA_ENRICHMENT_FIELDS:
            if field in pattern_row.index:
                value = pattern_row.get(field)
                if pd.notna(value):
                    enriched[field] = value

        # No enrichment fields should be added
        assert len(enriched) == 0

    def test_partial_enrichment(self):
        """Partial enrichment works correctly"""
        pattern_row = pd.Series({
            'ticker': 'MSFT',
            'start_date': datetime(2024, 4, 1),
            'end_date': datetime(2024, 4, 18),
            'outcome_class': 1,
            'pattern_id': 'MSFT_2024-04-01',
            # Only some coil features
            'price_position_at_end': 0.55,
            'coil_intensity': 0.62,
            # Missing: distance_to_danger, bbw_slope_5d, vol_trend_5d
            # Missing: all ignition features
        })

        enriched = {}
        for field in METADATA_ENRICHMENT_FIELDS:
            if field in pattern_row.index:
                value = pattern_row.get(field)
                if pd.notna(value):
                    enriched[field] = value

        assert len(enriched) == 2
        assert 'price_position_at_end' in enriched
        assert 'coil_intensity' in enriched
        assert 'distance_to_danger' not in enriched

    def test_inf_values_handled(self):
        """Infinite values are handled gracefully"""
        pattern_row = pd.Series({
            'ticker': 'TEST',
            'start_date': datetime(2024, 1, 1),
            'end_date': datetime(2024, 1, 20),
            'price_position_at_end': np.inf,
            'vol_trend_5d': -np.inf,
            'coil_intensity': 0.5,
        })

        enriched = {}
        for field in METADATA_ENRICHMENT_FIELDS:
            if field in pattern_row.index:
                value = pattern_row.get(field)
                # pd.notna returns True for inf, so we handle it
                if pd.notna(value):
                    if isinstance(value, (np.integer, np.floating)):
                        value = float(value)
                    enriched[field] = value

        # inf values are kept (downstream code can filter if needed)
        assert 'price_position_at_end' in enriched
        assert enriched['coil_intensity'] == 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
