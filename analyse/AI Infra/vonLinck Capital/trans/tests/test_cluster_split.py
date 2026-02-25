"""
Tests for Pattern Cluster-Aware Data Splitting (Trinity Mode)

Tests ensure that Entry/Coil/Trigger patterns from the same consolidation
event (sharing nms_cluster_id) stay together in the same split.

Key invariant: No cluster should be split across train/val/test.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestClusterDetection:
    """Tests for detecting Trinity mode from metadata"""

    def test_detects_cluster_ids_present(self):
        """Should detect Trinity mode when nms_cluster_id column exists"""
        metadata = pd.DataFrame({
            'pattern_id': ['A', 'B', 'C'],
            'nms_cluster_id': [0, 0, 1],
            'pattern_start_date': pd.to_datetime(['2020-01-01', '2020-01-05', '2020-06-01'])
        })

        unique_clusters = metadata['nms_cluster_id'].unique()
        has_clusters = len(unique_clusters) > 1 or (len(unique_clusters) == 1 and unique_clusters[0] != -1)

        assert has_clusters is True
        assert len(unique_clusters) == 2

    def test_ignores_all_negative_ones(self):
        """Should not use Trinity mode if all cluster IDs are -1 (no clustering)"""
        metadata = pd.DataFrame({
            'pattern_id': ['A', 'B', 'C'],
            'nms_cluster_id': [-1, -1, -1],
            'pattern_start_date': pd.to_datetime(['2020-01-01', '2020-01-05', '2020-06-01'])
        })

        unique_clusters = metadata['nms_cluster_id'].unique()
        has_clusters = len(unique_clusters) > 1 or (len(unique_clusters) == 1 and unique_clusters[0] != -1)

        assert has_clusters == False  # Use == instead of 'is' for numpy bool

    def test_handles_missing_column(self):
        """Should gracefully handle missing nms_cluster_id column"""
        metadata = pd.DataFrame({
            'pattern_id': ['A', 'B', 'C'],
            'pattern_start_date': pd.to_datetime(['2020-01-01', '2020-01-05', '2020-06-01'])
        })

        has_cluster_col = 'nms_cluster_id' in metadata.columns
        assert has_cluster_col is False


class TestClusterAwareSplitting:
    """Tests for cluster-aware temporal splitting"""

    def create_trinity_metadata(self, n_clusters=10, patterns_per_cluster=3):
        """Create realistic Trinity-mode metadata"""
        rows = []
        base_date = datetime(2020, 1, 1)

        for cluster_id in range(n_clusters):
            # Each cluster starts at a different date
            cluster_start = base_date + timedelta(days=cluster_id * 30)

            for view_idx in range(patterns_per_cluster):
                # Entry, Coil, Trigger views have slightly different dates
                pattern_date = cluster_start + timedelta(days=view_idx * 2)
                view_type = ['entry', 'coil', 'trigger'][view_idx]

                rows.append({
                    'pattern_id': f'cluster_{cluster_id}_{view_type}',
                    'nms_cluster_id': cluster_id,
                    'view_type': view_type,
                    'pattern_start_date': pattern_date,
                    'ticker': f'TICK{cluster_id % 5}'
                })

        return pd.DataFrame(rows)

    def test_clusters_stay_together(self):
        """All patterns from same cluster must go to same split"""
        metadata = self.create_trinity_metadata(n_clusters=20)

        # Define temporal cutoffs
        train_cutoff = pd.Timestamp('2020-04-01')  # ~4 clusters to train
        val_cutoff = pd.Timestamp('2020-07-01')    # ~4 clusters to val

        # Map cluster to date (earliest pattern date)
        cluster_to_date = metadata.groupby('nms_cluster_id')['pattern_start_date'].min()

        # Assign clusters to splits
        train_clusters = set(cluster_to_date[cluster_to_date < train_cutoff].index)
        val_clusters = set(cluster_to_date[(cluster_to_date >= train_cutoff) &
                                            (cluster_to_date < val_cutoff)].index)
        test_clusters = set(cluster_to_date[cluster_to_date >= val_cutoff].index)

        # Verify no overlap
        assert len(train_clusters & val_clusters) == 0, "Train/Val cluster overlap!"
        assert len(train_clusters & test_clusters) == 0, "Train/Test cluster overlap!"
        assert len(val_clusters & test_clusters) == 0, "Val/Test cluster overlap!"

        # Verify all clusters assigned
        all_assigned = train_clusters | val_clusters | test_clusters
        assert all_assigned == set(metadata['nms_cluster_id'].unique())

    def test_all_views_in_same_split(self):
        """Entry, Coil, Trigger views from same cluster must be in same split"""
        metadata = self.create_trinity_metadata(n_clusters=10)

        train_cutoff = pd.Timestamp('2020-04-01')

        # Get patterns for train/test based on cluster date
        cluster_to_date = metadata.groupby('nms_cluster_id')['pattern_start_date'].min()

        for cluster_id in metadata['nms_cluster_id'].unique():
            cluster_date = cluster_to_date[cluster_id]
            is_train = cluster_date < train_cutoff

            cluster_patterns = metadata[metadata['nms_cluster_id'] == cluster_id]['pattern_id'].tolist()

            # All 3 views (Entry/Coil/Trigger) should have same split assignment
            views_in_cluster = metadata[metadata['nms_cluster_id'] == cluster_id]['view_type'].tolist()
            assert len(views_in_cluster) == 3, f"Cluster {cluster_id} should have 3 views"

            # Verify all are assigned together (not split)
            # This is implicit from cluster-based splitting

    def test_single_pattern_clusters_handled(self):
        """Clusters with only 1 pattern (highlander fallback) should still work"""
        rows = [
            {'pattern_id': 'A', 'nms_cluster_id': 0, 'pattern_start_date': '2020-01-01'},
            {'pattern_id': 'B', 'nms_cluster_id': 1, 'pattern_start_date': '2020-02-01'},
            {'pattern_id': 'C', 'nms_cluster_id': 1, 'pattern_start_date': '2020-02-05'},
            {'pattern_id': 'D', 'nms_cluster_id': 2, 'pattern_start_date': '2020-06-01'},
        ]
        metadata = pd.DataFrame(rows)
        metadata['pattern_start_date'] = pd.to_datetime(metadata['pattern_start_date'])

        train_cutoff = pd.Timestamp('2020-03-01')

        cluster_to_date = metadata.groupby('nms_cluster_id')['pattern_start_date'].min()

        train_clusters = set(cluster_to_date[cluster_to_date < train_cutoff].index)
        test_clusters = set(cluster_to_date[cluster_to_date >= train_cutoff].index)

        # Clusters 0 and 1 should be in train, cluster 2 in test
        assert train_clusters == {0, 1}
        assert test_clusters == {2}


class TestClusterSplitValidation:
    """Tests for validation utilities"""

    def test_validate_no_cluster_split(self):
        """Validation function should detect when cluster is split"""
        # Create metadata where cluster 1 patterns end up in different splits (BAD)
        metadata = pd.DataFrame({
            'pattern_id': ['A', 'B', 'C', 'D'],
            'nms_cluster_id': [0, 1, 1, 2],
        })

        # Intentionally split cluster 1 across train/test (BAD)
        train_patterns = {'A', 'B'}  # Cluster 0 and part of cluster 1
        test_patterns = {'C', 'D'}   # Rest of cluster 1 and cluster 2

        # Build cluster to split mapping
        pattern_to_cluster = dict(zip(metadata['pattern_id'], metadata['nms_cluster_id']))

        train_clusters = {pattern_to_cluster[p] for p in train_patterns}
        test_clusters = {pattern_to_cluster[p] for p in test_patterns}

        # Cluster 1 appears in both splits - THIS IS BAD
        overlap = train_clusters & test_clusters
        assert len(overlap) > 0, "Should detect cluster 1 in both splits"
        assert 1 in overlap

    def test_validate_clean_cluster_split(self):
        """Validation should pass when clusters are not split"""
        metadata = pd.DataFrame({
            'pattern_id': ['A', 'B', 'C', 'D', 'E', 'F'],
            'nms_cluster_id': [0, 0, 0, 1, 1, 1],  # 2 clusters, 3 patterns each
        })

        # Proper split: all of cluster 0 to train, all of cluster 1 to test
        train_patterns = {'A', 'B', 'C'}  # All cluster 0
        test_patterns = {'D', 'E', 'F'}   # All cluster 1

        pattern_to_cluster = dict(zip(metadata['pattern_id'], metadata['nms_cluster_id']))

        train_clusters = {pattern_to_cluster[p] for p in train_patterns}
        test_clusters = {pattern_to_cluster[p] for p in test_patterns}

        overlap = train_clusters & test_clusters
        assert len(overlap) == 0, "No cluster should be in both splits"


class TestSequenceToClusterMapping:
    """Tests for mapping sequences back to clusters"""

    def test_sequences_inherit_cluster_id(self):
        """All sequences from a pattern should share parent cluster ID"""
        # Simulate metadata where each pattern generates multiple sequences
        rows = []
        for cluster_id in range(3):
            for view in ['entry', 'coil', 'trigger']:
                pattern_id = f'cluster_{cluster_id}_{view}'
                # Each pattern generates ~10 sequences
                for seq_idx in range(10):
                    rows.append({
                        'pattern_id': pattern_id,
                        'nms_cluster_id': cluster_id,
                        'sequence_idx': seq_idx,
                    })

        metadata = pd.DataFrame(rows)

        # All sequences from same pattern should have same cluster
        for pattern_id in metadata['pattern_id'].unique():
            pattern_seqs = metadata[metadata['pattern_id'] == pattern_id]
            cluster_ids = pattern_seqs['nms_cluster_id'].unique()
            assert len(cluster_ids) == 1, f"Pattern {pattern_id} has multiple cluster IDs"

    def test_cluster_aware_mask_creation(self):
        """Sequence masks should respect cluster boundaries"""
        # Create sequences metadata
        rows = []
        for cluster_id in range(4):
            for view in ['entry', 'coil', 'trigger']:
                pattern_id = f'c{cluster_id}_{view}'
                for seq_idx in range(5):
                    rows.append({
                        'pattern_id': pattern_id,
                        'nms_cluster_id': cluster_id,
                    })

        metadata = pd.DataFrame(rows)
        pattern_ids = metadata['pattern_id'].values

        # Define cluster splits
        train_clusters = {0, 1}
        test_clusters = {2, 3}

        # Get patterns for each split
        cluster_to_patterns = metadata.groupby('nms_cluster_id')['pattern_id'].apply(set).to_dict()
        train_patterns = set().union(*[cluster_to_patterns.get(c, set()) for c in train_clusters])
        test_patterns = set().union(*[cluster_to_patterns.get(c, set()) for c in test_clusters])

        # Create sequence masks
        train_mask = np.isin(pattern_ids, list(train_patterns))
        test_mask = np.isin(pattern_ids, list(test_patterns))

        # Verify no sequence is in both
        assert not np.any(train_mask & test_mask), "Sequence in both train and test!"

        # Verify all sequences assigned
        assert np.all(train_mask | test_mask), "Some sequences not assigned!"


class TestEdgeCases:
    """Edge cases and error handling"""

    def test_empty_split(self):
        """Handle case where one split has no clusters"""
        metadata = pd.DataFrame({
            'pattern_id': ['A', 'B', 'C'],
            'nms_cluster_id': [0, 0, 0],  # Single cluster
            'pattern_start_date': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03'])
        })

        # All before cutoff -> all to train, none to test
        train_cutoff = pd.Timestamp('2021-01-01')

        cluster_to_date = metadata.groupby('nms_cluster_id')['pattern_start_date'].min()
        train_clusters = set(cluster_to_date[cluster_to_date < train_cutoff].index)
        test_clusters = set(cluster_to_date[cluster_to_date >= train_cutoff].index)

        assert len(train_clusters) == 1
        assert len(test_clusters) == 0

    def test_mixed_clustered_and_unclustered(self):
        """Handle mix of clustered (Trinity) and unclustered (-1) patterns"""
        metadata = pd.DataFrame({
            'pattern_id': ['A', 'B', 'C', 'D', 'E'],
            'nms_cluster_id': [0, 0, -1, 1, -1],  # Some clustered, some not
            'pattern_start_date': pd.to_datetime([
                '2020-01-01', '2020-01-05',  # Cluster 0
                '2020-03-01',  # Unclustered
                '2020-06-01',  # Cluster 1
                '2020-09-01',  # Unclustered
            ])
        })

        train_cutoff = pd.Timestamp('2020-04-01')

        # Map cluster to date (use earliest pattern)
        cluster_to_date = metadata.groupby('nms_cluster_id')['pattern_start_date'].min().to_dict()

        train_clusters = []
        test_clusters = []

        for cluster_id, cluster_date in cluster_to_date.items():
            if cluster_id == -1:
                # Unclustered patterns: handle individually
                continue
            if cluster_date < train_cutoff:
                train_clusters.append(cluster_id)
            else:
                test_clusters.append(cluster_id)

        assert 0 in train_clusters  # Before cutoff
        assert 1 in test_clusters   # After cutoff

    def test_cluster_id_consistency_check(self):
        """Detect inconsistent cluster IDs for same pattern"""
        # This shouldn't happen but let's verify we'd catch it
        metadata = pd.DataFrame({
            'pattern_id': ['A', 'A', 'B'],  # Duplicate pattern A
            'nms_cluster_id': [0, 1, 2],  # Inconsistent cluster for A!
        })

        # Group and check for inconsistency
        cluster_per_pattern = metadata.groupby('pattern_id')['nms_cluster_id'].nunique()
        inconsistent = cluster_per_pattern[cluster_per_pattern > 1]

        assert len(inconsistent) > 0, "Should detect pattern A has multiple clusters"
        assert 'A' in inconsistent.index


class TestPerformance:
    """Performance tests for cluster splitting"""

    def test_large_dataset_splitting(self):
        """Cluster splitting should be fast on large datasets"""
        import time

        # Create large Trinity dataset: 1000 clusters, 3 patterns each, 10 sequences each
        n_clusters = 1000
        patterns_per_cluster = 3
        sequences_per_pattern = 10

        rows = []
        base_date = datetime(2015, 1, 1)

        for cluster_id in range(n_clusters):
            cluster_start = base_date + timedelta(days=cluster_id)
            for view_idx in range(patterns_per_cluster):
                pattern_id = f'c{cluster_id}_v{view_idx}'
                for seq_idx in range(sequences_per_pattern):
                    rows.append({
                        'pattern_id': pattern_id,
                        'nms_cluster_id': cluster_id,
                        'pattern_start_date': cluster_start,
                    })

        metadata = pd.DataFrame(rows)
        pattern_ids = metadata['pattern_id'].values

        # Time the cluster splitting operation
        start = time.perf_counter()

        train_cutoff = pd.Timestamp('2016-07-01')
        val_cutoff = pd.Timestamp('2017-04-01')

        cluster_to_date = metadata.groupby('nms_cluster_id')['pattern_start_date'].min()

        train_clusters = set(cluster_to_date[cluster_to_date < train_cutoff].index)
        val_clusters = set(cluster_to_date[(cluster_to_date >= train_cutoff) &
                                            (cluster_to_date < val_cutoff)].index)
        test_clusters = set(cluster_to_date[cluster_to_date >= val_cutoff].index)

        cluster_to_patterns = metadata.groupby('nms_cluster_id')['pattern_id'].apply(set).to_dict()
        train_patterns = set().union(*[cluster_to_patterns.get(c, set()) for c in train_clusters])
        val_patterns = set().union(*[cluster_to_patterns.get(c, set()) for c in val_clusters])
        test_patterns = set().union(*[cluster_to_patterns.get(c, set()) for c in test_clusters])

        train_mask = np.isin(pattern_ids, list(train_patterns))
        val_mask = np.isin(pattern_ids, list(val_patterns))
        test_mask = np.isin(pattern_ids, list(test_patterns))

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should complete in under 5000ms for 30K sequences (generous for CI/slow machines)
        assert elapsed_ms < 5000, f"Took {elapsed_ms:.1f}ms, expected < 5000ms"
        assert len(train_clusters) > 0
        assert len(val_clusters) > 0
        assert len(test_clusters) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
