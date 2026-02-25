"""
Checkpoint Manager for AIv3 Training Pipeline

Saves intermediate results to GCS to avoid memory issues and enable resumable training.

Usage:
    checkpoint_mgr = CheckpointManager(bucket_name='your-bucket')

    # Save checkpoint
    checkpoint_mgr.save('step1_patterns', pattern_results)

    # Load checkpoint
    pattern_results = checkpoint_mgr.load('step1_patterns')

    # Resume training from last checkpoint
    last_step = checkpoint_mgr.get_last_checkpoint()
"""

import pickle
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, List
import os

try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    logging.warning("google-cloud-storage not installed. Checkpoints will be local only.")

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages training checkpoints with GCS backup.

    Automatically saves to both local disk and GCS for redundancy.
    Enables resumable training and memory-efficient processing.
    """

    def __init__(
        self,
        local_dir: str = 'output/checkpoints',
        bucket_name: Optional[str] = None,
        project_id: Optional[str] = None,
        gcs_prefix: str = 'aiv3_checkpoints'
    ):
        """
        Initialize checkpoint manager.

        Args:
            local_dir: Local directory for checkpoints
            bucket_name: GCS bucket name (optional)
            project_id: GCS project ID (optional)
            gcs_prefix: Prefix for GCS checkpoint paths
        """
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)

        self.bucket_name = bucket_name or os.getenv('GCS_BUCKET_NAME')
        self.project_id = project_id or os.getenv('PROJECT_ID')
        self.gcs_prefix = gcs_prefix

        # Initialize GCS client if available
        self.gcs_client = None
        self.bucket = None
        if GCS_AVAILABLE and self.bucket_name:
            try:
                self.gcs_client = storage.Client(project=self.project_id)
                self.bucket = self.gcs_client.bucket(self.bucket_name)
                logger.info(f"Checkpoint manager initialized with GCS: {self.bucket_name}/{self.gcs_prefix}")
            except Exception as e:
                logger.warning(f"Failed to initialize GCS client: {e}. Using local checkpoints only.")
        else:
            logger.info("Checkpoint manager initialized (local only)")

        # Track checkpoint metadata
        self.metadata_file = self.local_dir / 'checkpoint_metadata.json'
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load checkpoint metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {'checkpoints': [], 'last_checkpoint': None}

    def _save_metadata(self):
        """Save checkpoint metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def save(
        self,
        checkpoint_name: str,
        data: Any,
        metadata: Optional[Dict] = None,
        upload_to_gcs: bool = True
    ) -> bool:
        """
        Save checkpoint to local disk and optionally GCS.

        Args:
            checkpoint_name: Name of checkpoint (e.g., 'step1_patterns')
            data: Data to save (will be pickled)
            metadata: Optional metadata dictionary
            upload_to_gcs: Upload to GCS if available

        Returns:
            True if saved successfully
        """
        timestamp = datetime.now().isoformat()

        # Save locally
        local_path = self.local_dir / f'{checkpoint_name}.pkl'
        try:
            with open(local_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"✓ Checkpoint saved locally: {local_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint locally: {e}")
            return False

        # Upload to GCS if available and requested
        if upload_to_gcs and self.bucket:
            try:
                gcs_path = f'{self.gcs_prefix}/{checkpoint_name}.pkl'
                blob = self.bucket.blob(gcs_path)
                blob.upload_from_filename(str(local_path))
                logger.info(f"✓ Checkpoint uploaded to GCS: gs://{self.bucket_name}/{gcs_path}")
            except Exception as e:
                logger.warning(f"Failed to upload checkpoint to GCS: {e}")

        # Update metadata
        checkpoint_info = {
            'name': checkpoint_name,
            'timestamp': timestamp,
            'local_path': str(local_path),
            'size_bytes': local_path.stat().st_size,
            'metadata': metadata or {}
        }

        # Remove old entry if exists
        self.metadata['checkpoints'] = [
            c for c in self.metadata['checkpoints'] if c['name'] != checkpoint_name
        ]
        self.metadata['checkpoints'].append(checkpoint_info)
        self.metadata['last_checkpoint'] = checkpoint_name
        self._save_metadata()

        return True

    def load(
        self,
        checkpoint_name: str,
        try_gcs_first: bool = False
    ) -> Optional[Any]:
        """
        Load checkpoint from local disk or GCS.

        Args:
            checkpoint_name: Name of checkpoint to load
            try_gcs_first: Try loading from GCS before local

        Returns:
            Loaded data or None if not found
        """
        local_path = self.local_dir / f'{checkpoint_name}.pkl'

        # Try GCS first if requested and available
        if try_gcs_first and self.bucket:
            try:
                gcs_path = f'{self.gcs_prefix}/{checkpoint_name}.pkl'
                blob = self.bucket.blob(gcs_path)
                blob.download_to_filename(str(local_path))
                logger.info(f"✓ Checkpoint downloaded from GCS: {gcs_path}")
            except Exception as e:
                logger.warning(f"Failed to download from GCS: {e}. Trying local...")

        # Load from local
        if local_path.exists():
            try:
                with open(local_path, 'rb') as f:
                    data = pickle.load(f)
                logger.info(f"✓ Checkpoint loaded: {checkpoint_name}")
                return data
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                return None

        logger.warning(f"Checkpoint not found: {checkpoint_name}")
        return None

    def exists(self, checkpoint_name: str) -> bool:
        """Check if checkpoint exists locally."""
        local_path = self.local_dir / f'{checkpoint_name}.pkl'
        return local_path.exists()

    def get_last_checkpoint(self) -> Optional[str]:
        """Get name of last saved checkpoint."""
        return self.metadata.get('last_checkpoint')

    def list_checkpoints(self) -> List[Dict]:
        """List all available checkpoints."""
        return self.metadata.get('checkpoints', [])

    def delete(self, checkpoint_name: str, delete_from_gcs: bool = False):
        """
        Delete checkpoint.

        Args:
            checkpoint_name: Name of checkpoint to delete
            delete_from_gcs: Also delete from GCS
        """
        # Delete local
        local_path = self.local_dir / f'{checkpoint_name}.pkl'
        if local_path.exists():
            local_path.unlink()
            logger.info(f"✓ Checkpoint deleted locally: {checkpoint_name}")

        # Delete from GCS if requested
        if delete_from_gcs and self.bucket:
            try:
                gcs_path = f'{self.gcs_prefix}/{checkpoint_name}.pkl'
                blob = self.bucket.blob(gcs_path)
                blob.delete()
                logger.info(f"✓ Checkpoint deleted from GCS: {gcs_path}")
            except Exception as e:
                logger.warning(f"Failed to delete from GCS: {e}")

        # Update metadata
        self.metadata['checkpoints'] = [
            c for c in self.metadata['checkpoints'] if c['name'] != checkpoint_name
        ]
        if self.metadata.get('last_checkpoint') == checkpoint_name:
            self.metadata['last_checkpoint'] = None
        self._save_metadata()

    def clear_all(self, clear_gcs: bool = False):
        """
        Clear all checkpoints.

        Args:
            clear_gcs: Also clear GCS checkpoints
        """
        # Clear local
        for checkpoint_file in self.local_dir.glob('*.pkl'):
            checkpoint_file.unlink()

        # Clear GCS if requested
        if clear_gcs and self.bucket:
            try:
                blobs = self.bucket.list_blobs(prefix=self.gcs_prefix)
                for blob in blobs:
                    blob.delete()
                logger.info(f"✓ All GCS checkpoints cleared")
            except Exception as e:
                logger.warning(f"Failed to clear GCS checkpoints: {e}")

        # Reset metadata
        self.metadata = {'checkpoints': [], 'last_checkpoint': None}
        self._save_metadata()

        logger.info("✓ All checkpoints cleared")


def create_checkpoint_manager(
    bucket_name: Optional[str] = None,
    local_dir: str = 'output/checkpoints'
) -> CheckpointManager:
    """
    Factory function to create checkpoint manager.

    Args:
        bucket_name: GCS bucket name (uses env var if None)
        local_dir: Local checkpoint directory

    Returns:
        CheckpointManager instance
    """
    return CheckpointManager(
        local_dir=local_dir,
        bucket_name=bucket_name
    )
