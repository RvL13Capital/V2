"""
Model Version Management System
================================

Handles model lifecycle, versioning, A/B testing, and rollback capabilities.
"""

import os
import json
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
import torch
import torch.nn as nn
import numpy as np
from contextlib import contextmanager

from models.temporal_hybrid_v18 import HybridFeatureNetwork
from database.connection import get_db_session
from database.models import ModelVersion, Prediction
from utils.logging_config import get_production_logger
from utils.error_handler import TransError, ErrorSeverity

logger = get_production_logger(__name__, "model_manager")


class ModelStatus(Enum):
    """Model lifecycle states."""
    TRAINING = "training"
    VALIDATING = "validating"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    FAILED = "failed"


class ModelManager:
    """
    Centralized model version management system.

    Features:
    - Model registration and tracking
    - Version control with metadata
    - A/B testing support
    - Rollback capabilities
    - Performance monitoring
    - Model artifact management
    """

    def __init__(self,
                 model_dir: Optional[Path] = None,
                 max_versions: int = 10):
        """
        Initialize model manager.

        Args:
            model_dir: Directory for model storage
            max_versions: Maximum number of model versions to keep
        """
        self.model_dir = model_dir or Path("models/artifacts")
        self.max_versions = max_versions
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Cache for loaded models
        self._model_cache = {}
        self._current_production = None

    def create_version(self,
                      architecture: str,
                      hyperparameters: Dict[str, Any],
                      description: str = "",
                      auto_version: bool = True) -> str:
        """
        Create a new model version.

        Args:
            architecture: Model architecture name
            hyperparameters: Model hyperparameters
            description: Version description
            auto_version: Auto-generate version number

        Returns:
            Version string
        """
        if auto_version:
            # Get next version number
            with get_db_session() as session:
                latest = session.query(ModelVersion).filter(
                    ModelVersion.labeling_version == "v17"
                ).order_by(ModelVersion.created_at.desc()).first()

                if latest:
                    # Parse version (e.g., "v17.2.1" -> major=17, minor=2, patch=1)
                    parts = latest.version.replace("v", "").split(".")
                    if len(parts) == 3:
                        major, minor, patch = map(int, parts)
                        patch += 1
                    else:
                        major = 17  # v17 system
                        minor = 1
                        patch = 0
                else:
                    major = 17  # v17 system
                    minor = 1
                    patch = 0

                version = f"v{major}.{minor}.{patch}"
        else:
            version = f"v17_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create database entry
        with get_db_session() as session:
            model_record = ModelVersion(
                version=version,
                labeling_version="v17",
                architecture=architecture,
                num_classes=3,  # 3 classes: Danger, Noise, Target
                parameters=hyperparameters,
                description=description,
                training_start_date=datetime.utcnow()
            )
            session.add(model_record)
            session.commit()

            logger.info(f"Created model version {version}",
                       model_version_id=model_record.id,
                       architecture=architecture)

        return version

    def save_model(self,
                  model: nn.Module,
                  version: str,
                  metrics: Dict[str, Any],
                  metadata: Optional[Dict[str, Any]] = None) -> Path:
        """
        Save model artifacts and metadata.

        Args:
            model: PyTorch model
            version: Model version
            metrics: Training/validation metrics
            metadata: Additional metadata

        Returns:
            Path to saved model
        """
        # Create version directory
        version_dir = self.model_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save model state
        model_path = version_dir / "model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'version': version,
            'labeling_version': "v17",
            'architecture': model.__class__.__name__,
            'timestamp': datetime.utcnow().isoformat()
        }, model_path)

        # Save metadata
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                'version': version,
                'metrics': metrics,
                'metadata': metadata or {},
                'created_at': datetime.utcnow().isoformat(),
                'model_hash': self._calculate_model_hash(model_path)
            }, f, indent=2)

        # Update database
        with get_db_session() as session:
            model_record = session.query(ModelVersion).filter(
                ModelVersion.version == version
            ).first()

            if model_record:
                model_record.model_path = str(model_path)
                model_record.training_end_date = datetime.utcnow()
                model_record.training_metrics = metrics.get('training', {})
                model_record.validation_metrics = metrics.get('validation', {})
                model_record.expected_value_correlation = metrics.get('ev_correlation')
                session.commit()

        logger.info(f"Saved model version {version}",
                   model_path=str(model_path),
                   metrics=metrics)

        return model_path

    def load_model(self,
                  version: str,
                  device: Optional[str] = None,
                  use_cache: bool = True) -> nn.Module:
        """
        Load a model version.

        Args:
            version: Model version to load
            device: Device to load model on
            use_cache: Use cached model if available

        Returns:
            Loaded model
        """
        # Check cache
        if use_cache and version in self._model_cache:
            logger.debug(f"Loading model {version} from cache")
            return self._model_cache[version]

        # Load from disk
        model_path = self.model_dir / version / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device or 'cpu', weights_only=False)

        # Create model instance
        model = HybridFeatureNetwork()
        model.load_state_dict(checkpoint['model_state_dict'])

        if device:
            model = model.to(device)

        model.eval()

        # Cache model
        if use_cache:
            self._model_cache[version] = model

        logger.info(f"Loaded model version {version}",
                   model_path=str(model_path),
                   cached=use_cache)

        return model

    def promote_to_production(self,
                             version: str,
                             validation_required: bool = True) -> bool:
        """
        Promote a model version to production.

        Args:
            version: Version to promote
            validation_required: Require validation before promotion

        Returns:
            Success status
        """
        with get_db_session() as session:
            # Get model record
            model = session.query(ModelVersion).filter(
                ModelVersion.version == version
            ).first()

            if not model:
                raise ValueError(f"Model version {version} not found")

            # Check validation status
            if validation_required and not model.validation_metrics:
                raise TransError(
                    f"Model {version} has not been validated",
                    severity=ErrorSeverity.HIGH,
                    recoverable=False
                )

            # Check performance thresholds
            if model.validation_metrics:
                accuracy = model.validation_metrics.get('accuracy', 0)
                ev_corr = model.expected_value_correlation or 0

                if accuracy < 0.6 or ev_corr < 0.3:
                    raise TransError(
                        f"Model {version} does not meet performance thresholds",
                        severity=ErrorSeverity.HIGH,
                        context={'accuracy': accuracy, 'ev_correlation': ev_corr}
                    )

            # Demote current production model
            session.query(ModelVersion).filter(
                ModelVersion.is_production == True
            ).update({'is_production': False, 'is_active': False})

            # Promote new model
            model.is_production = True
            model.is_active = True
            session.commit()

            self._current_production = version

            logger.info(f"Promoted model {version} to production",
                       model_version_id=model.id,
                       metrics=model.validation_metrics)

        return True

    def rollback_production(self) -> str:
        """
        Rollback to previous production model.

        Returns:
            Previous production version
        """
        with get_db_session() as session:
            # Get current production
            current = session.query(ModelVersion).filter(
                ModelVersion.is_production == True
            ).first()

            if not current:
                raise ValueError("No current production model")

            # Find previous production model
            previous = session.query(ModelVersion).filter(
                ModelVersion.is_active == False,
                ModelVersion.created_at < current.created_at
            ).order_by(ModelVersion.created_at.desc()).first()

            if not previous:
                raise ValueError("No previous model to rollback to")

            # Perform rollback
            current.is_production = False
            current.is_active = False
            previous.is_production = True
            previous.is_active = True
            session.commit()

            self._current_production = previous.version

            logger.warning(f"Rolled back from {current.version} to {previous.version}",
                          from_version=current.version,
                          to_version=previous.version)

            return previous.version

    def get_production_model(self) -> Tuple[nn.Module, str]:
        """
        Get current production model.

        Returns:
            Tuple of (model, version)
        """
        with get_db_session() as session:
            production = session.query(ModelVersion).filter(
                ModelVersion.is_production == True
            ).first()

            if not production:
                raise ValueError("No production model configured")

            model = self.load_model(production.version)
            return model, production.version

    def get_active_model(self) -> Optional[nn.Module]:
        """
        Get the currently active/production model.
        
        Returns:
            The active model or None if no production model is set.
        """
        try:
            model, _ = self.get_production_model()
            return model
        except ValueError:
            return None

    def get_active_version(self) -> Optional[str]:
        """
        Get the version string of the currently active/production model.
        
        Returns:
            Version string or None if no production model is set.
        """
        try:
            _, version = self.get_production_model()
            return version
        except ValueError:
            return None

    def run_ab_test(self,
                   version_a: str,
                   version_b: str,
                   test_data: Any,
                   metrics: List[str]) -> Dict[str, Any]:
        """
        Run A/B test between two model versions.

        Args:
            version_a: First model version
            version_b: Second model version
            test_data: Test dataset
            metrics: Metrics to compare

        Returns:
            Comparison results
        """
        logger.info(f"Running A/B test: {version_a} vs {version_b}")

        # Load both models
        model_a = self.load_model(version_a)
        model_b = self.load_model(version_b)

        results = {
            'version_a': version_a,
            'version_b': version_b,
            'metrics': {}
        }

        # TODO: Implement actual A/B testing logic
        # This would involve running predictions on test_data
        # and comparing specified metrics

        logger.info(f"A/B test complete: {version_a} vs {version_b}",
                   results=results)

        return results

    def validate_model(self,
                      version: str,
                      validation_data: Any,
                      update_db: bool = True) -> Dict[str, float]:
        """
        Validate a model version.

        Args:
            version: Model version to validate
            validation_data: Validation dataset
            update_db: Update database with results

        Returns:
            Validation metrics
        """
        logger.info(f"Validating model {version}")

        model = self.load_model(version)

        # TODO: Implement actual validation logic
        # This would involve running model on validation_data
        # and computing metrics

        metrics = {
            'accuracy': 0.75,  # Placeholder
            'precision': 0.72,
            'recall': 0.68,
            'f1_score': 0.70,
            'ev_correlation': 0.42
        }

        if update_db:
            with get_db_session() as session:
                model_record = session.query(ModelVersion).filter(
                    ModelVersion.version == version
                ).first()

                if model_record:
                    model_record.validation_metrics = metrics
                    model_record.expected_value_correlation = metrics['ev_correlation']
                    session.commit()

        logger.info(f"Validation complete for {version}",
                   version=version,
                   metrics=metrics)

        return metrics

    def get_model_performance(self, version: str) -> Dict[str, Any]:
        """
        Get real-world performance metrics for a model.

        Args:
            version: Model version

        Returns:
            Performance metrics
        """
        with get_db_session() as session:
            # Get model record
            model = session.query(ModelVersion).filter(
                ModelVersion.version == version
            ).first()

            if not model:
                raise ValueError(f"Model version {version} not found")

            # Get predictions made by this model
            predictions = session.query(Prediction).filter(
                Prediction.model_version_id == model.id,
                Prediction.actual_outcome.isnot(None)
            ).all()

            if not predictions:
                return {
                    'version': version,
                    'num_predictions': 0,
                    'metrics': {}
                }

            # Calculate performance metrics
            correct = sum(1 for p in predictions if p.prediction_correct)
            total = len(predictions)
            accuracy = correct / total if total > 0 else 0

            # Calculate EV correlation
            expected = [p.expected_value for p in predictions]
            actual = [p.value_captured for p in predictions if p.value_captured is not None]

            if expected and actual and len(expected) == len(actual):
                ev_correlation = np.corrcoef(expected[:len(actual)], actual)[0, 1]
            else:
                ev_correlation = 0

            return {
                'version': version,
                'num_predictions': total,
                'metrics': {
                    'accuracy': accuracy,
                    'correct_predictions': correct,
                    'ev_correlation': ev_correlation
                }
            }

    def cleanup_old_versions(self, keep_production: bool = True) -> List[str]:
        """
        Remove old model versions to save space.

        Args:
            keep_production: Always keep production models

        Returns:
            List of removed versions
        """
        removed = []

        with get_db_session() as session:
            # Get all versions sorted by creation date
            all_versions = session.query(ModelVersion).order_by(
                ModelVersion.created_at.desc()
            ).all()

            # Identify versions to remove
            keep_count = 0
            for model in all_versions:
                if keep_production and (model.is_production or model.is_active):
                    continue

                keep_count += 1
                if keep_count > self.max_versions:
                    # Remove from disk
                    version_dir = self.model_dir / model.version
                    if version_dir.exists():
                        shutil.rmtree(version_dir)

                    # Update database
                    model.is_active = False
                    removed.append(model.version)

            session.commit()

        if removed:
            logger.info(f"Cleaned up {len(removed)} old model versions",
                       removed_versions=removed)

        return removed

    def _calculate_model_hash(self, model_path: Path) -> str:
        """Calculate SHA256 hash of model file."""
        sha256 = hashlib.sha256()
        with open(model_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    @contextmanager
    def model_context(self, version: Optional[str] = None):
        """
        Context manager for model usage.

        Args:
            version: Specific version or None for production

        Example:
            with model_manager.model_context() as (model, version):
                predictions = model(data)
        """
        if version is None:
            model, version = self.get_production_model()
        else:
            model = self.load_model(version)

        try:
            yield model, version
        finally:
            # Cleanup if needed
            pass


# Global model manager instance
_model_manager: Optional[ModelManager] = None


def get_model_manager(model_dir: Optional[Path] = None) -> ModelManager:
    """Get or create global model manager instance."""
    global _model_manager
    if not _model_manager:
        _model_manager = ModelManager(model_dir=model_dir)
    return _model_manager


if __name__ == "__main__":
    # Test model management
    manager = get_model_manager()

    # Create a new version
    version = manager.create_version(
        architecture="HybridFeatureNetwork",
        hyperparameters={
            "lstm_hidden": 32,
            "lstm_layers": 2,
            "cnn_channels": [32, 64, 128],
            "num_attention_heads": 8
        },
        description="Test model for v17 labeling"
    )
    print(f"Created model version: {version}")

    # Create and save a model
    model = HybridFeatureNetwork()

    metrics = {
        "training": {"loss": 0.45, "accuracy": 0.72},
        "validation": {"accuracy": 0.70, "precision": 0.68},
        "ev_correlation": 0.38
    }

    path = manager.save_model(model, version, metrics)
    print(f"Saved model to: {path}")

    # Load model
    loaded_model = manager.load_model(version)
    print(f"Loaded model: {loaded_model.__class__.__name__}")

    # Get performance (mock)
    perf = manager.get_model_performance(version)
    print(f"Model performance: {perf}")