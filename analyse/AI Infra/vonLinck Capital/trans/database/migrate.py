"""
Database Migration Script
=========================

Initialize and migrate the TRANS production database.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text, inspect
from database.connection import db_manager, DatabaseConfig
from database.models import Base, Pattern, Prediction, ModelVersion, SystemLog, MetricSnapshot, TaskQueue, MarketPhase
from utils.logging_config import get_production_logger
from config.production import get_config

logger = get_production_logger(__name__, "migrate")


def create_tables():
    """Create all database tables."""
    logger.info("Creating database tables...")

    try:
        # Initialize database connection
        db_manager.initialize()

        # Create all tables defined in models
        Base.metadata.create_all(db_manager.engine)

        logger.info("Database tables created successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to create tables: {e}", exc_info=True)
        return False


def verify_tables():
    """Verify all required tables exist."""
    logger.info("Verifying database tables...")

    required_tables = [
        'patterns',
        'predictions',
        'model_versions',
        'system_logs',
        'metric_snapshots',
        'task_queue',
        'market_phases'
    ]

    try:
        inspector = inspect(db_manager.engine)
        existing_tables = inspector.get_table_names()

        missing_tables = []
        for table in required_tables:
            if table not in existing_tables:
                missing_tables.append(table)
                logger.warning(f"Missing table: {table}")
            else:
                logger.info(f"✓ Table exists: {table}")

        if missing_tables:
            logger.error(f"Missing tables: {missing_tables}")
            return False

        logger.info("All required tables verified")
        return True

    except Exception as e:
        logger.error(f"Failed to verify tables: {e}", exc_info=True)
        return False


def create_indexes():
    """Create additional database indexes for performance."""
    logger.info("Creating database indexes...")

    indexes = [
        # Pattern indexes
        "CREATE INDEX IF NOT EXISTS idx_patterns_ticker_date ON patterns(ticker, end_date)",
        "CREATE INDEX IF NOT EXISTS idx_patterns_status ON patterns(status)",
        "CREATE INDEX IF NOT EXISTS idx_patterns_labeling ON patterns(labeling_version)",

        # Prediction indexes
        "CREATE INDEX IF NOT EXISTS idx_predictions_pattern ON predictions(pattern_id)",
        "CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_version_id)",
        "CREATE INDEX IF NOT EXISTS idx_predictions_signal ON predictions(signal_strength)",

        # System log indexes
        "CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON system_logs(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_logs_level ON system_logs(level)",
        "CREATE INDEX IF NOT EXISTS idx_logs_component ON system_logs(component)",

        # Metric indexes
        "CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metric_snapshots(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_metrics_period ON metric_snapshots(period)",

        # Task queue indexes
        "CREATE INDEX IF NOT EXISTS idx_tasks_status ON task_queue(status)",
        "CREATE INDEX IF NOT EXISTS idx_tasks_type ON task_queue(task_type)",

        # Market phase indexes
        "CREATE INDEX IF NOT EXISTS idx_market_phases_index_symbol ON market_phases(index_symbol)",
        "CREATE INDEX IF NOT EXISTS idx_market_phases_date ON market_phases(date)",
        "CREATE INDEX IF NOT EXISTS idx_market_phases_phase ON market_phases(phase)"
    ]

    try:
        with db_manager.get_session() as session:
            for index_sql in indexes:
                try:
                    session.execute(text(index_sql))
                    logger.info(f"Created index: {index_sql.split('ON')[1].split('(')[0].strip()}")
                except Exception as e:
                    # Index might already exist
                    logger.debug(f"Index creation skipped: {e}")

            session.commit()

        logger.info("Database indexes created successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to create indexes: {e}", exc_info=True)
        return False


def create_views():
    """Create database views for common queries."""
    logger.info("Creating database views...")

    views = {
        "active_patterns": """
            CREATE OR REPLACE VIEW active_patterns AS
            SELECT
                p.*,
                pred.expected_value,
                pred.signal_strength,
                pred.predicted_class
            FROM patterns p
            LEFT JOIN LATERAL (
                SELECT * FROM predictions
                WHERE pattern_id = p.pattern_id
                ORDER BY prediction_date DESC
                LIMIT 1
            ) pred ON true
            WHERE p.status IN ('DETECTED', 'LABELED')
            AND p.outcome_class IS NOT NULL
        """,

        "strong_signals": """
            CREATE OR REPLACE VIEW strong_signals AS
            SELECT
                p.ticker,
                p.pattern_id,
                p.start_date,
                p.end_date,
                pred.expected_value,
                pred.signal_strength,
                pred.class_probabilities
            FROM patterns p
            JOIN predictions pred ON p.pattern_id = pred.pattern_id
            WHERE pred.signal_strength IN ('STRONG', 'GOOD')
            AND pred.prediction_date >= CURRENT_DATE - INTERVAL '7 days'
            ORDER BY pred.expected_value DESC
        """,

        "model_performance": """
            CREATE OR REPLACE VIEW model_performance AS
            SELECT
                mv.version,
                mv.architecture,
                COUNT(p.id) as total_predictions,
                AVG(CASE WHEN p.prediction_correct THEN 1 ELSE 0 END) as accuracy,
                AVG(p.expected_value) as avg_expected_value,
                AVG(p.value_captured) as avg_value_captured
            FROM model_versions mv
            LEFT JOIN predictions p ON mv.id = p.model_version_id
            WHERE p.actual_outcome IS NOT NULL
            GROUP BY mv.id, mv.version, mv.architecture
        """
    }

    # Note: Views syntax varies between databases
    # This example uses PostgreSQL syntax
    db_type = db_manager.config.db_type

    if db_type == 'sqlite':
        logger.info("SQLite detected - skipping view creation (not fully supported)")
        return True

    try:
        with db_manager.get_session() as session:
            for view_name, view_sql in views.items():
                try:
                    session.execute(text(view_sql))
                    logger.info(f"Created view: {view_name}")
                except Exception as e:
                    logger.warning(f"View creation failed for {view_name}: {e}")

            session.commit()

        logger.info("Database views created")
        return True

    except Exception as e:
        logger.error(f"Failed to create views: {e}", exc_info=True)
        return False


def insert_initial_data():
    """Insert initial data if needed."""
    logger.info("Checking for initial data...")

    try:
        with db_manager.get_session() as session:
            # Check if we have any model versions
            model_count = session.query(ModelVersion).count()

            if model_count == 0:
                # Create initial model version
                initial_model = ModelVersion(
                    version="v17.1.0",
                    labeling_version="v17",
                    architecture="HybridFeatureNetwork",
                    num_classes=3,
                    parameters={
                        "lstm_hidden": 32,
                        "lstm_layers": 2,
                        "cnn_channels": [32, 64, 128],
                        "num_attention_heads": 8
                    },
                    description="Initial v17 production model",
                    is_active=True,
                    is_production=True
                )
                session.add(initial_model)
                session.commit()
                logger.info("Created initial model version")

            # Add initial system log entry
            log_count = session.query(SystemLog).count()
            if log_count == 0:
                initial_log = SystemLog(
                    level="INFO",
                    component="migrate",
                    message="Database initialized"
                )
                session.add(initial_log)
                session.commit()
                logger.info("Created initial system log entry")

        return True

    except Exception as e:
        logger.error(f"Failed to insert initial data: {e}", exc_info=True)
        return False


def run_migration():
    """Run complete database migration."""
    logger.info("=" * 50)
    logger.info("Starting database migration")
    logger.info("=" * 50)

    # Load configuration
    config = get_config()
    logger.info(f"Environment: {config.environment.value}")
    logger.info(f"Database: {config.database.url}")

    # Step 1: Create tables
    if not create_tables():
        logger.error("Migration failed: Could not create tables")
        return False

    # Step 2: Verify tables
    if not verify_tables():
        logger.error("Migration failed: Table verification failed")
        return False

    # Step 3: Create indexes
    if not create_indexes():
        logger.warning("Index creation failed (non-critical)")

    # Step 4: Create views
    if not create_views():
        logger.warning("View creation failed (non-critical)")

    # Step 5: Insert initial data
    if not insert_initial_data():
        logger.warning("Initial data insertion failed (non-critical)")

    # Final verification
    health = db_manager.health_check()
    if health['status'] == 'healthy':
        logger.info("=" * 50)
        logger.info("✓ Database migration completed successfully")
        logger.info(f"  Patterns: {health.get('pattern_count', 0)}")
        logger.info(f"  Predictions: {health.get('prediction_count', 0)}")
        logger.info(f"  Models: {health.get('model_count', 0)}")
        logger.info("=" * 50)
        return True
    else:
        logger.error(f"Database health check failed: {health}")
        return False


def rollback_migration():
    """Rollback migration (drop all tables)."""
    logger.warning("Rolling back migration - this will DROP all tables!")

    response = input("Are you sure? Type 'yes' to confirm: ")
    if response.lower() != 'yes':
        logger.info("Rollback cancelled")
        return

    try:
        # Drop all tables
        Base.metadata.drop_all(db_manager.engine)
        logger.info("All tables dropped successfully")

    except Exception as e:
        logger.error(f"Rollback failed: {e}", exc_info=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Database migration tool")
    parser.add_argument("--rollback", action="store_true", help="Rollback migration (drop tables)")
    parser.add_argument("--verify", action="store_true", help="Verify tables only")
    args = parser.parse_args()

    try:
        if args.rollback:
            rollback_migration()
        elif args.verify:
            if verify_tables():
                print("✓ All tables verified")
            else:
                print("✗ Table verification failed")
                sys.exit(1)
        else:
            success = run_migration()
            if not success:
                sys.exit(1)
    finally:
        db_manager.cleanup()