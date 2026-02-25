"""
Database Connection Manager
============================

Handles database connections with pooling, retries, and health checks.
Supports multiple database backends (PostgreSQL, MySQL, SQLite).
"""

import os
import logging
from typing import Optional, Dict, Any, Generator
from contextlib import contextmanager
from datetime import datetime
import time

from sqlalchemy import create_engine, event, pool, text
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.exc import SQLAlchemyError, OperationalError, DisconnectionError
from sqlalchemy.pool import NullPool, QueuePool

from .models import Base

logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Database configuration settings"""

    def __init__(
        self,
        database_url: Optional[str] = None,
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
        echo: bool = False,
        use_pool: bool = True,
        retry_on_disconnect: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize database configuration.

        Args:
            database_url: Database connection URL
            pool_size: Number of connections to maintain in pool
            max_overflow: Maximum overflow connections allowed
            pool_timeout: Timeout for getting connection from pool
            pool_recycle: Recycle connections after this many seconds
            echo: Enable SQL logging
            use_pool: Use connection pooling (False for SQLite)
            retry_on_disconnect: Retry on connection errors
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.database_url = database_url or os.getenv(
            'DATABASE_URL',
            'sqlite:///trans_production.db'  # Default to SQLite for development
        )
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.echo = echo
        self.use_pool = use_pool
        self.retry_on_disconnect = retry_on_disconnect
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Determine database type
        self.db_type = self._get_db_type()

    def _get_db_type(self) -> str:
        """Determine database type from URL"""
        if 'postgresql' in self.database_url:
            return 'postgresql'
        elif 'mysql' in self.database_url:
            return 'mysql'
        elif 'sqlite' in self.database_url:
            return 'sqlite'
        else:
            return 'unknown'


class DatabaseManager:
    """
    Manages database connections with pooling and error handling.

    Features:
    - Connection pooling with configurable parameters
    - Automatic reconnection on connection errors
    - Health checks and connection validation
    - Transaction management with rollback on errors
    - Thread-safe session management
    """

    def __init__(self, config: Optional[DatabaseConfig] = None):
        """
        Initialize database manager.

        Args:
            config: Database configuration (uses defaults if None)
        """
        self.config = config or DatabaseConfig()
        self.engine = None
        self.session_factory = None
        self.scoped_session = None
        self._initialized = False

        # Performance tracking
        self.connection_count = 0
        self.error_count = 0
        self.last_error = None

    def initialize(self):
        """Initialize database engine and session factory"""
        if self._initialized:
            return

        try:
            # Create engine with appropriate pooling strategy
            engine_args = {
                'echo': self.config.echo,
                'future': True,  # Use SQLAlchemy 2.0 style
            }

            if self.config.db_type == 'sqlite':
                # SQLite doesn't support connection pooling well
                engine_args['poolclass'] = NullPool
                engine_args['connect_args'] = {'check_same_thread': False}
            elif self.config.use_pool:
                # Use QueuePool for other databases
                engine_args['poolclass'] = QueuePool
                engine_args['pool_size'] = self.config.pool_size
                engine_args['max_overflow'] = self.config.max_overflow
                engine_args['pool_timeout'] = self.config.pool_timeout
                engine_args['pool_recycle'] = self.config.pool_recycle
                engine_args['pool_pre_ping'] = True  # Verify connections

            self.engine = create_engine(self.config.database_url, **engine_args)

            # Set up event listeners
            self._setup_event_listeners()

            # Create session factory
            self.session_factory = sessionmaker(
                bind=self.engine,
                expire_on_commit=False,
                autoflush=False
            )

            # Create scoped session for thread safety
            self.scoped_session = scoped_session(self.session_factory)

            # Create tables if they don't exist
            Base.metadata.create_all(self.engine)

            self._initialized = True
            logger.info(f"Database initialized: {self.config.db_type}")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def _setup_event_listeners(self):
        """Set up SQLAlchemy event listeners for connection management"""

        @event.listens_for(self.engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            """Called when a new connection is created"""
            self.connection_count += 1
            logger.debug(f"New database connection created (total: {self.connection_count})")

        @event.listens_for(self.engine, "checkout")
        def receive_checkout(dbapi_conn, connection_record, connection_proxy):
            """Called when a connection is checked out from the pool"""
            # Perform connection validation for non-SQLite databases
            if self.config.db_type != 'sqlite':
                try:
                    # Simple ping to verify connection
                    cursor = dbapi_conn.cursor()
                    cursor.execute("SELECT 1")
                    cursor.close()
                except Exception:
                    # Connection is dead, raise DisconnectionError to trigger reconnect
                    raise DisconnectionError("Connection failed validation check")

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Get a database session with automatic cleanup.

        Yields:
            Database session

        Example:
            with db_manager.get_session() as session:
                patterns = session.query(Pattern).all()
        """
        if not self._initialized:
            self.initialize()

        session = self.scoped_session()

        try:
            yield session
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            self.error_count += 1
            self.last_error = str(e)
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()

    def execute_with_retry(
        self,
        func,
        *args,
        max_retries: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        Execute a function with retry logic for connection errors.

        Args:
            func: Function to execute
            max_retries: Override default max retries
            *args: Arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Function result

        Raises:
            Last exception if all retries fail
        """
        max_retries = max_retries or self.config.max_retries
        last_exception = None

        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)

            except (OperationalError, DisconnectionError) as e:
                last_exception = e
                if attempt < max_retries - 1:
                    delay = self.config.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"Database connection error (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {delay}s: {e}"
                    )
                    time.sleep(delay)

                    # Reset connection pool on connection errors
                    if self.engine:
                        self.engine.dispose()
                else:
                    logger.error(f"All retry attempts failed: {e}")

            except Exception as e:
                # Non-connection errors, don't retry
                logger.error(f"Non-retryable database error: {e}")
                raise

        raise last_exception

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on database connection.

        Returns:
            Dictionary with health status information
        """
        health = {
            'status': 'unknown',
            'database_type': self.config.db_type,
            'connection_count': self.connection_count,
            'error_count': self.error_count,
            'last_error': self.last_error,
            'timestamp': datetime.utcnow().isoformat()
        }

        try:
            with self.get_session() as session:
                # Try a simple query
                result = session.execute(text("SELECT 1")).scalar()
                if result == 1:
                    health['status'] = 'healthy'

                    # Get additional metrics for non-SQLite databases
                    if self.config.db_type == 'postgresql':
                        # Get PostgreSQL specific metrics
                        conn_info = session.execute(
                            text("SELECT count(*) FROM pg_stat_activity")
                        ).scalar()
                        health['active_connections'] = conn_info

                    elif self.config.db_type == 'mysql':
                        # Get MySQL specific metrics
                        conn_info = session.execute(
                            text("SHOW STATUS LIKE 'Threads_connected'")
                        ).first()
                        if conn_info:
                            health['active_connections'] = int(conn_info[1])

                    # Get table counts
                    from .models import Pattern, Prediction, ModelVersion
                    health['pattern_count'] = session.query(Pattern).count()
                    health['prediction_count'] = session.query(Prediction).count()
                    health['model_count'] = session.query(ModelVersion).count()

        except Exception as e:
            health['status'] = 'unhealthy'
            health['error'] = str(e)
            logger.error(f"Health check failed: {e}")

        return health

    def cleanup(self):
        """Clean up database connections and resources"""
        if self.scoped_session:
            self.scoped_session.remove()

        if self.engine:
            self.engine.dispose()
            logger.info("Database connections closed")

        self._initialized = False

    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()


# Global database manager instance
db_manager = DatabaseManager()


# Convenience functions
@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Get a database session.

    Yields:
        Database session

    Example:
        with get_db_session() as session:
            patterns = session.query(Pattern).all()
    """
    with db_manager.get_session() as session:
        yield session


def init_database(config: Optional[DatabaseConfig] = None):
    """
    Initialize the database with given configuration.

    Args:
        config: Database configuration
    """
    global db_manager
    if config:
        db_manager = DatabaseManager(config)
    db_manager.initialize()


def close_database():
    """Close database connections"""
    db_manager.cleanup()


if __name__ == "__main__":
    # Test database connection
    logging.basicConfig(level=logging.INFO)

    # Initialize with default configuration
    init_database()

    # Perform health check
    health = db_manager.health_check()
    print(f"Database health: {health}")

    # Test session
    with get_db_session() as session:
        print("Database session created successfully")

    # Cleanup
    close_database()