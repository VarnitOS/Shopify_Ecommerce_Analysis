from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from contextlib import contextmanager
from typing import Generator

from .config import get_config
from .logger import get_logger

logger = get_logger(__name__)
config = get_config()

# Get database configuration
DATABASE_URL = config.get('database.url')
POOL_SIZE = config.get('database.pool_size', 10)
MAX_OVERFLOW = config.get('database.max_overflow', 20)
POOL_TIMEOUT = config.get('database.timeout', 30)

# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    pool_size=POOL_SIZE,
    max_overflow=MAX_OVERFLOW,
    pool_timeout=POOL_TIMEOUT,
    pool_pre_ping=True,
    echo=config.get('app.debug', False)
)

# Create session factory
SessionFactory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
ScopedSession = scoped_session(SessionFactory)

# Create base model class
Base = declarative_base()


def get_db_session():
    """Get a database session.
    
    Returns:
        SQLAlchemy session
    """
    return ScopedSession()


@contextmanager
def db_session() -> Generator:
    """Context manager for database sessions.
    
    Yields:
        Database session
    
    Raises:
        Exception: Any exception that occurs during session use
    """
    session = get_db_session()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.exception(f"Database error: {str(e)}")
        raise
    finally:
        session.close()


def init_db():
    """Initialize the database by creating all tables."""
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")


def drop_db():
    """Drop all database tables."""
    logger.warning("Dropping all database tables...")
    Base.metadata.drop_all(bind=engine)
    logger.warning("Database tables dropped successfully") 