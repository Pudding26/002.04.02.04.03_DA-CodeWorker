import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import logging

load_dotenv()  # Load environment variables from .env


class TA01_A_SQLSetup:
    """
    Wrapper for setup and administrative operations on PostgreSQL databases.
    Reads configuration from environment variables.
    """
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    default_db = "postgres"

    connection_url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{default_db}"
    engine = create_engine(connection_url, isolation_level="AUTOCOMMIT")

    @classmethod
    def database_exists(cls, db_name: str) -> bool:
        """Check if a database with the given name exists."""
        query = text("SELECT 1 FROM pg_database WHERE datname = :db")
        with cls.engine.connect() as conn:
            result = conn.execute(query, {"db": db_name})
            return result.scalar() is not None

    @classmethod
    def create_database(cls, db_name: str) -> None:
        """Create the database if it does not already exist."""
        if cls.database_exists(db_name):
            logging.debug1(f"✅ Database '{db_name}' already exists.")
        else:
            try:
                with cls.engine.connect() as conn:
                    conn.execute(text(f"CREATE DATABASE \"{db_name}\""))
                    logging.info(f"🆕 Database '{db_name}' created successfully.")
            except SQLAlchemyError as e:
                logging.error(f"❌ Failed to create database '{db_name}': {e}", exc_info=True)

    @classmethod
    def createDatabases(cls) -> None:
        """Check and create all required databases defined by environment variables."""
        db_env_keys = [
            "DB_SOURCE_NAME",
            "DB_RAW_NAME",
            "DB_PRODUCTION_NAME",
            "DB_PROGRESS_NAME",
            "DB_TEMP_NAME"
        ]

        for key in db_env_keys:
            db_name = os.getenv(key)
            if db_name:
                cls.create_database(db_name)
            else:
                logging.warning(f"⚠️ Environment variable '{key}' is not set — skipping.")

    # Additional utility methods can be added below as needed

    @classmethod
    def test_connection(cls) -> bool:
        """Test if the connection to the default DB is working."""
        try:
            with cls.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logging.debug1("✅ Database connection successful.")
            return True
        except Exception as e:
            logging.error(f"❌ Database connection failed: {e}", exc_info=True)
            return False
