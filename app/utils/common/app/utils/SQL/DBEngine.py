import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.utils.common.app.utils.SQL.models.orm_BaseModel import orm_BaseModel

from sqlalchemy.schema import CreateTable

import logging
_engine_cache = {}  # Module-level cache, To allow to reuse engine connections

load_dotenv()

class DBEngine:
    def __init__(self, db_key: str):
        """
        db_key: The environment key like 'source_db', 'raw_db', etc.
        """
        self.db_key = db_key
        self.database_url = self._build_pg_url(db_key)
        
        if not self.database_url:
            raise ValueError(f"❌ No DB URL found in .env for key: {db_key}")
        
        # Reuse engine from cache if it exists
        if db_key not in _engine_cache:
            _engine_cache[db_key] = create_engine(self.database_url)
            logging.debug3(f"✅ Created new SQLAlchemy engine for {db_key} at {self.database_url}")
        else:
            logging.debug1(f"🔄 Reusing existing SQLAlchemy engine for {db_key} at {self.database_url}")
        
        self.engine = _engine_cache[db_key]
        self.SessionLocal = sessionmaker(bind=self.engine)


    def get_session(self):
        """Creates a new SQLAlchemy session bound to the engine."""
        return self.SessionLocal()

    def get_engine(self):
        """Returns the raw SQLAlchemy engine (for pandas or raw SQL)."""
        return self.engine

    def test_connection(self):
        """Simple test query to confirm DB connection is working."""
        with self.get_engine().connect() as conn:
            conn.execute("SELECT 1")

    def _build_pg_url(self, db_key: str) -> str:
        user = os.getenv("DB_USER")
        pwd = os.getenv("DB_PASSWORD")
        host = os.getenv("DB_HOST")
        port = os.getenv("DB_PORT")
        db = os.getenv(f"DB_{db_key.upper()}_NAME")  # e.g. DB_SOURCE_NAME

        if not all([user, pwd, host, port, db]):
            raise ValueError(f"❌ Missing env vars for {db_key} DB")

        return f"postgresql://{user}:{pwd}@{host}:{port}/{db}"
    




