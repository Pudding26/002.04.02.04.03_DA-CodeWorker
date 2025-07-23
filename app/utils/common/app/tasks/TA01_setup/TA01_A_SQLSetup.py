import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import inspect


import logging
from app.utils.common.app.utils.SQL.DBEngine import DBEngine

# SQL Events
#from app.utils.common.app.utils.SQL.events.events_WorkerJobs import (
#    sync_workerjob_links,
#    _roll_up_child_status
#)


load_dotenv()  # Load environment variables from .env


class TA01_A_SQLSetup:
    """
    Wrapper for setup and administrative operations on PostgreSQL databases.
    Reads configuration from environment variables.
    """
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOSTNAME_PUBLIC")
    port = os.getenv("DB_PORT_PUBLIC")
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
        from app.utils.common.app.utils.SQL.models.orm_BaseModel import orm_BaseModel

        """Check and create all required databases defined by environment variables."""
        db_env_keys = [
            "DB_SOURCE_NAME",
            "DB_RAW_NAME",
            "DB_PRODUCTION_NAME",
            "DB_PROGRESS_NAME",
            "DB_TEMP_NAME",
            "DB_JOBS_NAME",
            "DB_DEBUG_NAME",
        ]


        for key in db_env_keys:
            db_name = os.getenv(key)
            if db_name:
                cls.create_database(db_name)
            else:
                logging.warning(f"⚠️ Environment variable '{key}' is not set — skipping.")

    @staticmethod
    def create_all_tables():
        
        #progress
        from app.utils.common.app.utils.SQL.models.progress.orm.ProfileArchive import ProfileArchive
        from app.utils.common.app.utils.SQL.models.progress.orm.ProgressArchive import ProgressArchive
        
        # raw
        from app.utils.common.app.utils.SQL.models.raw.orm.PrimaryDataRaw import PrimaryDataRaw
        
        # production
        from app.utils.common.app.utils.SQL.models.production.orm.DS09 import DS09
        from app.utils.common.app.utils.SQL.models.production.orm.DS40 import DS40
        from app.utils.common.app.utils.SQL.models.production.orm.DS12 import DS12
        from app.utils.common.app.utils.SQL.models.production.orm.WoodTableA import WoodTableA
        from app.utils.common.app.utils.SQL.models.production.orm.WoodTableB import WoodTableB
        from app.utils.common.app.utils.SQL.models.production.orm.WoodMaster import WoodMaster
        from app.utils.common.app.utils.SQL.models.production.orm.WoodMasterPotential import WoodMasterPotential
        from app.utils.common.app.utils.SQL.models.production.orm.DoEArchive import DoEArchive
        from app.utils.common.app.utils.SQL.models.production.orm.ModellingResults import ModellingResults



        # jobs
        from app.utils.common.app.utils.SQL.models.jobs.orm_DoEJobs import orm_DoEJobs
        from app.utils.common.app.utils.SQL.models.jobs.orm_WorkerJobs import orm_WorkerJobs
        from app.utils.common.app.utils.SQL.models.jobs.orm_JobLink import orm_JobLink
    

        grouped_models = {
            "progress": [ProfileArchive, ProgressArchive],
            "raw": [PrimaryDataRaw],
            "production": [WoodMaster, WoodMasterPotential, WoodTableA, WoodTableB, DoEArchive, DS09, DS12, DS40, ModellingResults],
            "jobs" : [orm_DoEJobs, orm_WorkerJobs, orm_JobLink],
        }




        for db_key, model_list in grouped_models.items():
            try:
                db = DBEngine(db_key)
                engine = db.get_engine()
                for model in model_list:
                    model.__table__.create(bind=engine, checkfirst=True)
                    logging.debug2(f"✅ Created table '{model.__tablename__}' in database '{db_key}'.")

                    meta_cols = set(model.__table__.columns.keys())
                    
                    try:
                        real_cols = set(col['name'] for col in inspect(engine).get_columns(model.__tablename__))
                    except Exception as e:
                        logging.warning(f"⚠️ Could not inspect table '{model.__tablename__}': {e}")

                    if meta_cols - real_cols:
                        logging.warning(f"⚠️ Column mismatch in table '{model.__tablename__}': in metadata but missing in DB: {meta_cols - real_cols}")
                    if real_cols - meta_cols:
                        logging.warning(f"⚠️ Extra columns in DB for table '{model.__tablename__}': {real_cols - meta_cols}")





            except Exception as e:
                logging.warning(f"❌ Failed to create tables for {db_key}: {e}")


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
