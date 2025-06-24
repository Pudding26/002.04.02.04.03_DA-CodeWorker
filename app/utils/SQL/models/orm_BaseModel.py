# ORM Base with shared method for DataFrame persistence

from sqlalchemy.orm import declarative_base, Session
import pandas as pd
import logging


class _Base:
    @classmethod
    def store_dataframe(cls, df: pd.DataFrame, db_key: str, method: str = "append"):
        from app.utils.SQL.DBEngine import DBEngine
        session: Session = DBEngine(db_key).get_session()
        try:
            with session.begin():
                if method == "replace":
                    session.query(cls).delete()
                    session.flush()
                    logging.debug2(f"🧹 Cleared existing rows from {cls.__tablename__}")

                # ✅ Fast: vectorized conversion
                records = df.dropna(how="all").to_dict(orient="records")
                objs = [cls(**r) for r in records]
                session.bulk_save_objects(objs)
                logging.info(f"✅ Stored {len(objs)} rows to {cls.__tablename__} using ORM ({method})")

        except Exception as e:
            logging.error(f"❌ Failed to store to {cls.__tablename__}: {e}", exc_info=True)
            raise
        finally:
            session.close()

    @classmethod
    def store_dataframe_debug(cls, df: pd.DataFrame, db_key: str, method: str = "append"):
        from app.utils.SQL.DBEngine import DBEngine
        session: Session = DBEngine(db_key).get_session()
        try:
            with session.begin():
                if method == "replace":
                    session.query(cls).delete()
                    session.flush()
                    logging.debug2(f"🧹 Cleared existing rows from {cls.__tablename__}")

                # ✅ Fast: vectorized conversion
                records = df.dropna(how="all").to_dict(orient="records")
                objs = [cls(**r) for r in records]
                for i, obj in enumerate(objs):
                    try:
                        session.add(obj)
                        session.flush()  # Try to insert into DB
                    except Exception as e:
                        print(f"❌ Insert failed at row {i}")
                        print(obj.__dict__)
                        raise e
                logging.info(f"✅ Stored {len(objs)} rows to {cls.__tablename__} using ORM ({method})")

        except Exception as e:
            logging.error(f"❌ Failed to store to {cls.__tablename__}: {e}", exc_info=True)
            raise
        finally:
            session.close()






# Base class for all ORM models
orm_BaseModel = declarative_base(cls=_Base)
