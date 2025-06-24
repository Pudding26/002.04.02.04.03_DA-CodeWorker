import os, io
import yaml
import pandas as pd
import logging
from memory_profiler import profile

from app.tasks.TaskBase import TaskBase
from app.utils.SQL.SQL_Df import SQL_Df


mem_Streams = {
    "step1": io.StringIO(),
    "step2": io.StringIO(),
    "step3": io.StringIO(),
}

class TA11_E_Import_SecondaryData(TaskBase):
    def setup(self):
        self.controller.update_message("🔧 Initializing Secondary Data Import Task")
        self.db_writer = SQL_Df(self.instructions["dest_db_name"])
        self.csv_files = {}
        self.delimiter_mapper = self.load_delimiter_mapper("config/mapper/DataImport/TA05_delimtermapper.yaml")

    def run(self):
        try:
            logging.info("📁 Starting CSV discovery...")
            self.controller.update_message("Step 1: Discover CSV files")
            self.csv_files = self.find_csv_files()

            self.controller.update_item_count(len(self.csv_files))
            self.controller.update_message("Step 2: Import CSVs to SQL")
            self.import_csvs_to_sql()

            self.controller.update_progress(1.0)
            self.controller.finalize_success()
            self.set_needs_running(False) #mark as already processed for the wrapper

            logging.info("✅ CSV import completed successfully.")
        except Exception as e:
            logging.exception("❌ Task failed")
            self.controller.finalize_failure(str(e))
            raise
        finally:
            self.cleanup()

    def cleanup(self):
        self.flush_memory_logs()
        self.controller.archive_with_orm()
    
    @profile(stream=mem_Streams["step1"])
    def find_csv_files(self):
        base_path = self.instructions["sourceFile_path"]
        csv_files = {}
        for root, _, files in os.walk(base_path):
            for file in files:
                if file.endswith(".csv"):
                    rel_path = os.path.relpath(os.path.join(root, file), base_path)
                    table_name = file.split(".")[0]
                    csv_files[table_name] = rel_path
        logging.info(f"🔍 Found {len(csv_files)} CSV files under: {base_path}")
        return csv_files

    @profile(stream=mem_Streams["step2"])
    def import_csvs_to_sql(self):
        base_path = self.instructions["sourceFile_path"]
        encodings = ['utf-8', 'ISO-8859-1', 'cp1252', 'utf-16']
        total = len(self.csv_files)

        for idx, (table_name, rel_path) in enumerate(self.csv_files.items()):
            self.check_control()
            full_path = os.path.join(base_path, rel_path)
            delimiter = self.delimiter_mapper.get(rel_path, ",")

            df = None
            for enc in encodings:
                try:
                    df = pd.read_csv(full_path, sep=delimiter, encoding=enc, engine='python', on_bad_lines='skip')
                    logging.debug2(f"📄 Successfully read {rel_path} with encoding {enc}")
                    break
                except UnicodeDecodeError:
                    logging.debug2(f"❌ Encoding {enc} failed for {rel_path}, trying next...")
                    continue
                except Exception as e:
                    logging.warning(f"⚠️ Error reading {full_path} with encoding {enc}: {e}")
                    break

            if df is None:
                logging.error(f"❌ Could not decode {full_path} with any encoding.")
                continue

            try:
                self.db_writer.store(table_name, df, method="replace")
                logging.debug1(f"✅ Stored {table_name} from {rel_path} to SQL (replace)")
            except Exception as e:
                logging.error(f"❌ Error writing table '{table_name}' from file '{rel_path}': {e}")

            progress = (idx + 1) / total
            self.controller.update_progress(progress)

    def load_delimiter_mapper(self, path):
        try:
            with open(path, "r") as f:
                mapping = yaml.safe_load(f)
                logging.debug2(f"🗺️ Loaded delimiter mapping from {path}")
                return mapping
        except Exception as e:
            logging.warning(f"⚠️ Could not load delimiter mapper from {path}: {e}")
            return {}
