import re
import pandas as pd
import logging
import os
from sqlalchemy.orm import Session

from app.tasks.TaskBase import TaskBase
from app.utils.common.app.utils.controlling.TaskController import TaskController
from app.utils.common.app.utils.SQL.DBEngine import DBEngine
from app.utils.common.app.utils.mapping.YamlColumnMapper import YamlColumnMapper
from app.utils.common.app.utils.SQL.models.production.orm.DS09 import DS09
from app.utils.common.app.utils.SQL.SQL_Df import SQL_Df

if os.getenv("DEBUG_MODE") == "True":
    import memory_profiler
    memory_profiler.profile.disable = lambda: None

class TA13_A_DataTransfer_DS09(TaskBase):
    def setup(self):
        self.db_session: Session = DBEngine(self.instructions["dest_db_name"]).get_session()
        self.src_db = SQL_Df(self.instructions["src_db_name"])
        self.table_name = self.instructions["table_name"]
        self.dataset_name = self.instructions["taskName"]
        self.data_raw = None
        self.data_cleaned = None

        logging.debug2(f"[{self.dataset_name}] 🔧 Task setup initialized.")
        self.controller.update_message(f"Initialized {self.dataset_name}")

    def run(self):
        try:
            logging.info(f"[{self.dataset_name}] 🚀 Starting data transfer process.")
            self.controller.update_message("Loading data from source DB...")
            self.data_raw = self.src_db.load(self.table_name)
            logging.debug1(f"[{self.dataset_name}] Loaded {len(self.data_raw)} rows from '{self.table_name}'.")
            self.controller.update_progress(0.1)

            self.rename_columns()
            self.controller.update_progress(0.2)
            self.check_control()

            self.annotate_ifaw_state()
            self.controller.update_progress(0.3)
            self.check_control()

            self.split_data_based_on_ifaw_state()
            self.controller.update_progress(0.5)

            self.addgenus()
            self.controller.update_progress(0.7)

            self.finalize_data()
            self.controller.update_progress(0.8)

            self.controller.update_message("Storing cleaned data in destination DB using ORM...")
            self.persist_cleaned_data()
            self.controller.update_progress(1.0)

            self.controller.finalize_success()
            self.set_needs_running(False) #mark as already processed for the wrapper

            logging.info(f"[{self.dataset_name}] 🎉 Task completed successfully.")

        except Exception as e:
            self.controller.finalize_failure(str(e))
            logging.error(f"[{self.dataset_name}] ❌ Task failed: {e}", exc_info=True)
            raise

    def cleanup(self):
        logging.debug2(f"[{self.dataset_name}] 🧹 Running cleanup.")
        self.controller.archive_with_orm()
        self.db_session.close()

    def rename_columns(self):
        logging.debug2(f"[{self.dataset_name}] 🔤 Renaming columns using YamlColumnMapper.")
        YamlColumnMapper.update_column_mapping(
            df=self.data_raw,
            yaml_path=self.instructions["path_col_name_mapper"],
            keys_list=[self.dataset_name, "rename"]
        )
        self.data_raw = YamlColumnMapper.rename_columns(
            df=self.data_raw,
            yaml_path=self.instructions["path_col_name_mapper"],
            keys_list=[self.dataset_name, "rename"]
        )
        logging.debug2(f"[{self.dataset_name}] 🔤 Columns renamed: {self.data_raw.columns.tolist()}")

    def annotate_ifaw_state(self):
        def classify_code(code: str) -> int | None:
            if isinstance(code, str):
                code = code.strip().upper()
                if code.endswith("XX"):
                    return 2
                elif len(code) == 4 and "X" not in code[2:4]:
                    return 1
                else:
                    return 3
        self.data_raw["ifaw_state_temp"] = self.data_raw["IFAW_code"].apply(classify_code)

    def split_data_based_on_ifaw_state(self):
        data_case_1 = self.handle_case_1(self.data_raw[self.data_raw["ifaw_state_temp"] == 1])
        data_case_2 = self.handle_case_2(self.data_raw[self.data_raw["ifaw_state_temp"] == 2])
        data_case_3 = self.handle_case_3(self.data_raw[self.data_raw["ifaw_state_temp"] == 3])
        self.data_cleaned = pd.concat([data_case_1, data_case_2, data_case_3], ignore_index=True)

    def handle_case_1(self, df):
        df = df.copy()
        df["species"] = df["species_todo"].apply(self.to_upper_camel_case)
        df = df[~df["species_todo"].str.contains("spp", case=False, na=False)]
        return df

    def handle_case_2(self, df):
        df = df.copy()
        df["genus"] = df["species_todo"].apply(self.extract_genus)
        return df

    def handle_case_3(self, df):
        manual_matches = {
            "ATTX": ["AntiarisToxicaria"],
            "BLTX": ["BaillonellaToxisperma"],
            "BRXB": ["BrachystegiaLaurentii"],
            "BRXN": ["BrachystegiaCynometrides", "BrachystegiaEurocyma", "BrachystegiaLeonensis", "BrachystegiaNigerica"],
            "COXA": ["CordiaAlliodora", "CordiaTrichotoma"],
            "COXB": ["CordiaAbyssinica", "CordiaMillenii", "CordiaPlatythyrsa"],
            "DEEX": ["DiniziaExcelsa"],
            "FXEX": ["FraxinusExcelsior"],
            "HEXN": ["HeritieraUtilis", "HeritieraDensiflora"],
            "OXOX": ["OxystigmaOxyphyllum"],
            "PLXH": ["PlatanusHispanica"],
            "QCXE": ["QuercusPetraea", "QuercusRobur"],
            "QCXA": ["QuercusAlba"],
            "QCXR": ["QuercusRubra"],
            "QCXJ": ["QuercusMongolica"],
            "ULXH": ["UlmusHollandica"]
        }
        flat = [(code, sp) for code, lst in manual_matches.items() for sp in lst]
        match_df = pd.DataFrame(flat, columns=["IFAW_code", "species"])
        return match_df.merge(df, how="left", on="IFAW_code")

    def addgenus(self):
        data_gen = self.data_cleaned
        mask = data_gen["genus"].isna() & data_gen["species"].notna()
        data_gen.loc[mask, "genus"] = data_gen.loc[mask, "species"].apply(
            lambda x: re.findall(r'[A-Z][a-z]*', x)[0] if re.findall(r'[A-Z][a-z]*', x) else None
        )
        self.data_cleaned = data_gen

    def extract_genus(self, species_str):
        try:
            if isinstance(species_str, str):
                parts = species_str.strip().split()
                if parts:
                    return parts[0].capitalize()
        except Exception as e:
            logging.warning(f"extract_genus failed: {e}")
        return None


    def to_upper_camel_case(self, species_str):
        try:
            if isinstance(species_str, str):
                parts = species_str.strip().split()
                if len(parts) >= 2:
                    return parts[0].capitalize() + parts[1].capitalize()
        except Exception as e:
            logging.warning(f"to_upper_camel_case failed: {e}")
        return None



    def finalize_data(self):
        self.data_cleaned.rename(columns={"species_todo": "species_drop"}, inplace=True)
        self.data_cleaned.drop(
            columns=[col for col in self.data_cleaned.columns if col.endswith("drop") or col.endswith("temp")],
            inplace=True
        )

    def persist_cleaned_data(self):
        
        DS09.store_dataframe(self.data_cleaned, db_key="production", method="replace")

