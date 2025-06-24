import os
import yaml
import logging
import pandas as pd
from pathlib import Path
from typing import List, Union

class YamlColumnMapper:
    """Handles column renaming and static column injection from YAML files."""

    @staticmethod
    def rename_columns(df: pd.DataFrame, yaml_path: str, keys_list=None) -> pd.DataFrame:
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)

        section = yaml_data
        if keys_list:
            for key in keys_list:
                section = section.get(key, {})

        if not isinstance(section, dict):
            raise ValueError(f"YAML section at {' -> '.join(keys_list)} is not a dictionary.")

        return df.rename(columns=section)

    @staticmethod
    def add_static_columns(df: pd.DataFrame, yaml_path: str, keys_list: list) -> pd.DataFrame:
        logger = logging.getLogger(__name__)
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)

            section = yaml_data
            for key in keys_list:
                section = section.get(key, {})

            if not isinstance(section, dict):
                raise ValueError(f"Expected dictionary at {' -> '.join(keys_list)}, got: {type(section)}")

            for col, val in section.items():
                df[col] = val

            logger.info(f"✅ Injected {len(section)} static columns.")
            return df

        except Exception as e:
            logger.warning(f"YAML column injection failed: {e}")
            return df

    @staticmethod
    def update_column_mapping(df: pd.DataFrame, yaml_path: str, keys_list: list, default_value="TODO"):
        logger = logging.getLogger(__name__)
        yaml_data = {}

        if os.path.exists(yaml_path):
            with open(yaml_path, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f) or {}

        current_section = yaml_data
        for key in keys_list:
            current_section.setdefault(key, {})
            current_section = current_section[key]

        added = 0
        for col in df.columns:
            if col not in current_section:
                current_section[col] = default_value
                added += 1

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_data, f, sort_keys=False, allow_unicode=True)

        logger.debug(f"📝 Updated YAML with {added} new column mappings under {' -> '.join(keys_list)}.")



    @staticmethod
    def yaml_col_value_mapper(yaml_path: str, data_source_key: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies value mappings from a YAML configuration to the given DataFrame.

        Args:
            yaml_path (str): Path to the unified YAML file.
            data_source_key (str): Key to identify the data source in the YAML file (e.g., 'DS01').
            df (pd.DataFrame): DataFrame to process.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        logger = logging.getLogger(__name__)
        logger.debug2(f"🔍 Loading value mappings from: {yaml_path}")
        yaml_file = Path(yaml_path)

        if not yaml_file.exists():
            raise FileNotFoundError(f"YAML file does not exist: {yaml_path}")

        with yaml_file.open("r") as file:
            try:
                mappings = yaml.safe_load(file)
                logger.debug2(f"✅ YAML parsed successfully. Top-level keys: {list(mappings.keys())}")
            except yaml.YAMLError as e:
                logger.error(f"❌ Failed to parse YAML: {e}")
                raise

        if data_source_key not in mappings:
            logger.warning(f"⚠️ No mapping found for key: {data_source_key}")
            return df

        source_mappings = mappings[data_source_key]

        # Normalize to list of mappings if necessary
        if isinstance(source_mappings, dict) and "col" in source_mappings:
            source_mappings = [source_mappings]
        elif isinstance(source_mappings, dict):
            # Handle dict-of-dicts
            source_mappings = [v for k, v in source_mappings.items() if isinstance(v, dict) and "col" in v]

        for mapping in source_mappings:
            col = mapping.get("col")
            col_old = mapping.get("col_old")
            values = mapping.get("values", {})

            if col not in df.columns:
                logger.debug2(f"⏭️ Column '{col}' not found in DataFrame. Skipping.")
                continue

            logger.debug2(f"🔁 Processing mapping for column: {col}")
            if col_old:
                df[col_old] = df[col].copy()
                logger.debug2(f"💾 Backed up original column to: {col_old}")

            df[col] = df[col].map(values).fillna(df[col])  # fallback to original if no mapping

            # Auto rename todo_ columns
            if col.startswith("todo_"):
                new_col = col.replace("todo_", "")
                df.rename(columns={col: new_col}, inplace=True)
                logger.debug2(f"🔤 Renamed column '{col}' → '{new_col}'")

        logger.debug2(f"✅ Value mapping applied successfully for key: {data_source_key}")
        return df
    

    @staticmethod
    def map_columns_from_yaml(
        df: pd.DataFrame,
        new_cols: List[str],
        source_cols: List[str],
        yaml_paths: List[Union[str, List[str]]],
        yaml_file_path: str,
        overwrite: bool = False
    ) -> pd.DataFrame:
        """
        Maps values from YAML to DataFrame columns, with optional overwrite behavior.
        Logs a summary of updates, including sample values.

        Args:
            df: The input DataFrame.
            new_cols: List of target columns where mapped values will be stored.
            source_cols: List of source columns in df whose values will be used as mapping keys.
            yaml_paths: List of paths (dot-separated strings or lists) to key-value mappings in the YAML.
            yaml_file_path: Path to the YAML file.
            overwrite: If True, overwrite existing values. If False, only fill where value is missing.

        Returns:
            A new DataFrame with mapped values.
        """
        if not (len(new_cols) == len(source_cols) == len(yaml_paths)):
            raise ValueError("new_cols, source_cols, and yaml_paths must be of the same length")

        with open(yaml_file_path, 'r') as f:
            yaml_data = yaml.safe_load(f)

        df_mapped = df.copy()

        logging.debug2("\n🧾 YAML Mapping Summary:")
        for target_col, source_col, path in zip(new_cols, source_cols, yaml_paths):
            if source_col not in df_mapped.columns:
                raise ValueError(f"Source column '{source_col}' not found in DataFrame")

            # Resolve mapping from YAML
            keys = path if isinstance(path, list) else path.split(".")
            mapping = yaml_data
            try:
                for key in keys:
                    mapping = mapping[key]
            except KeyError:
                raise ValueError(f"Invalid path '{path}' in YAML file")

            # Perform the mapping
            mapped_series = df_mapped[source_col].map(mapping)

            if target_col not in df_mapped.columns:
                df_mapped[target_col] = pd.NA

            before_series = df_mapped[target_col].copy()

            if overwrite:
                df_mapped[target_col] = mapped_series
            else:
                mask = df_mapped[target_col].isna() & mapped_series.notna()
                df_mapped.loc[mask, target_col] = mapped_series[mask]

            after_series = df_mapped[target_col]
            changes = (before_series != after_series) & after_series.notna()
            num_updated = changes.sum()

            # Get only the *newly written* values
            new_values = after_series[changes].dropna().unique()
            distinct_vals = len(new_values)
            sample_values = new_values[:3]
            sample_str = ", ".join(map(str, sample_values)) if sample_values.size > 0 else "None"


            # Show up to 3 example mapped values (non-null)
            sample_values = df_mapped.loc[changes, target_col].dropna().unique()[:3]
            sample_str = ", ".join(str(v) for v in sample_values)

            logging.debug2(
                f"  ➤ '{target_col}': {num_updated} values {'overwritten' if overwrite else 'filled'}, "
                f"{distinct_vals} distinct. Examples: [{sample_str}]"
            )

        logging.debug3("✅ Mapping complete.\n")
        return df_mapped

