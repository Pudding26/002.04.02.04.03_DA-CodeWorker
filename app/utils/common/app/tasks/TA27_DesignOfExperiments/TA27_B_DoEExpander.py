import pandas as pd
import hashlib
import itertools
import logging

logger = logging.getLogger(__name__)

class TA27_B_DoEExpander:
    @staticmethod
    def expand(yaml_dict: dict) -> pd.DataFrame:
        logger.debug3("🔍 Starting DoE expansion...")

        if not yaml_dict:
            logger.warning("⚠️ Empty YAML dictionary provided. Returning empty DataFrame.")
            return pd.DataFrame()

        primary = yaml_dict.get("primary_data", {})
        other = {k: v for k, v in yaml_dict.items() if k != "primary_data"}

        if not primary:
            logger.warning("⚠️ No 'primary_data' section found in YAML. Expansion may be trivial.")

        primary_items = [(k, v if isinstance(v, list) else [v]) for k, v in primary.items()]
        primary_keys, primary_values = zip(*primary_items) if primary_items else ([], [])
        primary_combos = list(itertools.product(*primary_values)) if primary_values else [()]

        other_keys = list(other.keys())
        other_values = list(itertools.product(*[other[k] if isinstance(other[k], list) else [other[k]] for k in other_keys])) if other_keys else [()]

        logger.debug3(f"📊 Primary keys: {primary_keys}, combinations: {len(primary_combos)}")
        logger.debug3(f"📊 Other keys: {other_keys}, combinations: {len(other_values)}")

        combined_rows = []
        for p_combo in primary_combos:
            for o_combo in other_values:
                row = dict(zip(list(primary_keys) + other_keys, list(p_combo) + list(o_combo)))
                row = {k: v if isinstance(v, list) else [v] for k, v in row.items()}
                base_str = str(sorted(row.items()))
                row["DoE_UUID"] = "DoE_" + hashlib.sha1(base_str.encode()).hexdigest()[:10]
                combined_rows.append(row)

        df = pd.DataFrame(combined_rows)
        logger.info(f"✅ Expanded DoE to {len(df)} combinations with {len(df.columns)} columns.")
        return df
