import pandas as pd
import logging
import hashlib
from typing import Type, Optional, Any, List, TypeVar
import string

def split_df_based_on_max_split(df: pd.DataFrame, column: str, separator: str = '_') -> dict:
    """
    Splits a DataFrame into multiple sub-DataFrames based on the number of parts after splitting a target column.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - column (str): Column to split by.
    - separator (str): Delimiter used to split the column string.

    Returns:
    - dict[int, pd.DataFrame]: Dictionary where keys are split lengths, and values are DataFrames.
    """
    split_dict = {}
    try:
        for idx, row in df.iterrows():
            val = row[column]
            if not isinstance(val, str):
                continue
            parts = val.split(separator)
            max_split = len(parts)
            row_data = row.copy()
            row_data["max_split"] = max_split
            for i, part in enumerate(parts):
                row_data[f"col-{i}"] = part

            if max_split not in split_dict:
                split_dict[max_split] = []
            split_dict[max_split].append(row_data)

        # Convert lists to DataFrames
        for key in split_dict:
            split_dict[key] = pd.DataFrame(split_dict[key])

        logging.debug1(f"🔍 Split column '{column}' into {len(split_dict)} groups by max_split.")
        return split_dict

    except Exception as e:
        logging.error(f"❌ Error splitting DataFrame on column '{column}': {e}")
        raise



def generate_deterministic_string_uuid(
        series: pd.Series, length: int = 6
    ) -> pd.Series:
        """
        Generate a deterministic short UUID for each row in a pandas Series using SHA256 + Base62.

        Args:
            series (pd.Series): Input column to hash.
            length (int): Length of the resulting string UUID (default: 6).

        Returns:
            pd.Series: New Series with deterministic string-based UUIDs.
        """

        BASE62 = string.digits + string.ascii_letters  # 0-9, A-Z, a-z
        base = len(BASE62)

        def int_to_base62(n: int) -> str:
            if n == 0:
                return BASE62[0] * length
            chars = []
            while n > 0:
                n, rem = divmod(n, base)
                chars.append(BASE62[rem])
            return ''.join(reversed(chars)).rjust(length, BASE62[0])[:length]

        def hash_to_base62(x: str) -> str:
            h = hashlib.sha256(str(x).encode("utf-8")).digest()
            int_hash = int.from_bytes(h[:6], "big")  # 48-bit hash
            return int_to_base62(int_hash)

        return series.astype(str).map(hash_to_base62)