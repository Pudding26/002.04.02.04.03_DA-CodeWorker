import cudf
import pandas as pd

def split_numeric_non_numeric_cudf(
    df: pd.DataFrame | cudf.DataFrame,
    index_col: str
) -> tuple[cudf.DataFrame, cudf.DataFrame, cudf.Series]:
    """
    Splits a DataFrame into:
    - numeric-only cudf.DataFrame
    - non-numeric cudf.DataFrame (excluding index_col)
    - index column as cudf.Series

    Parameters
    ----------
    df : pd.DataFrame or cudf.DataFrame
        The input data.
    index_col : str
        The name of the column to use as a unique index (preserved separately).

    Returns
    -------
    numeric_df : cudf.DataFrame
    non_numeric_df : cudf.DataFrame
    index_series : cudf.Series
    """
    # Convert pandas to cudf if needed
    if isinstance(df, pd.DataFrame):
        df = cudf.DataFrame.from_pandas(df)

    if index_col not in df.columns:
        raise ValueError(f"Index column '{index_col}' not found in DataFrame.")

    # Preserve index column
    index_series = df[index_col].copy(deep=True)

    # Drop the index col for type filtering
    df_wo_index = df.drop(columns=[index_col])

    numeric_cols = df_wo_index.select_dtypes(include=["number"]).columns
    non_numeric_cols = [col for col in df_wo_index.columns if col not in numeric_cols]

    numeric_df = df_wo_index[numeric_cols].copy(deep=True)
    non_numeric_df = df_wo_index[non_numeric_cols].copy(deep=True)

    

    index_series = cudf.Series(index_series)
    # Re-add index column to non-numeric DataFrame
    non_numeric_df[index_col] = index_series

    return numeric_df, non_numeric_df, index_series

