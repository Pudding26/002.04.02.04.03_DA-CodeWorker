def drop_high_nan_cols(job, threshold: float = 0.5):
    """
    Drops columns with more than `threshold` fraction of NaN values.

    Parameters
    ----------
    job : ModelerJob
        The job object containing `data_num` and `encoder`.
    threshold : float
        Fraction of NaNs above which a column is dropped (default = 0.5).

    Returns
    -------
    job : ModelerJob
        Updated job with reduced `data_num`, `encoder.cols`, and `index_col`.
    """

    import cupy as cp

    X = job.attrs.data_num
    nan_mask = cp.isnan(X)
    col_nan_frac = nan_mask.sum(axis=0) / X.shape[0]

    # Identify columns to keep
    keep_mask = col_nan_frac <= threshold
    X_cleaned = X[:, keep_mask]

    # Update encoder
    old_col_map = job.attrs.encoder.cols
    old_cols = list(old_col_map.keys())
    new_cols = [col for i, col in enumerate(old_cols) if keep_mask[i]]
    new_encoder = {col: i for i, col in enumerate(new_cols)}

    # Update job
    job.attrs.data_num = X_cleaned
    job.attrs.encoder.cols = new_encoder

    # Recalculate index_col (if the scope column was kept)
    scope = job.input.scope or "shotID"
    if scope in new_encoder:
        job.input.index_col = new_encoder[scope]
    else:
        raise ValueError(f"Scope column '{scope}' was dropped due to high NaNs.")

    job.stats["preprocessing"]["dropped_nan_columns"] = {
        "threshold": threshold,
        "num_dropped": int((~keep_mask).sum()),
        "original_cols": X.shape[1],
        "remaining_cols": X_cleaned.shape[1]
    }

    return job
