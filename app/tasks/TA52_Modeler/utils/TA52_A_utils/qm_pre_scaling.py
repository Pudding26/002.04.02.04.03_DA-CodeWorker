def qm_pre_scaling(job, variance_threshold=1e-8, min_rows=2, threshold_col_fraction=0.5):
    """
    Pre-scaling quality check for cleaning only truly breaking columns.

    Drops all non-blacklisted columns that:
      - Have near-zero variance (constant values)
      - Contain any ±Inf
      - Contain more than threshold_col_fraction NaNs

    Leaves remaining NaNs untouched for downstream processing (e.g. weighting).

    Parameters
    ----------
    job : ModelerJob
        Job with:
            - attrs.data_num (CuPy ndarray)
            - attrs.encoder.cols
            - attrs.blacklist

    variance_threshold : float
        Minimum variance to keep column

    min_rows : int
        Minimum number of rows required to proceed

    threshold_col_fraction : float
        Fraction (0–1) of allowed NaNs in a column (e.g. 0.5 → drop if >50% NaN)

    Returns
    -------
    job : ModelerJob
    """
    import cupy as cp
    import logging

    X = job.attrs.data_num
    encoder = job.attrs.encoder.cols
    bl = job.attrs.blacklist
    colnames = list(encoder.keys())

    blacklist_set = set(bl["index_cols"] + bl["str_cols"])
    blacklist_idx = {encoder[c] for c in blacklist_set if c in encoder}

    var = cp.var(X, axis=0)
    keep_mask = cp.ones(X.shape[1], dtype=bool)
    dropped_cols = []

    num_rows = X.shape[0]

    for i in range(X.shape[1]):
        if i in blacklist_idx:
            continue

        col_data = X[:, i]
        name = colnames[i]

        reason = None
        if var[i] < variance_threshold:
            reason = "zero variance"
        elif cp.isinf(col_data).any():
            reason = "contains Inf"
        else:
            nan_ratio = cp.isnan(col_data).sum() / num_rows
            if nan_ratio > threshold_col_fraction:
                reason = f"NaN > {threshold_col_fraction:.0%}"

        if reason:
            keep_mask[i] = False
            dropped_cols.append((name, reason))

    # If we dropped any columns, apply changes
    if not keep_mask.all():
        X = X[:, keep_mask]

        new_colnames = [colnames[i] for i in range(len(keep_mask)) if keep_mask[i]]
        new_encoder = {name: idx for idx, name in enumerate(new_colnames)}

        job.attrs.data_num = X
        job.attrs.encoder.cols = new_encoder

        if hasattr(job.attrs.encoder, "vals") and job.attrs.encoder.vals is not None:
            job.attrs.encoder.vals = {
                k: v for k, v in job.attrs.encoder.vals.items() if k in new_colnames
            }

    if X.shape[0] < min_rows:
        job.status = "FAILED"
        job.input.fail_trail.mark("preprocessing", "qm_pre_scaling", f"Too few rows: {X.shape[0]}")
        return job

    job.stats.setdefault("preprocessing", {})
    job.stats["preprocessing"]["qm_pre_scaling"] = {
        "dropped_columns": dropped_cols,
        "num_dropped": len(dropped_cols),
        "num_remaining": X.shape[1]
    }

    job.input.fail_trail.mark("preprocessing", "qm_pre_scaling", "passed")
    return job
