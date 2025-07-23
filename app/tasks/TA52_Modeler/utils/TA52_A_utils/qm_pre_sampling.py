def qm_pre_sampling(job, min_classes=2, min_per_class=5, min_rows=20):
    """
    Step – Pre-Sampling Quality Check

    Cleans and validates class distribution within the active scope column prior to 
    resampling or class balancing. This step removes underrepresented classes that 
    could break resampling strategies such as SMOTE, undersampling, or hybrid methods.

    Specifically:
      ▸ Drops all rows belonging to classes with fewer than `min_per_class` members
      ▸ Operates on the scope-defined column (given by `job.input.index_col`)
      ▸ Fails the job only if the resulting dataset has:
            - Fewer than `min_classes` unique labels
            - Fewer than `min_rows` total samples

    Parameters
    ----------
    job : ModelerJob
        Job object containing:
          - `attrs.data_train` : CuPy array of numeric data
          - `input.index_col` : Integer or string indicating the active class/scope column
          - `input.scope` : Used for logging (e.g., 'species', 'region')
          - `attrs.encoder_vals` : Optional label-encoded class metadata

    min_classes : int, default=2
        Minimum number of distinct classes required to continue.

    min_per_class : int, default=5
        Minimum number of samples a class must have to be retained.

    min_rows : int, default=20
        Minimum number of total rows required after filtering.

    Returns
    -------
    job : ModelerJob
        Cleaned job with filtered data. If requirements are not met, `job.status` is
        set to `"FAILED"` and an appropriate warning is logged.

    Notes
    -----
    - This method is **scope-aware**: it uses the column identified by 
      `job.input.index_col` to group and evaluate classes.
    - Safe for use before any sampling logic (e.g., over/under/SMOTE).
    - Downstream components should check `job.status` before continuing.
    - This step does not modify blacklists or class encodings directly.

    Logging
    -------
    Logs dropped row count and remaining class stats.
    Marks failures for low row or class counts after filtering.

    Example
    -------
    job = qm_pre_sampling(job, min_per_class=5)
    if job.status == "FAILED":
        return job
    """

    import cupy as cp
    import logging
    
    X = job.attrs.data_train
    scope_col_idx = int(job.input.index_col)  # Dynamic scope-aware index
    y = X[:, scope_col_idx]

    labels, counts = cp.unique(y, return_counts=True)

    # Filter out low-count classes
    keep_labels = labels[counts >= min_per_class]
    if len(keep_labels) < len(labels):
        keep_mask = cp.isin(y, keep_labels)
        dropped = int(X.shape[0] - cp.sum(keep_mask).item())

        job.attrs.data_train = X[keep_mask]
        if hasattr(job.attrs, "encoder_vals") and job.attrs.encoder_vals is not None:
            job.attrs.encoder_vals = job.attrs.encoder_vals[keep_mask]

        logging.warning(f"[QM-PRE] Dropped {dropped} rows with underrepresented classes (<{min_per_class}) in scope='{job.input.scope}'")

        # Recalculate y and class distribution
        y = job.attrs.data_train[:, scope_col_idx]
        labels, counts = cp.unique(y, return_counts=True)
        job.stats.setdefault("preprocessing", {})["qm_pre_sampling"] = {
            "scope_col": job.input.scope,
            "scope_col_idx": int(scope_col_idx),
            "dropped_rows": int(dropped),
            "remaining_rows": int(job.attrs.data_train.shape[0]),
            "remaining_classes": int(len(labels)),
            "class_distribution": {
                int(l): int(c) for l, c in zip(labels.tolist(), counts.tolist())
            }
        }

    if job.attrs.data_train.shape[0] < min_rows:
        job.status = "FAILED"
        logging.warning(f"[QM-PRE] FAIL – total rows = {job.attrs.data_train.shape[0]} < {min_rows} after cleaning")
        job.input.fail_trail.mark("preprocessing", "qm_pre_sampling", f"Too few rows: {job.attrs.data_train.shape[0]}")
        return job

    if len(labels) < min_classes:
        job.status = "FAILED"
        logging.warning(f"[QM-PRE] FAIL – only {len(labels)} class(es) left in scope='{job.input.scope}' after filtering")
        job.input.fail_trail.mark("preprocessing", "qm_pre_sampling", f"Too few classes: {len(labels)} < {min_classes}")
        return job

    return job