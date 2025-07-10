def smote_sampler(job):
    """
    Applies SMOTE to synthetically generate samples for minority classes.
    This version is fully scope-aware: it uses `index_col` to determine labels
    based on the current job scope.

    Only rows without NaNs are considered. Uses RAPIDS NearestNeighbors for GPU acceleration.

    Returns:
        cp.ndarray: SMOTE-resampled dataset (features only)
    """
    from cuml.neighbors import NearestNeighbors
    from imblearn.over_sampling import SMOTE
    import numpy as np
    import cupy as cp

    # Get sampler config from flattened structure
    res_cfg = job.input.preProcessing_instructions.resampling
    if not res_cfg or res_cfg.method != "SMOTESampler":
        raise ValueError("Invalid or missing SMOTE configuration")
    p = getattr(res_cfg, "SMOTESampler", None)
    if p is None:
        raise ValueError("Missing SMOTESampler configuration block")

    X_gpu = job.attrs.data_num
    idx = int(job.input.index_col)

    X_np = X_gpu.get()
    y_np = X_gpu[:, idx].get()

    valid_mask = ~np.isnan(X_np).any(axis=1)
    X_clean = X_np[valid_mask]
    y_clean = y_np[valid_mask]

    if len(X_clean) == 0:
        raise ValueError("All rows contain NaNs; SMOTE cannot be applied.")

    knn = NearestNeighbors(n_neighbors=p.k_neighbors, output_type='numpy')
    knn.fit(X_clean)

    smote = SMOTE(
        k_neighbors=knn,
        sampling_strategy=p.sampling_strategy,
        random_state=p.random_state
    )

    X_resampled, _ = smote.fit_resample(X_clean, y_clean)
    return cp.asarray(X_resampled)
