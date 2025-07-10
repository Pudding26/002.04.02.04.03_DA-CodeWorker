def random_sampler(job):
    """
    Generalized random sampler that supports:
    - Undersampling
    - Oversampling
    - Hybrid (balanced) sampling

    Parameters
    ----------
    job : ModelerJob
        Job object containing encoded GPU data and target index.

    Returns
    -------
    cp.ndarray
        Resampled dataset.
    """
    import cupy as cp

    res_cfg = job.input.preProcessing_instructions.resampling
    if not res_cfg or res_cfg.method != "RandomSampler":
        raise ValueError("Invalid or missing RandomSampler configuration")
    p = getattr(res_cfg, "RandomSampler", None)
    if p is None:
        raise ValueError("Missing RandomSampler configuration block")

    X = job.attrs.data_num
    idx = int(job.input.index_col)
    labels = X[:, idx]

    unique_ids, counts = cp.unique(labels, return_counts=True)

    strategy = p.strategy
    mode = p.mode

    if strategy == "min":
        target = int(cp.min(counts))
    elif strategy == "median":
        target = int(cp.median(counts))
    elif strategy == "mean":
        target = int(cp.mean(counts))
    elif strategy == "max":
        target = int(cp.max(counts))
    elif isinstance(strategy, int):
        target = int(strategy)
    else:
        raise ValueError(f"Unsupported sampling strategy: {strategy}")

    result = []
    for label in unique_ids.tolist():
        mask = labels == label
        X_group = X[mask]
        n = X_group.shape[0]

        if mode == "under":
            if n > target:
                selected = cp.random.choice(n, size=target, replace=False)
                X_group = X_group[selected]

        elif mode == "over":
            if n < target:
                extra_idx = cp.random.choice(n, size=target - n, replace=True)
                X_extra = X_group[extra_idx]
                X_group = cp.concatenate([X_group, X_extra])

        elif mode == "hybrid":
            if n < target:
                extra_idx = cp.random.choice(n, size=target - n, replace=True)
                X_extra = X_group[extra_idx]
                X_group = cp.concatenate([X_group, X_extra])
            elif n > target:
                selected = cp.random.choice(n, size=target, replace=False)
                X_group = X_group[selected]

        else:
            raise ValueError(f"Unsupported mode: {mode}. Must be 'under', 'over', or 'hybrid'.")

        result.append(X_group)

    return cp.concatenate(result, axis=0)


