def bootstrap_sampler(job):
    """
    Simple bootstrap sampling: randomly samples rows (with replacement)
    until the total number of rows meets or exceeds a target.

    Args:
        job: Job object with .attrs.data_num and .input.index_col
             and .input.preProcessing_instructions.bootstrapping.n_samples

    Returns:
        job: Same job with .attrs.data_num replaced by bootstrapped sample
    """
    import cupy as cp

    X = job.attrs.data_num
    n_samples = job.input.preProcessing_instructions.bootstrapping.n_samples
    n_total = X.shape[0]

    if n_total == 0:
        raise ValueError("Original dataset is empty — cannot bootstrap.")

    sampled_idx = cp.random.choice(n_total, size=n_samples, replace=True)
    job.attrs.data_num = X[sampled_idx]

    return job
