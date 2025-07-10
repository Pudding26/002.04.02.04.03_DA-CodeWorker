def minmax_scaler(job, feature_range=(0, 1)):
    """
    Applies MinMax scaling to selected features in the dataset, excluding blacklisted columns.

    This method performs MinMax normalization on a subset of columns in `job.attrs.data_num`, 
    scaling them to a given feature range (default: [0, 1]). It automatically excludes columns 
    that are semantically unsuitable for scaling, such as encoded identifiers or categorical labels.

    The columns to exclude are derived from the job’s internal "blacklist", which includes:
        ▸ `index_cols`: hierarchy or identifier fields (e.g. "species", "shotID")  
        ▸ `str_cols`: originally non-numeric string columns that were label-encoded

    These columns are blacklisted because:
        - **Scaling categorical codes can mislead models**, introducing false magnitudes or order.
        - **Scaling IDs or scope labels corrupts grouping structure**, which is critical for downstream tasks like resampling, clustering, and validation.

    Parameters
    ----------
    job : ModelerJob
        A job object containing:
        - `attrs.data_num`: CuPy array of encoded numeric data.
        - `attrs.encoder.cols`: Dict mapping column names to numeric indices.
        - `attrs.blacklist`: Dict with keys `"index_cols"` and `"str_cols"` listing excluded columns.

    feature_range : tuple (float, float), optional
        Desired range for transformed features (default: (0, 1)).

    Returns
    -------
    cp.ndarray
        Scaled CuPy array with same shape as input. Blacklisted columns are copied over unmodified.

    Notes
    -----
    - Only columns **not present in the blacklist** are scaled.
    - A full copy of the data (`X_scaled`) is made to preserve original values for excluded columns.
    - If no columns are eligible for scaling, the method logs a warning and returns the data unchanged.
    - Use `TA52_A_Preprocessor.qm_scaling_check()` after this method to verify output stability.

    Example
    -------
    job.attrs.data_num = TA52_A_Preprocessor._minmax_scaler(job)

    See Also
    --------
    TA52_A_Preprocessor.qm_scaling_check : Post-scaling data validation
    TA52_A_Preprocessor.prepare_numeric_gpu_data : Where blacklist is defined
    """
    
    from cuml.preprocessing import MinMaxScaler
    import cupy as cp
    import logging

    cfg = job.input.preProcessing_instructions
    p = cfg.scaling.standardization.MinMaxScaler

    # Ensure data is present
    if not hasattr(job.attrs, "data_num") or job.attrs.data_num is None:
        raise ValueError("job.attrs.data_num is not initialized before scaling.")

    X = job.attrs.data_num
    cols = job.attrs.encoder.cols
    bl = job.attrs.blacklist

    # Get indices of all blacklisted columns (index + str)
    blacklist_cols = set(bl["index_cols"] + bl["str_cols"])
    blacklist_idx = [cols[c] for c in blacklist_cols if c in cols]

    # Compute all columns to scale (those NOT in blacklist)
    scale_idx = [i for i in range(X.shape[1]) if i not in blacklist_idx]

    # Create empty array to hold scaled values
    X_scaled = cp.array(X)  # full copy, keeps unscaled values in-place

    if scale_idx:
        scaler = MinMaxScaler(feature_range=feature_range)
        X_scaled[:, scale_idx] = scaler.fit_transform(X[:, scale_idx])
    else:
        logging.warning("[SCALER] No columns left to scale after applying blacklist.")

    return X_scaled
