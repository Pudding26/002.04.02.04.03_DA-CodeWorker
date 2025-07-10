import cupy as cp

def bin_fraction_weighting(job) -> cp.ndarray:
    """
    Applies custom feature weighting using encoded fraction columns per bin.

    This method adjusts feature values by multiplying them with their corresponding 
    fraction values, based on a binning pattern in the column names. Each "bin" 
    (e.g. 'p00-p05') is expected to contain:
      ▸ Several input features
      ▸ One associated "_fraction" feature that acts as a weight

    The operation:
        weighted_value = raw_value * fraction_value
    is applied per bin, per feature, in-place within `job.attrs.data_num`.

    Assumptions
    -----------
    - The data matrix (`job.attrs.data_num`) is a CuPy array containing all features.
    - All columns have been label-encoded, and their mapping is stored in
      `job.attrs.encoder.cols`, a dict[str → int].
    - Bin names follow the naming pattern:
        'area_by_area_<bin>_<feature>', e.g., 'area_by_area_p00-p05_height'
        'area_by_area_p00-p05_fraction' ← used as the weight

    Returns
    -------
    cp.ndarray
        The weighted data matrix, with features adjusted by their respective
        per-bin fraction values.

    Notes
    -----
    - Blacklisted columns are not considered, but should already be excluded upstream.
    - This method is typically used **after scaling** and **before modeling**.
    - NaNs in fraction columns will propagate unless handled separately.
    - This method mutates the job's feature matrix in-place and returns it.

    See Also
    --------
    - TA52_A_Preprocessor.qm_pre_scaling : Drops columns that could corrupt this step
    - TA52_B_Modeler : Uses weighted matrix for model fitting
    """
    import cupy as cp
    import re
    import logging

    def extract_bin_info_from_encoder(encoder_cols):
        """
        From a flat encoder mapping (name → index), extract per-bin:
            - all feature indices
            - the single fraction column index
        Returns:
            dict: {
                'p00-p05': {
                    'feature_indices': [...],
                    'fraction_index': int
                }
            }
        """
        bin_info = {}

        for name, idx in encoder_cols.items():
            match = re.search(r"area_by_area_(p\d{2}-p\d{2})_", name)
            if not match:
                continue

            bin_label = match.group(1)

            if name.endswith("_fraction"):
                bin_info.setdefault(bin_label, {})["fraction_index"] = idx
            else:
                bin_info.setdefault(bin_label, {}).setdefault("feature_indices", []).append(idx)

        return bin_info

    X = cp.asarray(job.attrs.data_num)
    bin_info = extract_bin_info_from_encoder(job.attrs.encoder.cols)

    if not bin_info:
        logging.debug1("No bin info found — skipping bin weighting.")
        return X

    X_weighted = X.copy()

    for bin_label, info in bin_info.items():
        if "fraction_index" not in info or "feature_indices" not in info:
            continue

        frac = X[:, info["fraction_index"]].reshape(-1, 1)  # (n_samples, 1)
        cols = info["feature_indices"]
        X_weighted[:, cols] = cp.nan_to_num(X[:, cols] * frac, nan=0.0)

    return X_weighted