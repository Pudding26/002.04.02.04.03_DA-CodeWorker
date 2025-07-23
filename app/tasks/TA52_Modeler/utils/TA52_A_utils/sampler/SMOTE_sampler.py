def smote_sampler(job):
    """
    Applies SMOTE to synthetically generate samples for minority classes.
    This version is fully scope-aware: it uses `index_col` to determine labels
    based on the current job scope.

    Only rows without NaNs are considered. Uses RAPIDS NearestNeighbors for GPU acceleration.

    Returns:
        cp.ndarray: SMOTE-resampled dataset (features only)
    """
    from app.tasks.TA52_Modeler.TA52_0_ModelerOrchestrator import suppress_logging
    from cuml.neighbors import NearestNeighbors
    from imblearn.over_sampling import SMOTE
    import numpy as np
    import cupy as cp
    try:
        X_gpu = job.attrs.data_train
        idx = int(job.input.index_col)

        # Define valid index columns based on idx position
        valid_indexes = list(range(idx + 1))  # columns 0..idx
        dropped_indexes = list(range(idx + 1, 8))  # columns idx+1..7 (hierarchical drop)

        # Slice relevant portions of X_gpu
        valid_index_cols_gpu = X_gpu[:, valid_indexes]
        feature_cols = [i for i in range(X_gpu.shape[1]) if i not in valid_indexes + dropped_indexes]
        X_features_gpu = X_gpu[:, feature_cols]
        labels_gpu = X_gpu[:, idx]

        # Convert to NumPy for imbalanced-learn compatibility
        X_np = X_features_gpu.get()
        y_np = labels_gpu.get().astype(np.int32)
        valid_index_cols_np = valid_index_cols_gpu.get()

        # Filter rows with no NaNs (clean dataset)
        valid_mask = ~np.isnan(X_np).any(axis=1)
        X_clean = X_np[valid_mask]
        y_clean = y_np[valid_mask]
        valid_index_clean = valid_index_cols_np[valid_mask]

        if len(X_clean) == 0:
            raise ValueError("All rows contain NaNs; SMOTE cannot be applied.")

        # Configure nearest neighbor estimator for SMOTE
        knn = NearestNeighbors(n_neighbors=p.k_neighbors, output_type='numpy')
        knn.fit(X_clean)

        # Initialize SMOTE sampler
        smote = SMOTE(
            k_neighbors=knn,
            sampling_strategy=p.sampling_strategy,
            random_state=p.random_state
        )

        # Apply SMOTE resampling
        with suppress_logging():
            X_resampled, y_resampled = smote.fit_resample(X_clean, y_clean)

        # --------------------------
        # Rebuild valid_indexes after SMOTE (LEFT JOIN semantics):
        # - For each y_resampled (left table), attempt to match on idx column.
        # - If match: use row from valid_index_clean.
        # - If no match: fallback to template row.
        # --------------------------

        # Prepare lookup table from valid_index_clean keyed on column idx
        index_lookup = {
            row[idx]: row.copy() for row in valid_index_clean
        }

        # Define fallback template row: copy last valid_index_clean row
        template_row = valid_index_clean[-1].copy()
        template_row[:idx] = -1  # Optional: sentinel values for unmatched rows

        # Iterate over y_resampled and construct reconstructed valid_index
        reconstructed_indexes = []
        for y in y_resampled:
            row = index_lookup.get(y, template_row.copy())
            row[idx] = y  # Ensure idx column always equals y_resampled value
            reconstructed_indexes.append(row)

        valid_index_reconstructed = np.vstack(reconstructed_indexes)

        # Concatenate reconstructed valid_index and resampled features
        result_np = np.concatenate([valid_index_reconstructed, X_resampled], axis=1)

        # Convert back to CuPy for consistency with pipeline expectations
        result = cp.asarray(result_np)
        job.attrs.encoder.cols = _update_col_encoder(job, idx)

    except Exception as e:
        # Log failure to job trail and fallback safely
        message = f"SMOTE sampling failed: {str(e)}"
        job.input.fail_trail.mark("modelling", "SMOTE Sampler", "FAILED", message)
        result = X_gpu  # Fallback to original input
    
    finally:
        return result, job


def _update_col_encoder(job, idx):
    orig_encoder_cols = job.attrs.encoder.cols

    # 1️⃣ Build ordered list of column names by original position
    sorted_colnames = sorted(orig_encoder_cols.items(), key=lambda kv: kv[1])
    ordered_colnames = [name for name, _ in sorted_colnames]

    # 2️⃣ Slice index columns (0..idx) directly:
    index_colnames = ordered_colnames[:idx + 1]

    # 3️⃣ Remaining columns are simply feature columns after dropping 7 (if 7 was removed):
    # So just pick everything after 7:
    feature_colnames = ordered_colnames[8:]

    # 4️⃣ Assemble new encoder:
    new_encoder_cols = {}

    # Index columns retain positions 0..idx:
    for i, colname in enumerate(index_colnames):
        new_encoder_cols[colname] = i

    # Feature columns shift down to idx+1 onward:
    for j, colname in enumerate(feature_colnames):
        new_encoder_cols[colname] = idx + 1 + j

    # 5️⃣ Update encoder in place:
    return new_encoder_cols
