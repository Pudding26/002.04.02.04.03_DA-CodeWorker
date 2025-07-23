from copy import deepcopy
from typing import List
import numpy as np
import cupy as cp
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
import logging


def create_train_test_subjobs(
    job,
    n_splits: int = 5,
    test_size: float = 0.2,
    random_state: int = 42
) -> List:
    """
    Split a preprocessed `ModelerJob` into sub-jobs with TRAIN/TEST partitions.

    - TRAIN → `attrs.data_train`
    - TEST  → `attrs.data_test`
    - Removes `data_num` from subjobs (clean state)
    - Validates minimum class size before splitting

    Returns list of subjobs, or a single `[job]` marked FAILED if min class check fails.
    """

    def _split_Xy(job) -> tuple[np.ndarray, np.ndarray]:
        X_gpu = job.attrs.data_num
        idx   = int(job.input.index_col)
        y_np  = cp.asnumpy(X_gpu[:, idx]).astype("int32")
        X_np  = cp.asnumpy(X_gpu)
        return X_np, y_np

    def min_class_size_check(y: np.ndarray, min_count: int = 2) -> bool:
        counts = Counter(y)
        too_small = [cls for cls, cnt in counts.items() if cnt < min_count]
        return len(too_small) == 0  # True = safe, False = too small

    X, y = _split_Xy(job)
    original_shape = X.shape
    
    logging.debug2(f"[SPLIT] Starting split for job {job.job_uuid} | data_num shape: {original_shape} | test_size={test_size} | n_splits={n_splits}")
    
    # 🔒 Check minimum class size before proceeding:
    if not min_class_size_check(y, min_count=2):
        job.status = "FAILED"
        job.input.fail_trail.mark("preprocessing", "splitter", "FAILED: min class size threshold")
        return job

    splitter = StratifiedShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state
    )

    subjobs = []
    train_shapes = []
    test_shapes = []

    for fold_no, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        sj = deepcopy(job)
        sj.input.outer_fold = fold_no

        X_train, X_test = X[train_idx], X[test_idx]

        sj.attrs.data_train = cp.asarray(X_train)
        sj.attrs.data_test  = cp.asarray(X_test)

        if hasattr(sj.attrs, 'data_num'):
            del sj.attrs.data_num

        # Optional stats:
        sj.stats.setdefault("preprocessing", {})
        sj.stats["preprocessing"]["split"] = {
            "fold_no": fold_no,
            "shape_before": original_shape,
            "shape_train": X_train.shape,
            "shape_test": X_test.shape
        }
        train_shapes.append(X_train.shape[0])
        test_shapes.append(X_test.shape[0])

        subjobs.append(sj)

        # 🔔 Summary log after all folds:
    avg_train_size = round(np.mean(train_shapes), 1) if train_shapes else 0
    avg_test_size = round(np.mean(test_shapes), 1) if test_shapes else 0
    logging.debug2(f"[SPLIT] Completed {len(subjobs)} folds | avg train size={avg_train_size} rows | avg test size={avg_test_size} rows")

    return subjobs
