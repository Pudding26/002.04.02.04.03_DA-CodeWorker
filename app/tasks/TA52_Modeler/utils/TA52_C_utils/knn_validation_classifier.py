from app.tasks.TA52_Modeler.TA52_0_ModelerOrchestrator import ModelerJob
import cupy as cp

def knn_validation_classifier(
    job: ModelerJob,
    Z_train: cp.ndarray,
    y_train: cp.ndarray,
    Z_test: cp.ndarray,
    y_test: cp.ndarray,
    label: str,
    frac: float
) -> ModelerJob:
    """
    Run single-shot KNN validation on explicitly pre-split train/test sets.

    Parameters
    ----------
    job : ModelerJob
        Current pipeline job object.
    Z_train : cp.ndarray
        PCA-reduced features for training.
    y_train : cp.ndarray
        Encoded target labels for training.
    Z_test : cp.ndarray
        PCA-reduced features for validation.
    y_test : cp.ndarray
        Encoded target labels for validation.
    label : str
        The index column being validated.
    frac : float
        Dimensionality reduction fraction.

    Notes
    -----
    - Trains on provided Z_train/y_train.
    - Predicts and scores on provided Z_test/y_test.
    - Accuracy stored in `job.attrs.validation_results_dict`.
    - Timing stored in `job.stats['validation']`.
    - Failure status recorded via `fail_trail.mark_validation(...)`.
    """
    from cuml.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    import time
    import logging

    from app.tasks.TA52_Modeler.TA52_C_Validator import _GPU_LOCK
    from app.tasks.TA52_Modeler.TA52_C_Validator import PARAMS_DICT

    bootstrap_no = getattr(job.input, "bootstrap_iteration", 0)
    fold_no = getattr(job.input, "outer_fold", 0)


    try:
        if len(cp.unique(y_test)) < 2:
            logging.debug1(
                f"[RF] Skip label='{label}' fold={fold_no} bootstrap={bootstrap_no} frac={frac:.2f} "
                f"insufficient class diversity"
            )
            job.input.fail_trail.mark_validation(
                fold_no=fold_no,
                bootstrap_no=bootstrap_no,
                label=label,
                frac=frac,
                model="rf",
                status="skipped",
                error="Insufficient class diversity"
            )
            return job

        t0 = time.time()

        with _GPU_LOCK:
            knn = KNeighborsClassifier(
                n_neighbors=PARAMS_DICT["KNN_N_NEIGHBORS"]
            )
            knn.fit(Z_train, y_train)
            preds = knn.predict(Z_test)
            cp.cuda.runtime.deviceSynchronize()

        acc = float(accuracy_score(cp.asnumpy(y_test), cp.asnumpy(preds)))
        elapsed = time.time() - t0

        logging.debug1(f"[KNN] label='{label}' frac={frac:.2f} KNN={acc:.4f}")

        job.attrs.validation_results_dict \
            .setdefault(fold_no, {}) \
            .setdefault(bootstrap_no, {}) \
            .setdefault(label, {}) \
            .setdefault(frac, {})['knn_acc'] = acc

        job.stats \
            .setdefault("validation", {}) \
            .setdefault(fold_no, {}) \
            .setdefault(bootstrap_no, {}) \
            .setdefault(label, {}) \
            .setdefault(frac, {})["knn_time"] = elapsed

        job.input.fail_trail.mark_validation(
            fold_no=fold_no,
            bootstrap_no=bootstrap_no,
            label=label,
            frac=frac,
            model="knn",
            status="passed"
        )

    except Exception as e:
        logging.warning(f"[KNN] Failed KNN validation label='{label}' frac={frac:.2f}: {e}")
        job.input.fail_trail.mark_validation(
            fold_no=fold_no,
            bootstrap_no=bootstrap_no,
            label=label,
            frac=frac,
            model="knn",
            status="failed",
            error=str(e)
        )

    return job
