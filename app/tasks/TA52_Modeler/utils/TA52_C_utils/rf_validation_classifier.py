from app.tasks.TA52_Modeler.TA52_0_ModelerOrchestrator import ModelerJob
import cupy as cp

def rf_validation_classifier(
    job: ModelerJob,
    Z_train: cp.ndarray,
    y_train: cp.ndarray,
    Z_test: cp.ndarray,
    y_test: cp.ndarray,
    label: str,
    frac: float
) -> ModelerJob:
    """
    Run single-shot Random Forest validation on explicitly pre-split train/test sets.

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
    from cuml.ensemble import RandomForestClassifier
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
                error="insufficient class diversity"
            )
            return job
        if Z_train.shape[1] < 5:
            logging.warning(f"[RF] Skipping RF validation: too few features ({Z_train.shape[1]})")
            job.input.fail_trail.mark_validation(
                fold_no=fold_no, bootstrap_no=bootstrap_no, label=label, frac=frac,
                model="rf", status="skipped", error="too few features"
            )
            return job


        t0 = time.time()

        with _GPU_LOCK:




            
            rf = RandomForestClassifier(
                n_estimators=PARAMS_DICT["RF_N_ESTIMATORS"],
                max_depth=PARAMS_DICT["RF_MAX_DEPTH"],
                max_features=PARAMS_DICT["RF_MAX_FEATURES"],
                n_bins=PARAMS_DICT["RF_N_BINS"],
                accuracy_metric=PARAMS_DICT["ACCURACY_METRIC"],
            )
            #logging.debug2("[FIT] RF fitting on GPU")
            rf.fit(Z_train, y_train)
            #logging.debug2("[PREDICT] RF fitting on GPU")
            preds = rf.predict(Z_test)
            cp.cuda.runtime.deviceSynchronize()

        acc = float(accuracy_score(cp.asnumpy(y_test), cp.asnumpy(preds)))
        elapsed = time.time() - t0

        logging.debug1(f"[RF] label='{label}' frac={frac:.2f} RF={acc:.4f}")

        job.attrs.validation_results_dict \
            .setdefault(fold_no, {}) \
            .setdefault(bootstrap_no, {}) \
            .setdefault(label, {}) \
            .setdefault(frac, {})['rf_acc'] = acc

        job.stats \
            .setdefault("validation", {}) \
            .setdefault(fold_no, {}) \
            .setdefault(bootstrap_no, {}) \
            .setdefault(label, {}) \
            .setdefault(frac, {})['rf_time'] = elapsed

        job.input.fail_trail.mark_validation(
            fold_no=fold_no,
            bootstrap_no=bootstrap_no,
            label=label,
            frac=frac,
            model="rf",
            status="passed"
        )

    except Exception as e:
        logging.warning(f"[RF] Failed RF validation label='{label}' frac={frac:.2f}: {e}")
        job.input.fail_trail.mark_validation(
            fold_no=fold_no,
            bootstrap_no=bootstrap_no,
            label=label,
            frac=frac,
            model="rf",
            status="failed",
            error=str(e)
        )

    return job
