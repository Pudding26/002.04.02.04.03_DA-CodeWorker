from app.tasks.TA52_Modeler.TA52_0_ModelerOrchestrator import ModelerJob
import cupy as cp

def rf_validation_classifier(job: ModelerJob, Z: cp.ndarray, y: cp.ndarray, label: str, frac: float) -> ModelerJob:
    """
    Run N-fold Random Forest classification on GPU and store results.

    Parameters
    ----------
    job : ModelerJob
        The active pipeline job object.
    Z : cp.ndarray
        Reduced-dimension embedding (features).
    y : cp.ndarray
        Encoded labels corresponding to the current index column.
    label : str
        The name of the index column being evaluated.
    frac : float
        The fraction of dimensions retained in the embedding.

    Notes
    -----
    - Stores accuracy into `job.attrs.validation_results_dict[bootstrap][label][frac]['rf_acc']`
    - Stores timing into `job.stats['validation'][bootstrap][label][frac]['rf_time']`
    - Tracks result status in `job.input.fail_trail.mark_validation(...)`
    - Skips execution if dataset has fewer than PARAMS_DICT["MIN_SAMPLE_THRESHOLD"] rows.
    """
    from cuml.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score
    import warnings
    import time
    import numpy as np
    import cupy as cp
    import logging

    from app.tasks.TA52_Modeler.TA52_C_Validator import _GPU_LOCK
    from app.tasks.TA52_Modeler.TA52_0_ModelerOrchestrator import ModelerJob
    from app.tasks.TA52_Modeler.TA52_C_Validator import PARAMS_DICT

    bootstrap = getattr(job.input, "bootstrap_iteration", 0)

    try:
        # --- Skip if too few samples ---
        if Z.shape[0] < PARAMS_DICT["MIN_SAMPLE_THRESHOLD"]:
            logging.debug1(f"[RF] Skip label='{label}' @ frac={frac:.2f} – too few rows ({Z.shape[0]})")
            job.input.fail_trail.mark_validation(
                bootstrap=bootstrap,
                label=label,
                frac=frac,
                model="rf",
                status="skipped",
                error="Too few samples"
            )
            return

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The number of bins.*", category=UserWarning)

            skf = StratifiedKFold(
                n_splits=PARAMS_DICT["RF_NUM_FOLDS"],
                shuffle=True,
                random_state=42
            )

            y_cpu = cp.asnumpy(y)
            rf_scores = []
            t0 = time.time()

            for train_idx, test_idx in skf.split(cp.zeros(len(y_cpu)), y_cpu):
                Z_train, Z_test = Z[train_idx], Z[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                with _GPU_LOCK:
                    rf = RandomForestClassifier(
                        n_estimators=PARAMS_DICT["RF_N_ESTIMATORS"],
                        max_depth=PARAMS_DICT["RF_MAX_DEPTH"],
                        max_features=PARAMS_DICT["RF_MAX_FEATURES"],
                        n_bins=PARAMS_DICT["RF_N_BINS"],
                        accuracy_metric=PARAMS_DICT["ACCURACY_METRIC"],
                    )
                    rf.fit(Z_train, y_train)
                    y_pred_rf = rf.predict(Z_test)
                    cp.cuda.runtime.deviceSynchronize()

                rf_acc = float(accuracy_score(cp.asnumpy(y_test), cp.asnumpy(y_pred_rf)))
                rf_scores.append(rf_acc)

            elapsed = time.time() - t0
            mean_rf = round(np.mean(rf_scores), 4)

            logging.debug1(f"[RF] label='{label}' frac={frac:.2f} RF={mean_rf:.4f}")

            # --- Store Accuracy ---
            job.attrs.validation_results_dict \
                .setdefault(bootstrap, {}) \
                .setdefault(label, {}) \
                .setdefault(frac, {})['rf_acc'] = mean_rf

            # --- Store Timing ---
            job.stats \
                .setdefault("validation", {}) \
                .setdefault(bootstrap, {}) \
                .setdefault(label, {}) \
                .setdefault(frac, {})['rf_time'] = elapsed

            # --- Mark as passed ---
            job.input.fail_trail.mark_validation(
                bootstrap=bootstrap,
                label=label,
                frac=frac,
                model="rf",
                status="passed",
                error=None
            )

    except Exception as e:
        logging.warning(f"[RF] Failed classification for label='{label}' @ frac={frac:.2f}: {e}")
        job.input.fail_trail.mark_validation(
            bootstrap=bootstrap,
            label=label,
            frac=frac,
            model="rf",
            status="failed",
            error=str(e)
        )
        return job

    return job
