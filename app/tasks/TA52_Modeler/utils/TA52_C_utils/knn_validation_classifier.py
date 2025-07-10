from app.tasks.TA52_Modeler.TA52_0_ModelerOrchestrator import ModelerJob
import cupy as cp

def knn_validation_classifier(job: ModelerJob, Z: cp.ndarray, y: cp.ndarray, label: str, frac: float) -> ModelerJob:
    """
    Run N-fold KNN classification using cuML and store results.

    Parameters
    ----------
    job : ModelerJob
        The active pipeline job object.
    Z : cp.ndarray
        Reduced-dimension feature matrix.
    y : cp.ndarray
        Encoded labels for current index column.
    label : str
        The index column being evaluated.
    frac : float
        Fraction of dimensions retained in the embedding.

    Notes
    -----
    - Uses cuML KNeighborsClassifier with N-fold StratifiedKFold.
    - Accuracy stored in `job.attrs.validation_results_dict[bootstrap][label][frac]['knn_acc']`
    - Timing stored in `job.stats['validation'][bootstrap][label][frac]['knn_time']`
    - Failure status recorded via `fail_trail.mark_validation(...)`
    """
    from cuml.neighbors import KNeighborsClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score
    import numpy as np
    import cupy as cp
    import time
    import logging

    from app.tasks.TA52_Modeler.TA52_C_Validator import _GPU_LOCK
    from app.tasks.TA52_Modeler.TA52_C_Validator import PARAMS_DICT

    bootstrap = getattr(job.input, "bootstrap_iteration", 0)

    try:
        Z_host, y_host = cp.asnumpy(Z), cp.asnumpy(y)

        if len(Z_host) < PARAMS_DICT["MIN_SAMPLE_THRESHOLD"]:
            logging.debug1(f"[KNN] Skip label='{label}' @ frac={frac:.2f} – too few rows ({len(Z_host)})")
            job.input.fail_trail.mark_validation(
                bootstrap=bootstrap,
                label=label,
                frac=frac,
                model="knn",
                status="skipped",
                error="Too few samples"
            )
            return

        skf = StratifiedKFold(n_splits=PARAMS_DICT["KNN_NUM_FOLDS"], shuffle=True)
        knn_scores = []
        t0 = time.time()

        for train_idx, test_idx in skf.split(Z_host, y_host):
            Z_train, Z_test = Z_host[train_idx], Z_host[test_idx]
            y_train, y_test = y_host[train_idx], y_host[test_idx]

            with _GPU_LOCK:
                knn = KNeighborsClassifier(n_neighbors=PARAMS_DICT["KNN_N_NEIGHBORS"])
                knn.fit(cp.asarray(Z_train), cp.asarray(y_train))
                y_pred_knn = knn.predict(cp.asarray(Z_test))
                #proba = knn.predict_proba(cp.asarray(Z_test))  # ✅ wrap in cp.asarray
                #confidences = knn.predict_proba(Z_test).max(axis=1)  # knn_conf

                cp.cuda.runtime.deviceSynchronize()

            #confidences = cp.asnumpy(proba).max(axis=1)  # ✅ convert before numpy ops


            acc = float(accuracy_score(y_test, cp.asnumpy(y_pred_knn)))
            knn_scores.append(acc)

        mean_knn = round(np.mean(knn_scores), 4)
        elapsed = time.time() - t0

        logging.debug1(f"[KNN] label='{label}' frac={frac:.2f} KNN={mean_knn:.4f}")

        # --- Store Accuracy ---
        job.attrs.validation_results_dict \
            .setdefault(bootstrap, {}) \
            .setdefault(label, {}) \
            .setdefault(frac, {})['knn_acc'] = mean_knn

        # --- Store Timing ---
        job.stats \
            .setdefault("validation", {}) \
            .setdefault(bootstrap, {}) \
            .setdefault(label, {}) \
            .setdefault(frac, {})["knn_time"] = elapsed

        # --- Status Marking ---
        job.input.fail_trail.mark_validation(
            bootstrap=bootstrap,
            label=label,
            frac=frac,
            model="knn",
            status="passed"
        )

    except Exception as e:
        logging.warning(f"[KNN] Failed classification for label='{label}' @ frac={frac:.2f}: {e}")
        job.input.fail_trail.mark_validation(
            bootstrap=bootstrap,
            label=label,
            frac=frac,
            model="knn",
            status="failed",
            error=str(e)
        )
        return job

    return job
