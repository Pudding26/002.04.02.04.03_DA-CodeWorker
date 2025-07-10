def hdbscan_worker(queue, Z_host, y_host, label, frac, bootstrap, min_cluster_size, min_samples, variant_name):
    import cupy as cp
    from cuml.cluster import HDBSCAN
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from app.tasks.TA52_Modeler.TA52_C_Validator import _GPU_LOCK, PARAMS_DICT
    import numpy as np
    import time

    try:
        t0 = time.time()

        Z_gpu = cp.asarray(Z_host)
        y_gpu = cp.asarray(y_host)

        with _GPU_LOCK:
            if Z_gpu.dtype != cp.float32:
                Z_gpu = Z_gpu.astype(cp.float32)

            has_nan = cp.isnan(Z_gpu).any()
            has_inf = not cp.isfinite(Z_gpu).all()
            constant_cols = int(cp.any(cp.std(Z_gpu, axis=0) == 0))
            zero_dim = Z_gpu.shape[1] == 0
            too_small = Z_gpu.shape[0] < PARAMS_DICT["HDBSCAN_MIN_SAMPLE_THRESHOLD"]
            too_small_cluster = min_cluster_size >= Z_gpu.shape[0]

            if any([has_nan, has_inf, zero_dim, too_small, too_small_cluster, constant_cols]):
                reason = " → ".join([
                    f"NaN={has_nan}",
                    f"Inf={has_inf}",
                    f"ZeroDim={zero_dim}",
                    f"TooSmall={too_small}",
                    f"MinCluster>{Z_gpu.shape[0]}",
                    f"ConstCols={constant_cols}"
                ])
                queue.put(("skip", reason))
                return

            clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, prediction_data=False)
            cluster_labels = clusterer.fit_predict(Z_gpu)
            cp.cuda.runtime.deviceSynchronize()

        cluster_labels_host = cp.asnumpy(cluster_labels)
        valid_mask = cluster_labels_host >= 0

        if valid_mask.sum() < 2:
            queue.put(("skip", "<2 valid clusters"))
            return

        ari = adjusted_rand_score(y_gpu[valid_mask], cluster_labels_host[valid_mask])
        nmi = normalized_mutual_info_score(y_gpu[valid_mask], cluster_labels_host[valid_mask])

        result = {
            "ari": ari,
            "nmi": nmi,
            "duration": time.time() - t0,
        }
        queue.put(("success", result))

    except Exception as e:
        queue.put(("error", str(e)))
