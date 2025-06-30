from app.utils.common.app.utils.dataModels.Jobs.ModelerJob import ModelerJob
import cudf
import time
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import cupy as cp
import numpy as np
from typing import List, Dict, Any
from collections import Counter



class TA52_B_Modeler:
    
    @staticmethod
    def run(job):
        return
        X = job.attrs.data_num
        cfg = job.input.metricModel_instructions
        stats = {}

        # 1️⃣ binning (if any)
        if cfg.binning_cfg and cfg.binning_cfg.strategy != "none":
            t_beg = time.time()
            X = TA52_B_Modeler_apply_binning(X, cfg.binning_cfg)
            stats["split_columns_s"] = round(time.time() - t_beg, 4)

        self.job.attrs.preProcessed_data = X  # feed into DR
        shape_before = list(X.shape)

        # 2️⃣ dim reduction (timed externally)
        t0 = time.time()
        dr = cfg.dim_reduction
        match dr.method:
            case "linear":
                if dr.linear.submethod == "PCA":
                    X = self._pca()
            case "manifold":
                if dr.manifold.submethod == "UMAP":
                    X = self._umap()
                elif dr.manifold.submethod == "tSNE":
                    X = self._tsne()
            case "encoder":
                if dr.encoder.submethod == "autoencoderSeg":
                    X = self._seg_autoencoder()
                elif dr.encoder.submethod == "autoencoderImg":
                    X = self._img_autoencoder()
            case _:
                raise ValueError(f"Unknown reduction method: {dr.method}")
        t1 = time.time()

        shape_after = list(X.shape)
        shape_change = [
            round(a / b, 3) if b else None
            for a, b in zip(shape_after, shape_before)
        ]

        self.job.attrs.engineered_data = X
        self.job.stats["feature_engineering"] = {
            "shape_before":  shape_before,
            "shape_after":   shape_after,
            "shape_change":  shape_change,
            "split_columns_s": stats.get("split_columns_s", 0.0),
            "expand_data_s":  0.0,
            "load_config_s":  0.0,
            "preprocess_core_s": round(t1 - t0, 4),
            "postprocess_s":  0.0,
            "final_assignment_s": 0.0,
            "total_s": round(t1 - t0 + stats.get("split_columns_s", 0.0), 4),
        }




    def _pca(self):
        from cuml.decomposition import PCA as cuPCA
        from sklearn.decomposition import PCA as skPCA
        import cupy as cp

        cfg = self.job.input.metricModel_instructions.dim_reduction.linear.PCA
        X_in = self.job.attrs.preProcessed_data

        try:
            mdl = cuPCA(n_components=cfg.n_components, whiten=cfg.whiten, output_type="cupy")
            Z = mdl.fit_transform(X_in)
        except Exception as e:
            Z = cp.asarray(
                skPCA(n_components=cfg.n_components, whiten=cfg.whiten)
                .fit_transform(cp.asnumpy(X_in))
            )
        return Z

