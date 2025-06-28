from app.utils.common.app.utils.dataModels.Jobs.ModelerJob import ModelerJob
import cudf
import time
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import cupy as cp
import numpy as np
from collections import Counter
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from cuml.neighbors import NearestNeighbors
from app.tasks.TA52_Modeler.utils.split_numeric_non_numeric_cudf import split_numeric_non_numeric_cudf

class TA52_A_Preprocessor:

    @staticmethod
    def run(job: ModelerJob) -> None:
        t0 = time.time()
        cfg = job.input.preProcessing_instructions
        df_raw = job.attrs.raw_data
        t1 = time.time()
        # 1. Split numeric and string columns early
        data_num, data_str, index_col = split_numeric_non_numeric_cudf(df=df_raw, index_col="shotID")
        shape_before = list(data_num.shape)
        data_num = cudf.DataFrame({col: data_num[col].astype("float32").copy(deep=True) for col in data_num.columns})


        t2 = time.time()
        factor = 2500
        repeat_idx = cp.tile(cp.arange(len(data_num)), factor)
        data_num = data_num.iloc[repeat_idx]
        repeat_idx = cp.tile(cp.arange(len(data_str)), factor)
        data_str = data_str.iloc[repeat_idx]

        t3 = time.time()
        if cfg.method is None:
            logging.debug1("No preprocessing method specified, skipping.")
            job.attrs.preProcessed_data = data_num
            job.stats["preprocessing"] = {
                "method": None,
                "submethod": None,
                "subsubmethod": None,
                "shape_before": shape_before,
                "shape_after": shape_before,
                "elapsed_time": 0.0,
            }
            return

        try:
            method_cfg = getattr(cfg, cfg.method)
        except Exception as e:
            logging.error(
                f"[PREPROCESS ERROR] Failed to access sub-config for method='{cfg.method}' "
                f"(type: {type(cfg.method)}). Job ID: {getattr(job, 'id', 'unknown')}",
                exc_info=True
            )


        subsubmethod = getattr(method_cfg, method_cfg.submethod).subsubmethod
        job.context = f"Preprocessing with method: {subsubmethod}"
        logging.debug1(f"Preprocessing with method: {subsubmethod}")

        # Use only numeric data throughout
        match cfg.method:
            case "scaling":
                match method_cfg.submethod:
                    case "standardization":
                        match subsubmethod:
                            case "MinMaxScaler":
                                X_scaled = TA52_A_Preprocessor._minmax_scaler(data_num, cfg)
                            case "StandardScaler":
                                X_scaled = TA52_A_Preprocessor._standard_scaler(data_num, cfg)
                            case _:
                                raise ValueError(f"Unsupported standardization.subsubmethod: {subsubmethod}")
                    case _:
                        raise ValueError(f"Unsupported scaling.submethod: {method_cfg.submethod}")

            case "resampling":
                match method_cfg.submethod:
                    case "oversampling":
                        match subsubmethod:
                            case "RandomOverSampler":
                                X_scaled = TA52_A_Preprocessor._random_oversample(data_num, data_str, index_col, cfg)
                            case "SMOTESampler":
                                X_scaled = TA52_A_Preprocessor._smote_sampler(data_num, data_str, index_col, cfg)
                            case _:
                                raise ValueError(f"Unsupported oversampling.subsubmethod: {subsubmethod}")
                    case "undersampling":
                        match subsubmethod:
                            case "RandomUnderSampler":
                                X_scaled = TA52_A_Preprocessor._random_undersample(data_num, data_str, index_col, cfg)
                            case _:
                                raise ValueError(f"Unsupported undersampling.subsubmethod: {subsubmethod}")
                    case "bootstrapping":
                        match subsubmethod:
                            case "BootstrapSampler":
                                X_scaled = TA52_A_Preprocessor._bootstrap(data_num, data_str, index_col, cfg)
                            case _:
                                raise ValueError(f"Unsupported bootstrapping.subsubmethod: {subsubmethod}")
                    case "hybrid":
                        match subsubmethod:
                            case "HybridSampler":
                                X_scaled = TA52_A_Preprocessor._hybrid_sampler(data_num, data_str, index_col, cfg)
                            case _:
                                raise ValueError(f"Unsupported hybrid.subsubmethod: {subsubmethod}")
                    case _:
                        raise ValueError(f"Unsupported resampling.submethod: {method_cfg.submethod}")
            case _:
                raise ValueError(f"Unsupported preProcessing method: {cfg.method}")

        t4 = time.time()
        shape_after = list(X_scaled.shape)

        t5 = time.time()
        job.attrs.preProcessed_data = X_scaled
        job.context = f"SUCCESS of: Preprocessing with method: {subsubmethod}"
        logging.debug1("SUCCESS")
        t6 = time.time()


        job.stats["preprocessing"] = {
            "method": cfg.method,
            "submethod": method_cfg.submethod,
            "subsubmethod": subsubmethod,
            "shape_before": shape_before,
            "shape_after": shape_after,
            "load_config_s": round(t1 - t0, 4),
            "split_columns_s": round(t2 - t1, 4),
            "expand_data_s": round(t3 - t2, 4),
            "preprocess_core_s": round(t4 - t3, 4),
            "postprocess_s": round(t5 - t4, 4),
            "final_assignment_s": round(t6 - t5, 4),
            "total_s": round(t6 - t0, 4)

        }
        logging.debug1(f"Preprocessing completed in {job.stats['preprocessing']['total_s']} seconds")

    @staticmethod
    def _encode_str_df(str_df: cudf.DataFrame):
        encoders = {}
        encoded_df = cudf.DataFrame()
        for col in str_df.columns:
            codes, categories = str_df[col].factorize()
            encoded_df[col] = codes
            encoders[col] = categories
        return encoded_df, encoders

    
    @staticmethod
    def _decode_str_df(encoded_df: cudf.DataFrame, encoders):
        decoded_df = cudf.DataFrame()
        for col in encoded_df.columns:
            categories = encoders[col]
            decoded_df[col] = categories.take(encoded_df[col])
        return decoded_df



    @staticmethod
    def _standard_scaler(data_num, cfg):
        from cuml.preprocessing import StandardScaler
        p = cfg.scaling.standardization.StandardScaler
        scaler = StandardScaler(with_mean=p.with_mean, with_std=p.with_std)
        return scaler.fit_transform(data_num)

    @staticmethod
    def _minmax_scaler(data_num, cfg):
        from cuml.preprocessing import MinMaxScaler
        p = cfg.scaling.standardization.MinMaxScaler
        scaler = MinMaxScaler(feature_range=(p.min, p.max))
        return scaler.fit_transform(data_num)

    @staticmethod
    def _create_scope_label(data_str: cudf.DataFrame, scope: str) -> cudf.Series:
        return data_str[scope]

    @staticmethod
    def _random_oversample(data_num: cudf.DataFrame, data_str: cudf.DataFrame, index_col: str, cfg) -> cudf.DataFrame:
        p = cfg.resampling.oversampling.RandomOverSampler
        y = TA52_A_Preprocessor._create_scope_label(data_str, p.scope)

        # Ensure aligned indices before assignment
        df_combined = data_num.reset_index(drop=True).copy()
        y = y.reset_index(drop=True)
        df_combined["target"] = y

        # Count class distribution
        class_counts = df_combined["target"].value_counts()
        max_class_size = class_counts.max()

        # Oversample each class to match max_class_size
        dfs = []
        arange_cache = {}  # Cache for cp.arange to avoid recreating arrays

        for label in class_counts.index.to_arrow().to_pylist():
            df_class = df_combined[df_combined["target"] == label]
            count = len(df_class)

            if count < max_class_size:
                if count not in arange_cache:
                    arange_cache[count] = cp.arange(count)
                idxs = cp.random.choice(arange_cache[count], size=max_class_size - count, replace=True)
                df_extra = df_class.iloc[idxs]
                df_class = cudf.concat([df_class, df_extra], ignore_index=True)

            dfs.append(df_class)

        df_oversampled = cudf.concat(dfs, ignore_index=True)

        # Drop target and return oversampled features
        df_final = df_oversampled.drop(columns=["target"])
        return df_final




    @staticmethod
    def _random_undersample(data_num: cudf.DataFrame, data_str: cudf.DataFrame, index_col: str, cfg) -> cudf.DataFrame:
        p = cfg.resampling.undersampling.RandomUnderSampler
        y = TA52_A_Preprocessor._create_scope_label(data_str, p.scope)

        X_np = data_num.values_host
        y_np = y.values_host

        sampler = RandomUnderSampler(sampling_strategy="auto", random_state=p.random_state)
        X_resampled, _ = sampler.fit_resample(X_np, y_np)

        return cudf.DataFrame(X_resampled, columns=data_num.columns)

        
    @staticmethod
    def _smote_sampler(data_num: cudf.DataFrame, data_str: cudf.DataFrame, index_col: str, cfg) -> cudf.DataFrame:
        p = cfg.resampling.oversampling.SMOTESampler
        y = TA52_A_Preprocessor._create_scope_label(data_str, p.scope)
        valid_mask = ~data_num.isna().any(axis=1)
        X_clean = data_num[valid_mask]
        y_clean = y[valid_mask]
        X_np = X_clean.to_pandas().values
        y_np = y_clean.to_pandas().values
        knn = NearestNeighbors(n_neighbors=p.k_neighbors)
        smote = SMOTE(k_neighbors=knn, sampling_strategy=p.sampling_strategy, random_state=p.random_state)
        X_resampled, _ = smote.fit_resample(X_np, y_np)
        return cudf.DataFrame.from_records(X_resampled, columns=data_num.columns)

    @staticmethod
    def _bootstrap(data_num: cudf.DataFrame, data_str: cudf.DataFrame, index_col: str, cfg) -> cudf.DataFrame:
        p = cfg.resampling.bootstrapping.BootstrapSampler
        n = p.n_samples or len(data_num)
        indices = cp.random.choice(cp.arange(len(data_num)), size=n, replace=True)
        return data_num.iloc[indices]


    @staticmethod
    def _hybrid_sampler(data_num: cudf.DataFrame, data_str: cudf.DataFrame, index_col: str, cfg) -> cudf.DataFrame:
        p = cfg.resampling.hybrid.HybridSampler
        y = TA52_A_Preprocessor._create_scope_label(data_str, p.scope)

        X_np = data_num.values_host
        y_np = y.values_host
        class_counts = Counter(y_np)

        if p.target_size is not None:
            target = p.target_size
        elif p.strategy == "mean":
            target = int(np.mean(list(class_counts.values())))
        elif p.strategy == "max":
            target = int(np.max(list(class_counts.values())))
        elif p.strategy == "min":
            target = int(np.min(list(class_counts.values())))
        else:
            target = int(np.median(list(class_counts.values())))

        X_chunks = []
        for label in class_counts:
            indices = np.where(y_np == label)[0]
            X_class = X_np[indices]
            y_class = y_np[indices]

            if len(X_class) < target:
                sampler = RandomOverSampler(sampling_strategy={label: target}, random_state=p.random_state)
            elif len(X_class) > target:
                sampler = RandomUnderSampler(sampling_strategy={label: target}, random_state=p.random_state)
            else:
                X_chunks.append(X_class)
                continue

            X_resampled, _ = sampler.fit_resample(X_class, y_class)
            X_chunks.append(X_resampled)

        X_balanced = np.vstack(X_chunks)
        return cudf.DataFrame(X_balanced, columns=data_num.columns)

