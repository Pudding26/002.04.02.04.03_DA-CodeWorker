import pandas as pd
import cupy as cp
from cuml.decomposition import PCA
from cuml.manifold import UMAP
import time
import logging

def generate_comparative_umap_embeddings(job):
    """
    Generate UMAP embeddings for best and worst scoring models (same frac) per label and scope.
    Annotates each point with label metadata and UMAP type ("best" or "worst").
    Stores timing in job.stats["embedding"][bootstrap][label][frac][umap_type].
    Stores failure info via job.input.fail_trail.mark_validation.

    Parameters
    ----------
    job : ModelerJob

    Returns
    -------
    pd.DataFrame
        DataFrame containing UMAP coordinates and metadata.
    """
    df = job.attrs.validation_results_df
    if df is None or df.empty:
        logging.debug2(f"[UMAP] No results DataFrame for job {job.job_uuid}")
        return None

    rows = []
    data_num = job.attrs.data_num
    encoder = job.attrs.encoder.cols
    pca_results = job.attrs.multi_pca_results
    fail_trail = job.input.fail_trail
    uuid = getattr(job, "job_uuid", "unknown")
    bootstrap = getattr(job.input, "bootstrap_iteration", 0)

    INDEX_COLS = ["family", "genus", "species", "sourceID", "specimenID", "sampleID", "stackID", "shotID"]

    grouped = (
        df[df["frac"] < 0.25]
        .sort_values(["label", "scope", "frac", "rf_acc"], ascending=[True, True, True, False])
        .groupby(["scope", "label"])
    )

    for (scope, label), group in grouped:
        if group.empty:
            continue

        best_row = group.iloc[0]
        frac = best_row["frac"]
        same_frac_rows = group[group["frac"] == frac]

        rows_to_process = [("best", best_row)]
        if len(same_frac_rows) > 1:
            worst_row = same_frac_rows.iloc[-1]
            rows_to_process.append(("worst", worst_row))

        for umap_type, row in rows_to_process:
            try:
                Z = pca_results.get(frac, {}).get("Z_total")
                if Z is None or Z.shape[0] < 3:
                    raise ValueError("Invalid or too small PCA input")

                t0 = time.time()
                Z_gpu = cp.ascontiguousarray(Z.astype(cp.float32))
                Z_reduced = PCA(n_components=50).fit_transform(Z_gpu) if Z_gpu.shape[1] > 50 else Z_gpu
                Z_umap = UMAP(n_components=2, random_state=42).fit_transform(Z_reduced)
                Z_umap_host = cp.asnumpy(Z_umap)
                cp.get_default_memory_pool().free_all_blocks()
                cp.cuda.runtime.deviceSynchronize()
                elapsed = round(time.time() - t0, 4)

                # Store timing
                job.stats.setdefault("embedding", {}).setdefault(bootstrap, {}).setdefault(label, {}).setdefault(frac, {})[umap_type] = elapsed

                # Determine hierarchy depth
                scope_idx = INDEX_COLS.index(scope) if scope in INDEX_COLS else -1
                active_levels = INDEX_COLS[:scope_idx + 1]

                for i, (x_val, y_val) in enumerate(Z_umap_host):
                    row_data = {
                        "DoE_UUID": uuid,
                        "bootstrap": bootstrap,
                        "frac": frac,
                        "scope": scope,
                        "label": label,
                        "x": float(x_val),
                        "y": float(y_val),
                        "umap_type": umap_type,
                    }
                    for col in active_levels:
                        idx = encoder.get(col)
                        row_data[col] = int(data_num[i, idx]) if idx is not None else None

                    rows.append(row_data)

            except Exception as e:
                error_msg = f"UMAP-{umap_type} failed: {e}"
                logging.warning(f"[UMAP-{umap_type}] '{label}' frac={frac:.2f}: {e}")
                fail_trail.mark_validation(
                    bootstrap=bootstrap,
                    label=label,
                    frac=frac,
                    model=f"umap_{umap_type}",
                    status="failed",
                    error=error_msg
                )

    if not rows:
        logging.debug1(f"[UMAP] No embeddings extracted for job {job.job_uuid}")
        return None

    df_result = pd.DataFrame(rows)

    # Ensure all index/hierarchy columns exist
    for col in INDEX_COLS:
        if col not in df_result.columns:
            df_result[col] = None

    return df_result
