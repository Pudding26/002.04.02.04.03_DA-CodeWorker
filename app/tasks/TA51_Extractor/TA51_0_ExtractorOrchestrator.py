# -----------------------------------------------------------------
# 3.  TA51_0_ExtractorOrchestrator.py
# -----------------------------------------------------------------
"""Batch runner that wires the two GPU modules together.

Public flow identical to legacy orchestrator – entry‑point is
``TA51_0_ExtractorOrchestrator().run(job_df_raw)`` which internally
calls ``_create_job_list`` then ``_run_pipeline``.  Improvements:

* **Two CUDA streams** (copy & compute) so host↔device transfers overlap
  with region‑prop kernels (no more explicit *synchronise* per batch).
* Power‑of‑two padding & 3‑D extraction in the FeatureExtractor class
  almost eliminates kernel‑launch overhead → GPU utilisation ≫ 50 %.
"""
from __future__ import annotations
import time, logging
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict, deque
import numpy as np


import cupy as cp
import cupyx.profiler as profiler
from cupyx.scipy.ndimage import label
from cucim.skimage.measure import regionprops_table




import numpy as np, cupy as cp, pandas as pd, cudf

# project‑internal imports (unchanged import paths)
from app.utils.common.app.utils.dataModels.Jobs.ExtractorJob import (
    ExtractorJob,
    ExtractorJobInput,
)
#from app.tasks.TA51_Extractor.TA51_A_FeatureExtractor_GPU import TA51_A_FeatureExtractor_GPU
#from app.tasks.TA51_Extractor.TA51_B_FeatureProcessor_GPU import TA51_B_FeatureProcessor_GPU


class TA51_0_ExtractorOrchestrator:  # same class name ✔︎
    BATCH_SIZE = 32
    OUT_DIR = Path("./data/features")

    # --------------------------------------------------------------
    # public API (unchanged)
    # --------------------------------------------------------------
    def run(self, job_df_raw: pd.DataFrame) -> None:  # noqa: N802 – keep legacy name
        self.OUT_DIR.mkdir(parents=True, exist_ok=True)
        self.jobs = self._create_job_list(job_df_raw)


        
        
        self._run_pipeline()


    # --------------------------------------------------------------
    # internals
    # --------------------------------------------------------------
    def _create_job_list(self, job_df_raw: pd.DataFrame) -> List[ExtractorJob]:  # noqa: N802
        jobs = []
        for _, row in job_df_raw.iterrows():
            payload: Dict[str, Any] = row["payload"]["input"]
            inp = ExtractorJobInput(
                mask=payload.get("mask"),
                n_images=payload["n_images"],
                width=payload["width"],
                height=payload["height"],
                stackID=payload["stackID"],
            )
            inp.mask = inp.get_mask()
            jobs.append(
                ExtractorJob(
                    job_uuid=row["job_uuid"],
                    job_type=row["job_type"],
                    status=row["status"],
                    attempts=row["attempts"],
                    next_retry=row["next_retry"],
                    created=row["created"],
                    updated=row["updated"],
                    parent_job_uuids=row["parent_job_uuids"],
                    input=inp,
                )
            )
        return jobs

    def create_large_batch(self, jobs, BATCH_SIZE=32):
        def _deal_with_leftovers(batches, queue):
            for shape, data in batches.items():
                if data["masks"]:
                    queue.append(
                        {
                            "masks": np.stack(data["masks"], axis=0),
                            "shot_ids": list(data["shot_ids"]),
                            "shape": shape,
                        }
                    )
        queue = deque()
        batches = defaultdict(lambda: {"masks": [], "shot_ids": []})
        added_masks_no = 0
        shot_counters = defaultdict(int)  # counter per stack_id

        for job_no, job in enumerate(jobs):
            stack_id = job.input.stackID
            if job_no % 10 == 0:
                logging.debug2(
                    f"Processing job {job_no}, added a total of {added_masks_no} masks so far"
                )

            for mask in job.input.mask:
                added_masks_no += 1
                shape = mask.shape
                shot_counters[stack_id] += 1
                shot_id = f"{stack_id}_{shot_counters[stack_id]:03d}"
                
                batches[shape]["masks"].append(mask)
                batches[shape]["shot_ids"].append(shot_id)
                
                
                
                if len(batches[shape]["masks"]) >= BATCH_SIZE:
                    # Emit this full batch
                    queue.append(
                        {
                            "masks": np.stack(batches[shape]["masks"], axis=0),
                            "shot_ids": list(batches[shape]["shot_ids"]),
                            "shape": shape,
                        }
                    )
                    # reset this batch
                    batches[shape]["masks"].clear()
                    batches[shape]["shot_ids"].clear()

        # Handle leftover masks
        _deal_with_leftovers(batches, queue)
        return queue

    def extract_regionprops_from_canvas(self, batch, props=None):
        import time
        import cupy as cp
        import cudf
        from cupyx.scipy.ndimage import label
        from cucim.skimage.measure import regionprops_table
        from pathlib import Path
        import pandas as pd
        from datetime import datetime
        import pylibraft


        masks = batch["masks"]
        shot_ids = batch["shot_ids"]
        B, h, w = masks.shape

        logging.debug2(f"Starting extract_regionprops_from_canvas for batch of {B} masks of size {h}x{w}")
        t_start = time.perf_counter()

        # Push masks to GPU
        t0 = time.perf_counter()
        masks_gpu = cp.asarray(masks, dtype=cp.uint8)
        copy_time_ms = (t0 - t_start) * 1000.0
        logging.debug2(f"Copied masks to GPU in {copy_time_ms:.2f} ms")

        # Properties to extract
        base_props = props or [
            "label",
            "area",
            "major_axis_length",
            "solidity",
            "centroid",
            "eccentricity",
            "feret_diameter_max",
            "moments_hu",
            "perimeter",
            "orientation"

        ]
        logging.debug2(f"Proceeding to extract these properties: {base_props}")

        # Prepare per-mask output
        dfs = []
        streams = [cp.cuda.Stream(non_blocking=True) for _ in range(B)]

        # Label & regionprops
        for i in range(B):
            with streams[i]:
                lbl, num = label(masks_gpu[i], structure=cp.ones((3, 3), dtype=cp.uint8))
                props_table = regionprops_table(lbl, properties=base_props)
                dfs.append((props_table, shot_ids[i]))

        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        labeling_time_ms = (t1 - t0) * 1000.0
        logging.debug2(f"GPU labeling + regionprops_table took {labeling_time_ms:.2f} ms")

        # Concatenate into one cudf.DataFrame
        all_df = []
        for props_table, shot_id in dfs:
            df = cudf.DataFrame({k: v for k, v in props_table.items()})
            df["shot_id"] = shot_id
            all_df.append(df)

        df_full = cudf.concat(all_df, ignore_index=True)
        t2 = time.perf_counter()
        concat_time_ms = (t2 - t1) * 1000.0
        logging.debug2(f"cuDF concatenation took {concat_time_ms:.2f} ms")

        total_time_ms = (t2 - t_start) * 1000.0
        logging.debug2(
            f"Completed extract_regionprops_from_canvas in {total_time_ms:.2f} ms for {B} masks"
        )



        # 🎯 Append performance log to CSV
        log_path = Path("./performance_log.csv")
        record = {
            "timestamp": datetime.now().isoformat(),
            "num_masks": B,
            "image_height": h,
            "image_width": w,
            "props_extracted": ";".join(base_props),
            "copy_time_ms": copy_time_ms,
            "gpu_labeling_regionprops_time_ms": labeling_time_ms,
            "cudf_concat_time_ms": concat_time_ms,
            "total_time_ms": total_time_ms,
        }
        df_record = pd.DataFrame([record])
        file_exists = log_path.exists()
        df_record.to_csv(log_path, mode="a", index=False, header=not file_exists)

        return df_full


    def extract_regionprops_from_canvas_cpu(self, batch, props=None):
        import time
        import pandas as pd
        from skimage.measure import label, regionprops_table
        from pathlib import Path
        from datetime import datetime

        masks = batch["masks"]
        shot_ids = batch["shot_ids"]
        B, h, w = masks.shape

        logging.debug2(f"Starting extract_regionprops_from_canvas_cpu for batch of {B} masks of size {h}x{w}")
        t_start = time.perf_counter()

        # Dummy copy time just for logging consistency
        copy_time_ms = 0.0

        # Properties to extract
        base_props = props or [
            "label",
            "area",
            "major_axis_length",
            "solidity",
            "centroid",
            "eccentricity",
            "feret_diameter_max",
            "moments_hu",
            "perimeter",
            "orientation",
        ]
        logging.debug2(f"Proceeding to extract these properties: {base_props}")

        # Label + regionprops_table
        dfs = []
        labeling_start = time.perf_counter()
        for i in range(B):
            lbl = label(masks[i])
            props_table = regionprops_table(lbl, properties=base_props)
            df = pd.DataFrame(props_table)
            df["shot_id"] = shot_ids[i]
            dfs.append(df)
        labeling_time_ms = (time.perf_counter() - labeling_start) * 1000.0
        logging.debug2(f"Labeling + regionprops_table took {labeling_time_ms:.2f} ms")

        # Concatenate
        concat_start = time.perf_counter()
        df_full = pd.concat(dfs, ignore_index=True)
        concat_time_ms = (time.perf_counter() - concat_start) * 1000.0
        logging.debug2(f"Pandas concatenation took {concat_time_ms:.2f} ms")

        total_time_ms = (time.perf_counter() - t_start) * 1000.0
        logging.debug2(
            f"Completed extract_regionprops_from_canvas_cpu in {total_time_ms:.2f} ms for {B} masks"
        )

        # Append performance log to CSV
        log_path = Path("./performance_log.csv")
        record = {
            "timestamp": datetime.now().isoformat(),
            "num_masks": B,
            "image_height": h,
            "image_width": w,
            "props_extracted": ";".join(base_props),
            "copy_time_ms": copy_time_ms,
            "gpu_labeling_regionprops_time_ms": labeling_time_ms,  # Keep column name consistent
            "cudf_concat_time_ms": concat_time_ms,
            "total_time_ms": total_time_ms,
        }
        df_record = pd.DataFrame([record])
        file_exists = log_path.exists()
        df_record.to_csv(log_path, mode="a", index=False, header=not file_exists)

        return df_full



    def _run_pipeline(self) -> None:  # noqa: N802
        
        
        job_queue = self.create_large_batch(jobs = self.jobs, BATCH_SIZE=16)tea        self.extract_regionprops_from_canvas_cpu(job_queue[0])

        df_raw = self.extract_regionprops_from_canvas(job_queue[0])


        pass


    def old():
        fe = TA51_A_FeatureExtractor_GPU()
        fp = TA51_B_FeatureProcessor_GPU()
        batch_masks: List[np.ndarray] = []
        batch_meta: List[Dict[str, str]] = []
        shot_tables: Dict[str, List[cudf.DataFrame]] = {}

        copy_stream = cp.cuda.Stream(non_blocking=True)
        compute_stream = cp.cuda.Stream(non_blocking=True)

        extract_ms = bin_ms = 0.0
        shots_done = stacks_done = 0

        for job in self.jobs:
            stack_id = job.input.stackID
            for shot_no, mask in enumerate(job.input.mask, start=1):
                batch_masks.append(mask)
                batch_meta.append({"stackID": stack_id, "shotID": f"{stack_id}_{shot_no:03d}"})
                shots_done += 1
                if len(batch_masks) >= self.BATCH_SIZE:
                    extract_ms += self._extract_and_store(
                        fe, batch_masks, batch_meta, shot_tables, copy_stream, compute_stream
                    )
                    batch_masks.clear()
                    batch_meta.clear()
            stacks_done += 1

        if batch_masks:
            extract_ms += self._extract_and_store(
                fe, batch_masks, batch_meta, shot_tables, copy_stream, compute_stream
            )

        # -------------------- process whole stacks --------------------
        for sid, dfs in shot_tables.items():
            t0 = time.perf_counter()
            result_df = fp.process(stackID=sid, feature_tables=dfs)
            result_df.to_parquet(self.OUT_DIR / f"{sid}.parquet")
            bin_ms += (time.perf_counter() - t0) * 1000

        logging.info(
            f"🟢 {shots_done} shots / {stacks_done} stacks\n"
            f"    extraction {extract_ms/1e3:.2f}s ({extract_ms/max(shots_done,1):.2f} ms/shot)\n"
            f"    binning    {bin_ms/1e3:.2f}s ({bin_ms/max(stacks_done,1):.2f} ms/stack)"
        )

    # ----------------------------------------------------------------
    def _extract_and_store(
        self,
        fe: TA51_A_FeatureExtractor_GPU,
        masks: List[np.ndarray],
        meta: List[Dict[str, str]],
        shot_tables: Dict[str, List[cudf.DataFrame]],
        copy_stream: cp.cuda.Stream,
        compute_stream: cp.cuda.Stream,
    ) -> float:  # noqa: N802
        t0 = time.perf_counter()

        # ---- async copy ------------------------------------------------
        with copy_stream:
            gpu_masks = [cp.asarray(m, dtype=cp.uint8) for m in masks]
        copy_stream.synchronize()  # ensure visible to compute stream

        # ---- async compute --------------------------------------------
        with compute_stream:
            dfs = fe.apply_batch_gpu(gpu_masks)
        compute_stream.synchronize()
        dur = (time.perf_counter() - t0) * 1000

        # ---- store per‑shot tables ------------------------------------
        for df, m in zip(dfs, meta, strict=True):
            df["stackID"], df["shotID"] = m["stackID"], m["shotID"]
            shot_tables.setdefault(m["stackID"], []).append(df)
        return dur


def regionprops_batch(masks: cp.ndarray):
    B, H, W = masks.shape
    canvas = masks.transpose(1, 0, 2).reshape(H, B * W)  # tiled canvas
    lbl, num = label(canvas, structure=cp.ones((3,3), dtype=cp.uint8))  # GPU labeling
    props = regionprops_table(
        lbl,
        properties=("label", "area",)
    )
    return props  # stays on GPU

