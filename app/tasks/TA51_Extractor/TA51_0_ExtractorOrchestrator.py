from app.utils.common.app.utils.dataModels.Jobs.ExtractorJob import ExtractorJob, ExtractorJobInput


from app.utils.common.app.tasks.TA41_ImageSegmentation.FeatureExtractor import FeatureExtractor
from app.tasks.TA51_Extractor.TA51_B_FeatureProcessor_GPU import TA51_B_FeatureProcessor_GPU


import logging
import pandas as pd
import cupy as cp
import time

class TA51_0_ExtractorOrchestrator:
    """
    This class orchestrates the extraction process for TA51 data.
    It initializes the extractor and manages the extraction workflow.
    """

    def __init__(self):
        """
        Initializes the TA51_0_ExtractorOrchestrator with a given extractor.

        :param extractor: An instance of the extractor to be used for data extraction.
        """
        pass

    def run(self, job_df_raw):
        """
        Runs the extraction process using the initialized extractor.
        """
        self.jobs = self.create_job_list(job_df_raw=job_df_raw)

        self.run_pipeline()

    



    def create_job_list(self, job_df_raw):

        """
        Converts a DataFrame of raw job data into a list of ExtractorJob instances.

        :param job_df_raw: DataFrame containing raw job data.
        :return: List of ExtractorJob instances.
        """
        jobs = []
        for index, row in job_df_raw.iterrows():
            payload = row["payload"]["input"]

            job_input = ExtractorJobInput(
                mask=payload.get("mask"),           # can be None
                n_images=payload["n_images"],
                width=payload["width"],
                height=payload["height"],
                stackID=payload["stackID"],
            )

            job_input.mask = job_input.get_mask()

            job = ExtractorJob(
                job_uuid=row['job_uuid'],
                job_type=row['job_type'],
                status=row['status'],
                attempts=row['attempts'],
                next_retry=row['next_retry'],
                created=row['created'],
                updated=row['updated'],
                parent_job_uuids=row['parent_job_uuids'],  # list[str] per your schema
                input=job_input
            )

            jobs.append(job)

        return jobs
    
    def run_pipeline(self):

        # -----------------------------------------------------------------------------
        # SET-UP
        # -----------------------------------------------------------------------------
        fe          = FeatureExtractor()
        feature_proc = TA51_B_FeatureProcessor_GPU()

        FLUSH_EVERY_STACKS = 10  # tune
        shot_tables: dict[str, list[pd.DataFrame]] = {}

        # timers
        extract_ms  = 0.0  # cumulative feature-extraction time
        bin_ms      = 0.0  # cumulative binning / summarising time
        shots_done  = 0
        stacks_done = 0

        # -----------------------------------------------------------------------------
        # MAIN LOOP
        # -----------------------------------------------------------------------------
        for job_no, job in enumerate(self.jobs):
            stack_id   = job.input.stackID
            mask_stack = job.input.mask

            for shot_no, mask in enumerate(mask_stack, start=1):
                t0 = time.perf_counter()
                df = fe.apply_one(mask, connectivity=2, use_gpu=True)
                cp.cuda.Device(0).synchronize()        # ensure GPU finished
                extract_ms += (time.perf_counter() - t0) * 1000

                # annotate
                df["stackID"] = stack_id
                df["shotID"]  = f"{stack_id}_{shot_no:03d}"
                shot_tables.setdefault(stack_id, []).append(df)
                shots_done += 1

            stacks_done += 1

            # flush in blocks to control VRAM
            if len(shot_tables) >= FLUSH_EVERY_STACKS:
                t0 = time.perf_counter()
                for sid, dfs in shot_tables.items():
                    result_df = feature_proc.process(stackID=sid, feature_tables=dfs)
                    result_df.to_parquet(f"/data/features/{sid}.parquet")
                cp.get_default_memory_pool().free_all_blocks()
                bin_ms += (time.perf_counter() - t0) * 1000
                shot_tables.clear()

        # final flush
        if shot_tables:
            t0 = time.perf_counter()
            for sid, dfs in shot_tables.items():
                result_df = feature_proc.process(stackID=sid, feature_tables=dfs)
                result_df.to_parquet(f"/data/features/{sid}.parquet")
            cp.get_default_memory_pool().free_all_blocks()
            bin_ms += (time.perf_counter() - t0) * 1000
            shot_tables.clear()

        # -----------------------------------------------------------------------------
        # SUMMARY
        # -----------------------------------------------------------------------------
        print(
            f"🟢   Finished {shots_done:,} shots in {stacks_done:,} stacks\n"
            f"    ├─ Extraction: {extract_ms/1e3:8.2f} s total "
            f"({extract_ms/shots_done:6.2f} ms/shot)\n"
            f"    └─ Binning   : {bin_ms/1e3:8.2f} s total "
            f"({bin_ms/stacks_done:6.2f} ms/stack)"
        )