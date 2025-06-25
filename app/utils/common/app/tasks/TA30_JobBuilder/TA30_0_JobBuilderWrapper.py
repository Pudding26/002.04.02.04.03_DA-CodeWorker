import logging
import os
import pandas as pd
from typing import List
import yaml
import random
import uuid
import time
import logging
from datetime import datetime, timezone

import warnings

from sqlalchemy.orm import object_session
from sqlalchemy import text, update, func, bindparam, String
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Session

from app.utils.common.app.utils.SQL.DBEngine import DBEngine

from app.tasks.TaskBase import TaskBase
from app.utils.common.app.utils.SQL.SQL_Df import SQL_Df
from app.utils.common.app.utils.SQL.SQL_Dict import SQL_Dict
from app.utils.common.app.utils.SQL.models.production.api.api_WoodTableA import WoodTableA_Out
from app.utils.common.app.utils.SQL.models.production.api.api_WoodTableB import WoodTableB_Out


from app.utils.common.app.utils.SQL.models.production.api.api_WoodMaster import WoodMaster_Out
from app.utils.common.app.utils.SQL.models.production.api.api_WoodMasterPotential import WoodMasterPotential_Out
from app.utils.common.app.utils.SQL.models.jobs.api_DoEJobs import DoEJobs_Out

from app.utils.common.app.utils.SQL.models.jobs.orm_WorkerJobs import orm_WorkerJobs


from app.utils.common.app.utils.SQL.models.production.api.api_ModellingResults import ModellingResults_Out


from app.utils.common.app.utils.dataModels.FilterModel.FilterModel import FilterModel
from app.utils.common.app.utils.dataModels.FilterModel.FilterModel import Border
from app.utils.common.app.utils.dataModels.Jobs.DoEJob import DoEJob


from app.tasks.TA30_JobBuilder.TA30_A_ProviderJobBuilder import TA30_A_ProviderJobBuilder
from app.tasks.TA30_JobBuilder.TA30_B_SegmenterJobBuilder import TA30_B_SegmenterJobBuilder
from app.tasks.TA30_JobBuilder.TA30_C_ExtractorJobBuilder import TA30_C_ExtractorJobBuilder
#from app.tasks.TA30_JobBuilder.TA30_C_ModelerJobBuilder import TA30_C_ModelerJobBuilder


#from app.utils.common.app.utils.SQL.models.temp.api.SegmentationJobs_out import SegmentationJobs_out


class TA30_0_JobBuilderWrapper(TaskBase):
    def setup(self):
        self.controller.update_message("Initializing Segmentation Job Builder")
        self.controller.update_progress(0.01)
        self.woodmaster_df = pd.DataFrame()
        self.general_job_df = pd.DataFrame()
        self.filtered_jobs_df = pd.DataFrame()
        self.segmentation_jobs = []
        logging.info("[TA30_A] Setup complete.")

        self.FOLLOW_UP_STEPS = [
            ("provider_status",  "segmenter_status"),
            ("segmenter_status", "extractor_status"),
            ("extractor_status", "modeler_status"),
            ("modeler_status",   "validator_status"),
        ]



    def run(self):
        try:
            logging.info("[TA30] Starting infinite job builder loop")
            loop = True
            loop_no = 0 
            while loop == True:
                loop_no += 1
                loop_start = datetime.now(timezone.utc)
                loop = False
                logging.info(f"[TA30] Job loop started at {loop_start.isoformat()}")

                self.controller.update_message("Scanning for new DoE Jobs")
                builders = ["provider", "segmenter", "extractor", "modeler"]

                for b in builders:
                    _unblock_follow_up_tasks(session = DBEngine("jobs").get_session(), FOLLOW_UP_STEPS = self.FOLLOW_UP_STEPS, loop_no = loop_no)
                    
                    
                    status_col = f"{b}_status"

                    include_cols = [
                        "sourceNo", "woodType", "family", "genus", "species",
                        "view", "lens", "noShotsRange", "maxShots", "filterNo"
                    ]


                    if "totalNumberShots" not in include_cols:
                        include_cols.append("totalNumberShots")
                    if "noShotsRange" in include_cols:
                        include_cols.remove("noShotsRange")

                    match b:
                        case "provider":
                            BuilderClass = TA30_A_ProviderJobBuilder
                            groupby_col = "sampleID"
                            id_field = "job_uuid"
                            if "maxShots" in include_cols:
                                include_cols.remove("maxShots")
                            if "filterNo" in include_cols:
                                include_cols.remove("filterNo")


                            filter_model = FilterModel.from_human_filter({"contains": {"provider_status": "ready"}})
                            filter_table = WoodMasterPotential_Out
                        case "segmenter":
                            groupby_col = "stackID"
                            if "maxShots" in include_cols:
                                include_cols.remove("maxShots")
                            if "filterNo" in include_cols: # later distinction if it is in then we can mark as done, either in jobbuilder or in the segmenter state, have to decide
                                include_cols.remove("filterNo")
                            id_field = "job_uuid"

                            BuilderClass = TA30_B_SegmenterJobBuilder

                            filter_model = FilterModel.from_human_filter({"contains": {"segmenter_status": "ready"}})
                            filter_table = WoodMaster_Out
                        case "extractor":
                            BuilderClass = TA30_C_ExtractorJobBuilder
                            TA30_C_ExtractorJobBuilder.build()
                            continue




                        case _:
                            continue  # Skip unsupported builders for now
                        #case "modeler":
                        #    BuilderClass = TA30_C_ModelerJobBuilder

                    self.controller.update_message(f"Checking {b} jobs (status: ready)")
                    logging.debug5(f"[TA30] Starting builder loop for: {b}")
                    raw_df = DoEJobs_Out.fetch(filter_model=filter_model)

                    raw_df = raw_df.drop(columns=[col for col in ["input", "attrs"] if col in raw_df.columns])


                    if raw_df.empty:
                        logging.debug(f"[TA30] No '{b}' jobs found.")
                        continue 

                    jobs = []
                    id_field = "job_uuid"
                    for _, row in raw_df.iterrows():
                        payload = row["payload"]
                        if id_field not in payload:
                            payload = payload.copy()  # avoid mutating original
                            payload[id_field] = row[id_field]
                        job = DoEJob.model_validate(payload)
                        jobs.append(job)

                    logging.info(f"[TA30] {len(jobs)} '{b}' jobs found in DB.")
    
                    job_df = self.expand_jobs_via_filters(
                        jobs,
                        include_cols=include_cols,
                        is_range_cols=["totalNumberShots"],
                        is_max=["maxShots"],
                        src_data_api=filter_table,
                        id_field=id_field,
                    )

                    if job_df.empty:
                        logging.warning(f"[TA30] No stack rows matched for {b} jobs.")
                        continue

                    logging.info(f"[TA30] Dispatching {len(job_df)} rows to {BuilderClass.__name__}")
                    BuilderClass.build(job_df, jobs)

                self.controller.update_message("Sleeping")
                logging.info("[TA30] Sleeping for 3 minutes to allow other tasks to process.")
                #time.sleep(180)
                time.sleep(10)

        except KeyboardInterrupt:
            logging.info("[TA30] Interrupted by user — shutting down gracefully.")
            self.controller.finalize_failure("Interrupted by user")
        except Exception as e:
            logging.exception("[TA30] JobBuilder task failed", exc_info=True)
            self.controller.finalize_failure(str(e))
            raise
        finally:
            self.cleanup()


    def cleanup(self):
        logging.info("[TA30_A] Running cleanup routine.")
        self.controller.archive_with_orm()


    def expand_jobs_via_filters(
        self,
        jobs: list,
        *,
        include_cols: list[str],
        is_range_cols: list[str] = None,
        is_max: list[str] = None,
        is_min: list[str] = None,
        border_rule: dict = None,
        global_logic: str = "and",
        src_data_api: WoodMaster_Out,
        id_field="job_uuid",
        groupby_col: str = "sampleID"
    ) -> pd.DataFrame:
        """
        Expand a list of Job models (e.g., DoEJob) into a long-form dataset
        using per-job FilterModels and SQL table fetching.

        Returns:
            DataFrame containing all matched rows with originating job_uuid
        """
        filter_df = jobs[0].__class__.to_filter_df(jobs)
        filter_df = filter_df.reset_index().rename(columns={"index": id_field})
        job_filters = FilterModel.from_dataframe(
            df=filter_df,
            include_cols=include_cols,
            is_range_cols=is_range_cols,
            is_max=is_max,
            is_min=is_min,
            border_rule=border_rule,
            global_logic=global_logic,
            job_id_field=id_field
        )

        dtypes = src_data_api.pydantic_model_to_dtype_dict()
        new_subset = None


        result_df = pd.DataFrame(columns=dtypes.keys()).astype(dtypes)
        total_jobs = len(job_filters)
        #job_filters = job_filters[200:300]
        end = False
        for i, filter_model in enumerate(job_filters):
            
            if end == True:
                logging.info("[Expand] Stopping job expansion due to end signal.")
                break
            
            if i % 100 == 0 or i == total_jobs - 1:
                logging.debug2(f"[Expand] Processing job {i+1}/{total_jobs}")
            
            del new_subset
            new_subset = pd.DataFrame(columns=dtypes.keys()).astype(dtypes)
            new_subset = new_subset.copy()
            with self.suppress_logging():
                new_subset = src_data_api.fetch(filter_model=filter_model, stream=False)
            
            new_subset["parent_job_uuids"] = filter_model.job_id
            new_subset = new_subset.copy()
            len_new = len(new_subset)
            len_before = len(result_df)

            if not new_subset.empty:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)
                    result_df = pd.concat([result_df, new_subset], ignore_index=True)

            len_after = len(result_df)
            delta = len_after - len_before

            if i % 100 == 0 or i == total_jobs - 1:
                ratio = delta / len_new if len_new > 0 else "N/A"
                logging.debug2(
                    f"[Expand] Job {i+1}/{total_jobs} Old: {len_before}, New: {len_new}, "
                    f"Combined: {len_after}, Added Ratio: {ratio}"
                )


        if not result_df.empty:
            group_cols = [col for col in result_df.columns if col != "parent_job_uuids"]
            result_df["parent_job_uuids"] = result_df["parent_job_uuids"].apply(lambda x: [x])

            result_df = (
                result_df.groupby(groupby_col, as_index=False)
                .agg({
                    **{col: "first" for col in group_cols if col != groupby_col},
                    "parent_job_uuids": lambda x: sorted(set(sum(x, [])))
                })
            )

        return result_df


    @staticmethod
    def store_and_update(to_create: list, to_update: list):
        if not to_create and not to_update:
            logging.info("No jobs to create or update.")
            return
        
        session: Session = DBEngine("jobs").get_session()
        created = updated = unchanged = 0
        start = time.time()
        try:
            # ---------------- Phase 1 – ORM insert/update ----------------
            for job in to_create:
                try:
                    session.add(orm_WorkerJobs(**job.to_sql_row()))
                    created += 1
                except Exception as e:
                    unchanged += 1
                    logging.warning(f"[Create] {job.job_uuid}: {e}")

            for job_no, job in enumerate(to_update):
                try:
                    job.update_timestamp()  # Ensure timestamps are fresh
                    row = job.to_sql_row()
                    
                    
                    
                    fields_to_update = {
                        "updated": job.updated,

                    }
                    session.query(orm_WorkerJobs).filter_by(job_uuid=job.job_uuid).update(
                        fields_to_update, synchronize_session=False
                    )
                    if job.parent_job_uuids:
                        stmt = text("""
                            UPDATE "WorkerJobs"
                            SET parent_job_uuids = (
                                SELECT jsonb_agg(DISTINCT elem)
                                FROM (
                                    SELECT jsonb_array_elements_text(parent_job_uuids) AS elem
                                    UNION
                                    SELECT unnest(:parent_job_uuids)
                                ) AS all_elems
                            )
                            WHERE job_uuid = :job_uuid
                        """).bindparams(
                            bindparam("parent_job_uuids", type_=ARRAY(String)),
                            bindparam("job_uuid", type_=String)
                        )
                        session.execute(
                            stmt,
                            {
                                "parent_job_uuids": job.parent_job_uuids,  # Must be a list of strings
                                "job_uuid": job.job_uuid,
                            }
                        )
                    updated += 1
                    if job_no % 500 == 0:
                        logging.debug2(f"[StoringWorkerJobs] for {job.job_type.value}  {job_no}/{len(to_update)} updated successfully.")
                except Exception as e:
                    unchanged += 1
                    logging.warning(f"[Update] {job.job_uuid}: {e}")

            session.commit()  # keep events for compatibility


            # ---------------- Phase 2 – batched JobLink sync -------------
            _bulk_upsert_joblinks_with_status(session, to_create + to_update)

            # ---------------- Phase 3 – status roll‑up ------------------
            kinds = {job.job_type for job in to_create + to_update}
            _roll_up_statuses(session, kinds)

            session.commit()

        except Exception as e:
            session.rollback()
            logging.exception("[JobStorage] transaction failed", exc_info=e)
        finally:
            session.close()

        logging.info(
            "Job storage summary → WorkerJobs:\n"
            f"  ✅ Created: {created}\n  🔁 Updated: {updated}\n  ⏭️ Unchanged/Failed: {unchanged}"
        )
        elapsed = time.time() - start
        logging.debug2(f"[Timing] Processed {len(to_create + to_update)} jobs in {elapsed:.2f} seconds ({elapsed/len(to_create + to_update):.3f} s/job)")


def _bulk_upsert_joblinks_with_status(session, jobs, batch_size: int = 5000):
    link_pairs: list[tuple[str, str]] = []
    for job in jobs:
        for p in job.parent_job_uuids:
            link_pairs.append((p, job.job_uuid))

    for i in range(0, len(link_pairs), batch_size):
        batch = link_pairs[i : i + batch_size]
        temp_table_name = f"temp_links_{uuid.uuid4().hex}"

        # 1) Create unique temp table
        session.execute(
            text(f"""
            CREATE TEMP TABLE {temp_table_name} (
                parent_uuid TEXT,
                child_uuid  TEXT
            ) ON COMMIT DROP;
            """)
        )

        # 2) Bulk insert into temp table
        session.execute(
            text(f"""
            INSERT INTO {temp_table_name} (parent_uuid, child_uuid)
            VALUES (:parent, :child)
            """),
            [{"parent": p, "child": c} for (p, c) in batch],  # batch is list of tuples!
        )

        # 3) Upsert into jobLink using the unique temp table
        session.execute(
            text(f"""
            INSERT INTO "jobLink" (parent_uuid, child_uuid, child_kind, rel_state)
            SELECT tl.parent_uuid,
                w.job_uuid,
                w.job_type,
                w.status
            FROM {temp_table_name} tl
            JOIN "WorkerJobs" w ON w.job_uuid = tl.child_uuid
            ON CONFLICT (parent_uuid, child_uuid) DO UPDATE
            SET rel_state = EXCLUDED.rel_state,
                child_kind = EXCLUDED.child_kind;
            """)
        )


def _roll_up_statuses(session: Session, child_kinds: set[str]):
    """Aggregate rel_state → update DoEJobs.<kind>_status in one pass per kind."""
    for kind in child_kinds:
        column = f"{kind.lower()}_status"
        session.execute(
            text(
                f"""
                UPDATE "DoEJobs" AS d
                SET {column} = sub.max_state
                FROM (
                    SELECT parent_uuid, MAX(rel_state) AS max_state
                    FROM "jobLink"
                    WHERE child_kind = :kind
                    GROUP BY parent_uuid
                ) AS sub
                WHERE d.job_uuid = sub.parent_uuid;
                """
            ),
            {"kind": kind},
        )


def _unblock_follow_up_tasks(session, FOLLOW_UP_STEPS, loop_no):
    """
    Promote the next pipeline column from 'blocked' → 'ready'
    whenever the previous stage is already 'done'.
    Runs once per job-builder loop; fully idempotent.
    """
    updates = []
    for done_col, next_col in FOLLOW_UP_STEPS:
        result = session.execute(
            text(f"""
                UPDATE "DoEJobs"
                SET    {next_col} = 'ready'
                WHERE  {done_col} = 'done'
                  AND  {next_col} = 'blocked';
            """)
        )
        changed = result.rowcount or 0
        if changed:
            updates.append(f"{changed} {done_col}→{next_col}")
    session.commit()

    if updates:
        logging.debug2(
            "[Loop %03d] Promoted: %s",
            loop_no, ", ".join(updates),
        )
