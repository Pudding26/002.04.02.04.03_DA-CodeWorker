from __future__ import annotations
import logging, json
import pandas as pd
from typing import List
from uuid import uuid4
from datetime import datetime
import hashlib

from sqlalchemy.orm import Session

from app.utils.general.HelperFunctions import add_hashed_uuid_column


from app.utils.dataModels.Jobs.JobEnums import JobKind, JobStatus

from app.utils.dataModels.Jobs.ProviderJob import (
    ProviderJob, ProviderJobInput, ProviderAttrs
)



from app.utils.SQL.models.jobs.api_WorkerJobs import WorkerJobs_Out


class TA30_A_ProviderJobBuilder:
    """
    Build WorkerJobs from a long-form DataFrame produced by the
    wrapper's `expand_jobs_via_filters` and push them into SQL.
    """

    HDF5_PATH_MAP = {
        "DS01": "data/rawData/primary/DS01.hdf5",
        "DS04": "data/rawData/primary/DS04.hdf5",
        "DS07": "data/rawData/primary/DS07.hdf5",
        "DS11": "https://iiif-images.lib.ncsu.edu/iiif/2/insidewood-{id}/full/full/0/default.jpg",
    }

    @classmethod
    def build(cls, job_df: pd.DataFrame, jobs) -> None:
        if job_df.empty:
            logging.info("[ProviderBuilder] Nothing to build.")
            return

        # ------------------------------------------------------------------
        # Avoid creating duplicate jobs (job_uuid is the primary key)
        # ------------------------------------------------------------------
        new_rows: List[dict] = []

        job_df['digitizedDate'] = (
            pd.to_datetime(job_df['digitizedDate'], errors='coerce')
            .dt.strftime('%Y-%m-%dT%H:%M:%S')
            .fillna('unknown')
        )
        
      
        job_df["job_uuid"] = job_df["sampleID"].apply(
            lambda val: "provider_" + hashlib.sha1(str(val).encode()).hexdigest()[:10]
        )

        existing = WorkerJobs_Out.fetch_distinct_values(column="job_uuid")  # returns set[str] #must be actual df, filter against col must be parsed in update func as well
        to_create = []
        to_update = []
        all_jobs = []

        job_df = job_df.iloc[:30]

        for row_no, row in job_df.reset_index(drop=True).iterrows():
            
            job_uuid = row["job_uuid"]

            rel_paths = row["sourceFilePath_rel"]
            if isinstance(rel_paths, str):
                rel_paths = json.loads(rel_paths)               # DS01/04/07 are stored as JSON text

            job = ProviderJob(
                
                
                job_uuid= job_uuid,
                parent_job_uuids=row["parent_job_uuids"],

                status = JobStatus.READY.value,
                job_type = JobKind.PROVIDER.value,
                input=ProviderJobInput(
                    src_file_path=cls.HDF5_PATH_MAP[row["sourceNo"]],
                    src_ds_rel_path=rel_paths,
                    stored_locally=row["sourceStoredLocally"],
                    dest_rel_path=row["hdf5_dataset_path"],
                ),
                attrs=ProviderAttrs(
                    # ------------ identical to legacy logic ------------
                    Level1 = {"woodType": row["woodType"]},
                    Level2 = {"family": row["family"], "genus": row["genus"]},
                    Level3 = {"genus":  row["genus"]},
                    Level4 = {
                        "species": row.get("species") or "unknown",
                        "engName": row.get("engName") or "unknown",
                        "deName":  row.get("deName")  or "unknown",
                        "frName":  row.get("frName")  or "unknown",
                        "japName": row.get("japName") or "unknown",
                    },
                    Level5 = {
                        "sourceID": row["sourceID"],
                        "sourceNo": row["sourceNo"],
                    },

                    Level6 = {
                        "specimenID": row["specimenID"],
                        "microscopicTechnic": row.get("microscopicTechnic") or "unknown",
                        "institution":       row.get("institution")       or "unknown",
                        "institutionCode":   row.get("institutionCode")   or "unknown",
                        "contributor":       row.get("contributor")       or "unknown",
                        "citeKey":           row.get("citeKey")           or "unknown",
                        "IFAW_code":         row.get("IFAW_code")         or "unknown",
                        "samplingPoint":     row.get("samplingPoint")     or "unknown",
                        "origin":            row.get("origin")            or "unknown",
                    },
                    Level7 = {
                        "sampleID":        row["sampleID"],
                        #"digitizedDate":   row.get("digitizedDate"),
                        "view":            row.get("view")            or "unknown",
                        "lens":            row.get("lens")            or "unknown",
                        "totalNumberShots":row.get("totalNumberShots") or "unknown",
                    },
                    dataSet_attrs = {
                        "pixelSize_um_per_pixel": row.get("pixelSize_um_per_pixel") or "unknown",
                        "DPI":           row.get("DPI")            or "unknown",
                        "area_x_mm":     row.get("area_x_mm")      or "unknown",
                        "area_y_mm":     row.get("area_y_mm")      or "unknown",
                        "numericalAperature_NA": row.get("numericalAperature_NA") or "unknown",
                        "GPS_Alt":  row.get("GPS_Alt"),
                        "GPS_Lat":  row.get("GPS_Lat"),
                        "GPS_Long": row.get("GPS_Long"),
                        #"digitizedDate": row.get("digitizedDate"),
                        "raw_UUID": ", ".join(row["raw_UUID"]) if isinstance(row["raw_UUID"], list) else str(row["raw_UUID"]),
                    },
                ),
                )


            all_jobs.append(job)
            if job_uuid in existing:
                to_update.append(job)

            else:
                to_create.append(job)
                logging.debug2(f"[ProviderBuilder] Adding job {job_uuid} ({row_no + 1}/{len(job_df)})")


        from app.tasks.TA30_JobBuilder.TA30_0_JobBuilderWrapper import TA30_0_JobBuilderWrapper
            
        TA30_0_JobBuilderWrapper.store_and_update(to_create = to_create, to_update = to_update)
                
 


