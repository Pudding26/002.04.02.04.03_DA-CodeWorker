import logging

from typing import Any

from sqlalchemy import event, delete, insert, update, select, func
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Mapper



from sqlalchemy import case

from app.utils.common.app.utils.dataModels.Jobs.JobEnums import JobStatus, JobKind, RelationState

from app.utils.common.app.utils.SQL.models.jobs.orm_WorkerJobs import orm_WorkerJobs
from app.utils.common.app.utils.SQL.models.jobs.orm_JobLink import orm_JobLink
from app.utils.common.app.utils.SQL.models.jobs.orm_DoEJobs import orm_DoEJobs


def _status_to_rel_state(status: str) -> RelationState:
    """
    Maps a JobStatus (TODO, FAILED, DONE) to a RelationState for JobLink rows.
    """
    if status == JobStatus.TODO:
        return RelationState.IN_PROGRESS
    elif status == JobStatus.FAILED:
        return RelationState.FAILED
    else:
        return RelationState.FREE


@event.listens_for(orm_WorkerJobs, "after_insert")
@event.listens_for(orm_WorkerJobs, "after_update")
def sync_workerjob_links(mapper: Mapper, conn: Connection, target: orm_WorkerJobs) -> None:
    """
    Generic link sync logic for all WorkerJobs (provider, segmenter, etc.).
    """
    #logging.debug2(f"Syncing links for WorkerJob {target.job_uuid} of type {target.job_type.value}")
    if not target.parent_job_uuids:
        return

    job_type = target.job_type.value
    rel_state = _status_to_rel_state(target.status)

    # Step 1: Remove old links for this job
    conn.execute(
        delete(orm_JobLink).where(
            orm_JobLink.child_uuid == target.job_uuid,
            orm_JobLink.child_kind == job_type,
        )
    )


    # Step 2: Insert new links
    rows = [
        {
            "parent_uuid": str(parent_id),
            "child_uuid": target.job_uuid,
            "child_kind": job_type,
            "rel_state": rel_state
        }
        for parent_id in target.parent_job_uuids
    ]
    conn.execute(insert(orm_JobLink), rows)

    # Step 3: Roll up for each parent DoEJob
    for parent_id in target.parent_job_uuids:
        _roll_up_child_status(
            conn=conn,
            parent_uuid=parent_id,
            child_kind=job_type
        )


def _roll_up_child_status(conn: Connection, parent_uuid: str, child_kind: str) -> None:
    """
    Generic version of provider roll-up that updates the correct status field on orm_DoEJobs.
    """
    #logging.debug2(f"Roll up triggered for parent {parent_uuid} with child kind {child_kind}")

    # Count child job states
    total, todo, failed = conn.execute(
        select(
            func.count(),
            func.sum(case((orm_JobLink.rel_state == RelationState.IN_PROGRESS, 1), else_=0)),
            func.sum(case((orm_JobLink.rel_state == RelationState.FAILED, 1), else_=0)),
        ).where(
            orm_JobLink.parent_uuid == parent_uuid,
            orm_JobLink.child_kind == child_kind
        )
    ).one()

    # Determine new status
    if total == 0:
        new_status = JobStatus.DONE
    elif todo > 0:
        new_status = JobStatus.TODO
    elif failed / total > 0.3:
        new_status = JobStatus.FAILED
    else:
        new_status = JobStatus.DONE

    # Dynamically resolve the column name on orm_DoEJobs (e.g. "provider_status")
    status_column_name = f"{child_kind}_status"
    status_column = getattr(orm_DoEJobs, status_column_name, None)

    if not status_column:
        raise ValueError(f"No status field '{status_column_name}' on orm_DoEJobs")

    # Update status
    conn.execute(
        update(orm_DoEJobs)
        .where(orm_DoEJobs.job_uuid == parent_uuid)
        .values({status_column_name: new_status})
    )
