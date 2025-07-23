from fastapi import APIRouter, HTTPException, Request
import threading
import logging

from app.utils.common.app.utils.dataModels.FilterModel.FilterModel import FilterModel
from app.utils.common.app.utils.SQL.models.jobs.api_WorkerJobs import WorkerJobs_Out

from app.utils.lobby.LobbyHandler import LobbyHandler

lobby = LobbyHandler()
GPU_router = APIRouter()
worker_status = {
    "busy": False,
    "current_job": None,
    "queue_length": 0
}

MAX_JOBS_PER_WORKER = 5
job_queue = []

@GPU_router.post("/start-job")
async def start_job(request: Request):
    body = await request.json()
    job_type = body.get("job_type")
    payload = body.get("payload")

    if job_type != "modeler" or not payload or "job_uuids" not in payload:
        raise HTTPException(status_code=400, detail="Invalid payload")

    job_uuids = payload["job_uuids"]
    logging.info(f"📥 Received start-job request for {len(job_uuids)} jobs.")

    try:
        lobby.add_job(job_uuids)
    except Exception as e:
        logging.warning(f"⚠️ add_job() failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))

    return {"status": "accepted", "queued": len(job_uuids)}
