import threading
import logging
import os
import time
import requests
from datetime import datetime, timezone

from contextlib import contextmanager

from app.utils.hosts import ORCHESTRATOR_URL
from app.tasks.TA52_Modeler.utils.input_queue import input_queue
from app.tasks.TA52_Modeler.utils.job_loader import load_jobs
from app.tasks.TA52_Modeler.TA52_0_ModelerOrchestrator import TA52_0_ModelerOrchestrator

class LobbyHandler:
    def __init__(self):
        self.allow_jobs = True
        self.cpu_server_url = ORCHESTRATOR_URL
        self.worker_name = os.getenv("GPU_WORKER_NAME", "gpu-worker")
        self.start_time = time.time()
        self.max_lifetime_sec = int(os.getenv("MAX_WORKER_LIFETIME_SEC", 3600))
        self.last_time_job_added = None
        self.needs_work = 0
        self.added_jobs = 0

    def start(self):
        from app.tasks.TA52_Modeler.TA52_0_ModelerOrchestrator import TA52_0_ModelerOrchestrator
        self.orchestrator = TA52_0_ModelerOrchestrator()
        threading.Thread(target=self.orchestrator.run, daemon=True).start()
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()



    def add_job(self, job_uuids):

        
        self.added_jobs =+ len(job_uuids)
        if not self.allow_jobs:
            raise Exception("Worker not accepting new jobs")
        with suppress_logging(logging.ERROR):
           needs_work_iter, already_done_iter = load_jobs(job_uuids)  # Pass UUID list directly!
        self.last_time_job_added = datetime.now(timezone.utc)

        self.needs_work += needs_work_iter
        already_done = already_done_iter
        if self.needs_work % 10 == 0 or self.needs_work == 0:
            logging.debug2(f"[LOBBY] Added {self.added_jobs} jobs, {already_done} already done, {self.needs_work} need work.")



    def _heartbeat_loop(self):
        while True:
            self._register_self()
            time.sleep(30)

    def _register_self(self):
        try:
            elapsed = time.time() - self.start_time
            if elapsed > self.max_lifetime_sec - 60:
                self.allow_jobs = False

            if not self.allow_jobs and input_queue.empty():
                logging.info("👋 Worker exceeded lifetime, shutting down gracefully.")
                os._exit(0)

            payload = {
                "worker_name": self.worker_name,
                "online": True,
                "last_update": datetime.now(timezone.utc).isoformat(),
                "queue_length": input_queue.qsize(),
                "queued_ids": [job.job_uuid for job in list(input_queue.queue)],
                "last_time_job_added": self.last_time_job_added.isoformat() if self.last_time_job_added else None,
                "allow_jobs": self.allow_jobs
            }

            requests.post(f"{self.cpu_server_url}/worker/register", json=payload, timeout=5)
            logging.info(f"📡 Heartbeat sent for {self.worker_name}")
        except Exception as e:
            logging.warning(f"⚠️ Heartbeat failed: {e}")


@contextmanager
def suppress_logging(level=logging.ERROR):
    """
    Temporarily suppress logging messages below the specified level.
    """
    logger = logging.getLogger()
    handlers = logger.handlers[:]
    original_levels = [h.level for h in handlers]

    try:
        for h in handlers:
            h.setLevel(level)
        yield
    finally:
        for h, original_level in zip(handlers, original_levels):
            h.setLevel(original_level)