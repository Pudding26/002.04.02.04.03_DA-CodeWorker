from datetime import datetime
import logging
from pathlib import Path
import time
import os
from dotenv import load_dotenv
import httpx

from app.tasks.TaskBase import TaskBase

from app.utils.rsync.RsyncUtils import RsyncUtils


class TA05_0_PassiveRsyncWatcher(TaskBase):
    def setup(self):
        self.rsync = RsyncUtils()
        self.controller.update_message("Passive Rsync Watcher initialized.")
        self.watch_dirs = [
            *Path(self.rsync.outbound).glob("*/"),
            *Path(self.rsync.inbound).glob("*/")
        ]

        load_dotenv()
        self.BACKEND_ORCHESTRATOR_BASE_URL = os.getenv("BACKEND_ORCH_BASE_URL")

    def run(self):
        try:
            while not self.controller.should_stop():
                for folder in Path(self.rsync.outbound).glob("*/"):
                    self._scan_dir_and_trigger(folder, "push")

                for folder in Path(self.rsync.inbound).glob("*/"):
                    self._scan_dir_and_trigger(folder, "pull")
                time.sleep(5)

        except Exception as e:
            self.controller.finalize_failure(str(e))
            raise
        finally:
            self.cleanup()

    def cleanup(self):
        self.flush_memory_logs()
        self.controller.archive_with_orm()
        logging.debug3("🧼 [TA05] Cleanup complete.")

    def _scan_dir_and_trigger(self, path, direction):
            for file in Path(path).rglob("*.hdf5"):
                if file.name.endswith(".partial"):
                    continue

                marker = file.with_suffix(file.suffix + ".synced")
                if marker.exists():
                    continue

                try:
                    marker.touch(exist_ok=True)
                    logging.debug(f"🧷 Marker created for {file}")
                except Exception as e:
                    logging.warning(f"⚠️ Could not create marker for {file.name}: {e}")
                    continue

                self._spawn_rsync_transfer_task(direction, str(file))
                time.sleep(0.5)



    def _spawn_rsync_transfer_task(self, direction, filepath):
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        task_name = "TA05_A_SingleRsyncWorker"
        custom_task_name = f"rsync_{direction}_{Path(filepath).stem}_{timestamp}"

        payload = {
            "task_name": task_name,
            "custom_task_name": custom_task_name,
            "params": {
                "filepath": filepath,
                "direction": direction
            }
        }

        try:
            self.trigger_task_via_http(task_name, payload=payload)
        except Exception as e:
            logging.error(f"❌ Failed to trigger {task_name}: {e}")



    def trigger_task_via_http(self, task_name, payload):
        url = f"{self.BACKEND_ORCHESTRATOR_BASE_URL}/tasks/start"

        logging.debug2(f"🌐 Triggering subtask via HTTP POST: {url} with payload {payload}")
        try:
            res = httpx.post(url, json=payload)
            if res.status_code == 200:
                logging.info(f"✅ Successfully triggered {task_name}")
            else:
                logging.error(f"❌ Failed to trigger {task_name}: {res.status_code} {res.text}")
        except Exception as e:
            logging.exception(f"❌ HTTP error while triggering {task_name}: {e}")