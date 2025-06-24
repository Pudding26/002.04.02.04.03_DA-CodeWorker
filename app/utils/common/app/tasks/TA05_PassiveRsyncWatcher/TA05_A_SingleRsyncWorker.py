from pathlib import Path
from app.tasks.TaskBase import TaskBase
from app.utils.rsync.RsyncUtils import RsyncUtils

class TA05_A_SingleRsyncWorker(TaskBase):
    def setup(self):
        self.filepath = Path(self.instructions.get("filepath"))
        self.direction = self.instructions.get("direction")
        self.rsync = RsyncUtils()
        self.controller.update_message(f"Initialized for {self.direction} of {self.filepath}")

    def run(self):
        try:
            def report_progress(p):
                self.controller.update_progress(p)

            if self.direction == "push":
                self.controller.update_message(f"Pushing {self.filepath}...")
                self.rsync.push_file_to_peer(self.filepath, on_progress=report_progress)
            elif self.direction == "pull":
                self.controller.update_message(f"Pulling {self.filepath}...")
                self.rsync.pull_file_from_peer(self.filepath, on_progress=report_progress)
            else:
                raise ValueError(f"Unknown direction: {self.direction}")

            self.controller.update_progress(1.0)
            self.rsync.cleanup_file_and_marker(self.filepath)
            self.controller.finalize_success()
        except Exception as e:
            self.controller.finalize_failure(str(e))
            raise
        finally:
            pass
    def cleanup(self):
        self.controller.archive_with_orm()