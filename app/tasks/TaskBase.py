from abc import ABC, abstractmethod
import uuid
import io
import os
from contextlib import contextmanager
import logging
import httpx
from pathlib import Path
import yaml
from datetime import datetime
from app.utils.controlling.TaskController import TaskController
from app.utils.SQL.DBEngine import DBEngine
from app.utils.logger.ProfileLogger import ProfileLogger 
from sqlalchemy.orm import Session

class TaskBase(ABC):
    """
    Abstract base class for all tasks. Enforces a standard lifecycle and integrates task control.
    """

    def __init__(self, instructions: dict, controller: TaskController, enable_profiling: bool = True):
        """
        :param instructions: Dictionary of task-specific parameters
        :param controller: An instance of TaskController (DB-backed for progress/control)
        """
        self.instructions = instructions
        self.controller = controller or TaskController(
            task_name=self.instructions.get("taskName"),
            task_uuid=str(self.task_uuid)
            )
        
        self.status = "Initialized"
        self.task_uuid = self.controller.task_uuid
        self.enable_profiling = enable_profiling
        self._profiler_streams = {}
        self.needs_running_yaml = Path("app/config/taskNeedsRunning.yaml")

        if self.enable_profiling:
            self._setup_memory_profiling()
        
        
        self.setup()

    def _setup_memory_profiling(self):
        """Prepare memory streams and ORM handler."""
        self.device = "Lele_Lenovo"
        self.profile_type = "memProfile"
        self.task_group = self.instructions.get("task_group", "UnknownGroup")
        self.task_name = self.instructions.get("taskName") or self.__class__.__name__

        # Create in-memory streams per step
        self._profiler_streams = {
            "step1": io.StringIO(),
            "step2": io.StringIO(),
            "step3": io.StringIO()
        }

        # Create DB session
        session: Session = DBEngine("progress").get_session()

        # ORM profiler handler
        self.ProfileLogger = ProfileLogger(
            task_group=self.task_group,
            task_name=self.task_name,
            device=self.device,
            session=session,
            profile_type=self.profile_type
        )
        self.ProfileLogger.task_uuid = self.task_uuid  # assign shared UUID

    def flush_memory_logs(self):
        """Push all memory logs to database."""
        if not self.enable_profiling:
            return
        for stream in self._profiler_streams.values():
            self.ProfileLogger.log_stream_to_db(stream)


    @contextmanager
    def suppress_logging(self, level=logging.WARNING):
        """
        Temporarily suppress logging below the given level (default: WARNING).

        Usage:
            with self.suppress_logging():
                self.fetch(...)
        """
        logger = logging.getLogger()
        previous_level = logger.level
        logger.setLevel(level)
        try:
            yield
        finally:
            logger.setLevel(previous_level)




    @abstractmethod
    def setup(self):
        """
        Setup method to prepare resources, configurations, or connections.
        Must be implemented by the subclass.
        """
        pass

    @abstractmethod
    def run(self):
        """
        The main logic of the task. Must be implemented by the subclass.
        This method should call self.check_control() periodically.
        """
        pass

    @abstractmethod
    def cleanup(self):
        """
        Clean up resources. Called at the end of task execution or after error.
        """
        pass

    def check_control(self):
        """
        Check if the task should pause or stop.
        Should be called periodically in long-running loops.
        """
        self.controller.wait_if_paused()
        if self.controller.should_stop():
            self.status = "Stopped"
            raise InterruptedError("Task was stopped.")

    @staticmethod
    def trigger_task_via_http(task_name):
        BACKEND_ORCH_BASE_PORT = os.getenv("BACKEND_ORCH_BASE_PORT")
        api_base_url = f"http://localhost:{BACKEND_ORCH_BASE_PORT}"
        
        url = f"{api_base_url}/tasks/start"
        payload = {"task_name": task_name}

        logging.debug(f"🌐 Triggering subtask via HTTP POST: {url} with payload {payload}")
        try:
            res = httpx.post(url, json=payload)
            if res.status_code == 200:
                logging.info(f"✅ Successfully triggered {task_name}")
            else:
                logging.error(f"❌ Failed to trigger {task_name}: {res.status_code} {res.text}")
        except Exception as e:
            logging.exception(f"❌ HTTP error while triggering {task_name}: {e}")

    def set_needs_running(self, value: bool):
        """
        Sets a flag in the YAML indicating whether this task should run in the future.
        Example file:
            TA11_A_Import_DS01: false
            TA11_B_Import_DS04: true
        """
        data = {}
        if self.needs_running_yaml.exists():
            with self.needs_running_yaml.open("r") as f:
                data = yaml.safe_load(f) or {}

        data[self.task_name] = value

        with self.needs_running_yaml.open("w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

        logging.debug2(f"📝 Set needs_running[{self.task_name}] = {value} in {self.needs_running_yaml}")

    def filter_runnable_tasks(self, task_list: list) -> list:
        """
        Filters out tasks marked as 'False' in the needs_running.yaml.
        Returns only the tasks allowed to run.
        """
        if not self.needs_running_yaml.exists():
            logging.debug2("🔍 No needs_running.yaml found. Assuming all tasks runnable.")
            return task_list

        with self.needs_running_yaml.open("r") as f:
            data = yaml.safe_load(f) or {}

        filtered = [task for task in task_list if data.get(task, True)]
        skipped = set(task_list) - set(filtered)

        logging.debug2(f"✅ Filtered runnable tasks: {filtered}")
        if skipped:
            logging.debug2(f"⛔ Skipped tasks due to needs_running = False: {list(skipped)}")

        return filtered
