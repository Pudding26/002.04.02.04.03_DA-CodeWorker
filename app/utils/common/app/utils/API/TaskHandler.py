
import logging
from fastapi import HTTPException
from pathlib import Path
from typing import Optional
import yaml
import importlib
from threading import Thread
import uuid
import time

from app.utils.controlling.TaskController import TaskController
from app.utils.SQL.SQL_Dict import SQL_Dict


class TaskHandler:
    def __init__(self, instructions_path: str = "app/config/instructions.yaml"):
        self.instructions_path = Path(instructions_path)
        logging.debug2(f"Initializing TaskHandler with config: {self.instructions_path}")
        self.instructions = self._load_instructions()

    def _load_instructions(self):
        logging.debug2(f"Loading instructions from: {self.instructions_path}")
        if not self.instructions_path.exists():
            logging.error(f"Instruction file not found: {self.instructions_path}")
            raise FileNotFoundError("Instruction file not found.")
        with self.instructions_path.open("r") as f:
            instructions = yaml.safe_load(f)
        logging.debug2("Instructions loaded successfully.")
        return instructions

    def reload(self):
        logging.debug2("Reloading instructions...")
        self.instructions = self._load_instructions()
        logging.debug2("Instructions reloaded successfully.")
        return self.instructions

    def find_task_config(self, task_name: str):
        logging.debug2(f"Searching for task config: {task_name}")
        for group_name, tasks in self.instructions.get("tasks", {}).items():
            if task_name in tasks:
                logging.debug2(f"Found config for '{task_name}' in group '{group_name}'")
                return tasks[task_name]
        logging.warning(f"Task config not found for: {task_name}")
        raise HTTPException(status_code=404, detail=f"Task '{task_name}' not found in config")

    def get_tasks(self):
        logging.debug2("Collecting task groups and task names...")
        task_groups = []

        for group_name, tasks in self.instructions.get("tasks", {}).items():
            logging.debug2(f"Parsing group: {group_name}")
            task_names = []

            for task_name in tasks:
                label = f"{task_name}" if "Wrapper" in task_name else task_name
                task_names.append(label)
                logging.debug2(f" - Task registered: {label}")

            task_groups.append({
                "group": group_name,
                "tasks": task_names
            })

        logging.debug2("Task group collection complete.")
        return task_groups

    def start_task(self, task_name: str, custom_task_name: Optional[str] = None, custom_params: Optional[dict] = None):
        
        
        logging.debug2(f"Starting task: {task_name}")
        task_config = self.find_task_config(task_name)
        logging.debug2(f"Task config found: {task_config}")
        
        

        
        if custom_params:
            logging.debug2(f"Custom parameters provided: {custom_params}")
            task_config.update(custom_params)
        if custom_task_name:
            logging.debug2(f"Custom task name provided: {custom_task_name}")
            task_config["task_name"] = task_name  # static, from YAML
            task_config["custom_task_name"] = custom_task_name  # dynamic



        actual_task_name = custom_task_name or task_name
        logging.debug2(f"Checking if task '{actual_task_name}' is already running...")
        db = SQL_Dict(db_key="progress", table_name=actual_task_name)
        current_status = db.get("Status")
        if current_status and current_status.lower() == "running":
            logging.warning(f"Task '{actual_task_name}' is already running.")
            raise HTTPException(status_code=409, detail=f"Task '{actual_task_name}' is already running.")


        logging.debug2(f"Task '{actual_task_name}' is not running. Proceeding to start...")
        try:
            db.set("Status", "running")
            prefix = task_name.split("_")[0]
            logging.debug2(f"Resolved task prefix: {prefix}")
            task_dir = next(
                d.name for d in Path("app/tasks").iterdir()
                if d.is_dir() and d.name.startswith(prefix)
            )
            module_path = f"app.tasks.{task_dir}.{task_name}"
            logging.debug2(f"🧠 Importing module: {module_path}")

            module = importlib.import_module(module_path)

            if not hasattr(module, task_name):
                logging.error(f"❌ Module '{module_path}' has no attribute '{task_name}'")
                raise AttributeError(f"❌ Module '{module_path}' has no attribute '{task_name}'")

            task_class = getattr(module, task_name)

            task_uuid = str(uuid.uuid4())
            logging.debug2(f"Generated task UUID: {task_uuid}")

            # Initialize fresh progress table with new UUID and start_time



            controller = TaskController(
                db_key="progress",
                task_name=actual_task_name,
                task_uuid=task_uuid
            )

            controller.db.set("task_uuid", task_uuid)
            controller.db.set("start_time", str(time.strftime("%Y-%m-%d %H:%M:%S")))
            controller.db.set("Finished", "0")  # clear any stale status
            controller.db.set("Status", "running")
            controller.db.set("message", "Starting task...")



            task_config["Thread_progress_db_path"] = "progress"
            task_config["Thread_progress_table_name"] = actual_task_name

            logging.debug2(f"Creating task instance: {task_class.__name__}")
            task_instance = task_class(task_config, controller)

            # Attach task_name for logging clarity if missing
            task_instance.task_name = getattr(task_instance, "task_name", actual_task_name)

            def _safe_task_run(task):
                try:
                    logging.debug2(f"[Thread] Running task: {task.task_name}")
                    task.run()
                except Exception as e:
                    logging.error(f"❌ Task {task.task_name} crashed in thread: {e}", exc_info=True)
                    if task.controller.progress_table_exists():
                        logging.warning(f"🗑️ Cleaning up progress in TaskHandler needed table for task {task.task_name} due to crash.")
                    
                        task.controller.finalize_failure(str(e))
                        task.cleanup()
                    else:
                        logging.info(f"❌ Task {task.task_name} crashed but no progress table found to clean up.")

            # Launch in background
            thread = Thread(target=_safe_task_run, args=(task_instance,))
            thread.start()

            # Watchdog: ensure it starts responding within 10 seconds
            startup_deadline = time.time() + 10
            while time.time() < startup_deadline:
                if db.get("message"):  # means run() has started
                    break
                time.sleep(0.5)
            else:
                logging.error(f"🚨 Task {task_name} did not initialize within timeout.")
                db.set("Status", "Failed")
                db.set("message", "Startup timeout — no message set within 10 seconds.")
                raise HTTPException(status_code=500, detail="Task failed to initialize (startup timeout).")

            logging.debug2(f"Task '{task_name}' started successfully.")
            return {"status": "started", "task": task_name, "config": task_config}

        except Exception as e:
            logging.exception("Failed to start task:")
            raise HTTPException(status_code=500, detail=f"Failed to start task: {str(e)}")


    def stop_task(self, task_name: str):
        logging.debug2(f"Received stop request for task: {task_name}")

        # Check if task is known
        all_tasks = [task for group in self.get_tasks() for task in group["tasks"]]
        if task_name not in all_tasks:
            logging.warning(f"Task to stop not found: {task_name}")
            raise HTTPException(status_code=404, detail="Task not found")

        try:
            # Construct the controller and issue the stop request
            controller = TaskController(
                db_key="progress",
                task_name=task_name,
                task_uuid=None  # Optional: read from DB if required
            )
            controller.request_stop()
            logging.info(f"🛑 Stop signal sent for task: {task_name}")
            return {"status": "stopping", "task": task_name}

        except Exception as e:
            logging.error(f"❌ Failed to stop task {task_name}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to stop task: {str(e)}")

    

    def reload(self):
        logging.debug2("Reloading task instructions...")
        self.instructions = self._load_instructions()
        return self.instructions
