import logging
import httpx
import os
from app.tasks.TaskBase import TaskBase
from pathlib import Path
import yaml


class TA13_0_TransferSecondaryDataWrapper(TaskBase):
    def setup(self):
        self.controller.update_message("🔍 Scanning for available subtasks...")
        self.base_url = self.instructions.get("api_base_url", "http://localhost:8000")
        self.task_prefix = "TA13_"
        self.task_dir = self.instructions.get("task_dir_path", "app/tasks/TA13_Tasks")
        self.valid_tasks = []
        logging.debug5("🔧 Wrapper task setup complete")

    def run(self):
        try:
            logging.debug3("📂 Starting subtask discovery phase")
            self.controller.update_message("📂 Listing known YAML tasks and matching Python files...")

            known_tasks = self.get_yaml_tasks()
            logging.debug2(f"🧾 YAML-defined tasks: {known_tasks}")

            matching_files = self.get_existing_task_files()
            logging.debug2(f"📁 Task files found: {matching_files}")

            self.valid_tasks = [t for t in known_tasks if t in matching_files]
            logging.debug3(f"✅ Valid subtasks to trigger: {self.valid_tasks}")
            self.controller.update_item_count(len(self.valid_tasks))
            self.valid_tasks = self.filter_runnable_tasks(self.valid_tasks)

            for idx, task_name in enumerate(self.valid_tasks):
                self.check_control()
                logging.debug3(f"➡️ Dispatching {task_name} ({idx+1}/{len(self.valid_tasks)})")
                self.trigger_task_via_http(task_name)
                self.controller.update_progress((idx + 1) / len(self.valid_tasks))

            self.controller.finalize_success()
            logging.debug5("🎉 All subtasks dispatched successfully")
        except Exception as e:
            logging.exception("❌ Wrapper execution failed")
            self.controller.finalize_failure(str(e))
            raise
        finally:
            logging.debug5("🧹 Running cleanup after task execution")
            self.cleanup()

    def cleanup(self):
        self.flush_memory_logs()
        self.controller.archive_with_orm()
        logging.debug5("🧼 Wrapper task cleanup complete")

    def get_yaml_tasks(self):
        config_path = Path("app/config/instructions.yaml")
        try:
            with config_path.open("r") as f:
                full_config = yaml.safe_load(f)
            all_tasks = []
            for group, tasks in full_config.get("tasks", {}).items():
                for task_name, task_config in tasks.items():
                    if isinstance(task_config, dict):
                        # third-level check
                        for subtask_name in task_config.keys():
                            if isinstance(task_config[subtask_name], dict):
                                all_tasks.append(subtask_name)
                        else:
                            all_tasks.append(task_name)
            logging.debug(f"📜 Discovered YAML tasks: {all_tasks}")
            return [
                t for t in all_tasks
                if t.startswith(self.task_prefix)
                and not t.endswith("Wrapper")
                and t != self.instructions.get("taskName")
            ]
        except Exception as e:
            logging.error(f"❌ Failed to load instructions.yaml: {e}")
            return []


    def get_existing_task_files(self):
        try:
            filenames = os.listdir(self.task_dir)
            logging.debug2(f"📂 Directory listing for {self.task_dir}: {filenames}")
            return [f.replace(".py", "") for f in filenames if f.endswith(".py")]
        except Exception as e:
            logging.error(f"❌ Failed scanning task directory: {e}")
            return []
