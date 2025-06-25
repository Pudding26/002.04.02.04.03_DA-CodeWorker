import logging

from fastapi import APIRouter
from app.tasks.TaskWrapper import TaskRequestBase as TaskRequest
from app.utils.common.app.utils.API.TaskHandler import TaskHandler


from app.utils.common.app.utils.controlling.TaskController import TaskController

router = APIRouter()
handler = TaskHandler()

@router.get("/")
def get_tasks():
    logging.debug1("Fetching tasks")

    return {"possible_tasks": handler.get_tasks()}

@router.post("/start")
def start_task(req: TaskRequest):
    logging.info(f"Starting task: {req.task_name} → {req.custom_task_name or req.task_name}")
    return handler.start_task(
        task_name=req.task_name,
        custom_task_name=req.custom_task_name,
        custom_params=req.params
    )

@router.post("/stop")
def stop_task(req: TaskRequest):
    logging.info("Fetching tasks")

    return handler.stop_task(req.task_name)

@router.post("/reload")
def reload_instructions():
    try:
        handler.reload() 
        return {"status": "reloaded", "message": "Instructions reloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
def get_task_status():
    return TaskController.fetch_all_status()



@router.post("/control")
def control_task(command: dict):
    task_name = command.get("task_name")
    action = command.get("action")

    controller = TaskController(task_name)
    if action == "pause":
        controller.db.set("Pause", "1")
        controller.db.set("Status", "Paused")
    elif action == "resume":
        controller.db.set("Pause", "0")
        controller.db.set("Status", "Running")
    else:
        raise HTTPException(400, "Unknown action")
    return {"status": "updated", "action": action}