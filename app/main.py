from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging

from app.utils.logger.loggingWrapper import LoggingHandler
from app.utils.API.TaskRouter import router as task_router
from fastapi.middleware.cors import CORSMiddleware

from app.tasks.TA01_setup.TA01_A_SQLSetup import TA01_A_SQLSetup

from app.utils.SQL.DBEngine import create_all_tables
from app.utils.HDF5.HDF5Utils import HDF5Utils
from app.utils.controlling.TaskController import TaskController

logger = logging.getLogger(__name__)
LoggingHandler(logging_level="DEBUG-2")


@asynccontextmanager
async def lifespan(app: FastAPI):
    
    TA01_A_SQLSetup.createDatabases()
    TaskController.clean_orphaned_tasks_on_start()

    create_all_tables()

    directories = [
    "data/rawData/",
    "data/productionData/",
    ]
    
    # ✅ This block runs on startup
    logger.info("🔓 Unlocking dirty HDF5 files...")
    HDF5Utils.unlock_dirty_hdf5_files(directories=directories)

    yield



app = FastAPI(lifespan=lifespan)




app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(task_router, prefix="/tasks", tags=["tasks"])

@app.get("/")
def read_root():
    return {"message": "Orchestrator is running"}
