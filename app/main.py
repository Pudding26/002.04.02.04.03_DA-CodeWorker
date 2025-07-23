from app.utils.common.app.utils.logger.loggingWrapper import LoggingHandler
import logging

logger = logging.getLogger(__name__)
LoggingHandler(logging_level="DEBUG-2")

logging.debug2("Start Importing: asyncio, os, FastAPI, CORSMiddleware, asynccontextmanager")
import asyncio
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
logging.debug2("Finished Importing: asyncio, os, FastAPI, CORSMiddleware, asynccontextmanager")



logging.debug2("Start Importing: LobbyHandler")
from app.utils.lobby.LobbyHandler import LobbyHandler
logging.debug2("Start Importing: GPU_router")
from app.utils.router.GPU_router import GPU_router

# Instantiate the handler
lobby_handler = LobbyHandler()



# Ensure SQL models initialized properly
def safe_init_orms():
    logging.info("Initializing SQL ORM models...")
    from app.utils.common.app.utils.SQL.models.progress.orm.ProfileArchive import ProfileArchive
    from app.utils.common.app.utils.SQL.models.progress.orm.ProgressArchive import ProgressArchive

    from app.utils.common.app.utils.SQL.models.raw.orm.PrimaryDataRaw import PrimaryDataRaw

    from app.utils.common.app.utils.SQL.models.production.orm.DS09 import DS09
    from app.utils.common.app.utils.SQL.models.production.orm.DS40 import DS40
    from app.utils.common.app.utils.SQL.models.production.orm.DS12 import DS12
    from app.utils.common.app.utils.SQL.models.production.orm.WoodTableA import WoodTableA
    from app.utils.common.app.utils.SQL.models.production.orm.WoodTableB import WoodTableB
    from app.utils.common.app.utils.SQL.models.production.orm.WoodMaster import WoodMaster
    from app.utils.common.app.utils.SQL.models.production.orm.WoodMasterPotential import WoodMasterPotential
    from app.utils.common.app.utils.SQL.models.production.orm.DoEArchive import DoEArchive
    from app.utils.common.app.utils.SQL.models.production.orm_ModellingResults import orm_ModellingResults
    from app.utils.common.app.utils.SQL.models.jobs.orm_DoEJobs import orm_DoEJobs
    from app.utils.common.app.utils.SQL.models.jobs.orm_WorkerJobs import orm_WorkerJobs
    from app.utils.common.app.utils.SQL.models.jobs.orm_JobLink import orm_JobLink

    logging.info("SQL ORM models initialized successfully.")

# Logging config

safe_init_orms()

# Delayed startup hook
async def delayed_start(delay_seconds: float = 2.0):
    await asyncio.sleep(delay_seconds)
    lobby_handler.start()
    print(f"✅ LobbyHandler started after {delay_seconds}s delay")

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(delayed_start())
    yield
    # Shutdown tasks here if needed

# FastAPI app
app = FastAPI(lifespan=lifespan)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(GPU_router)
