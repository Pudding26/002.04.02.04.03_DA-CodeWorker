import asyncio
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.utils.lobby.LobbyHandler import LobbyHandler
from app.utils.common.app.utils.logger.loggingWrapper import LoggingHandler

from contextlib import asynccontextmanager

# Instantiate the handler
lobby_handler = LobbyHandler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 🕐 wait for a bit to ensure the app is accepting connections
    asyncio.create_task(delayed_start())
    yield
    # optional shutdown code here


def safe_init_orms():
        #progress
        from app.utils.common.app.utils.SQL.models.progress.orm.ProfileArchive import ProfileArchive
        from app.utils.common.app.utils.SQL.models.progress.orm.ProgressArchive import ProgressArchive
        
        # raw
        from app.utils.common.app.utils.SQL.models.raw.orm.PrimaryDataRaw import PrimaryDataRaw
        
        # production
        from app.utils.common.app.utils.SQL.models.production.orm.DS09 import DS09
        from app.utils.common.app.utils.SQL.models.production.orm.DS40 import DS40
        from app.utils.common.app.utils.SQL.models.production.orm.DS12 import DS12
        from app.utils.common.app.utils.SQL.models.production.orm.WoodTableA import WoodTableA
        from app.utils.common.app.utils.SQL.models.production.orm.WoodTableB import WoodTableB
        from app.utils.common.app.utils.SQL.models.production.orm.WoodMaster import WoodMaster
        from app.utils.common.app.utils.SQL.models.production.orm.WoodMasterPotential import WoodMasterPotential
        from app.utils.common.app.utils.SQL.models.production.orm.DoEArchive import DoEArchive
        from app.utils.common.app.utils.SQL.models.production.orm.ModellingResults import ModellingResults



        # jobs
        from app.utils.common.app.utils.SQL.models.jobs.orm_DoEJobs import orm_DoEJobs
        from app.utils.common.app.utils.SQL.models.jobs.orm_WorkerJobs import orm_WorkerJobs
        from app.utils.common.app.utils.SQL.models.jobs.orm_JobLink import orm_JobLink

logging.getLogger("watchfiles.main").setLevel(logging.WARNING)
app = FastAPI(lifespan=lifespan)


logger = logging.getLogger(__name__)
LoggingHandler(logging_level="DEBUG-2")
safe_init_orms()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Run the handler after a short delay
async def delayed_start(delay_seconds: float = 2.0):
    await asyncio.sleep(delay_seconds)
    lobby_handler.start()
    print(f"✅ LobbyHandler started after {delay_seconds}s delay")





