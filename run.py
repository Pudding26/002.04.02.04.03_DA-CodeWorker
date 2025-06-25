import uvicorn
import os
from dotenv import load_dotenv


print("CWD =", os.getcwd())

from app.utils.lobby.LobbyHandler import LobbyHandler

from app.utils.common.app.utils.logger.loggingWrapper import LoggingHandler

from app.utils.common.app.utils.logger.UvicornLoggingFilter import LOGGING_CONFIG  
import logging


logging.getLogger("watchfiles.main").setLevel(logging.WARNING)


if __name__ == "__main__":


    logger = logging.getLogger(__name__)
    LoggingHandler(logging_level="DEBUG-2")

    load_dotenv()
    if os.getenv("DEBUG_MODE") == "True":
        GPU_WORKER_BASE_PORT = int(os.getenv("GPU_WORKER_BASE_PORT"))
    else:
        # Default port for the GPU worker
        GPU_WORKER_BASE_PORT = int(8000)

    logging.info(f"Starting GPU worker on port {GPU_WORKER_BASE_PORT}...")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=GPU_WORKER_BASE_PORT,
        reload=True,
        log_config=LOGGING_CONFIG
    )



