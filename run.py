
print("Import: uvicorn")
import uvicorn

print("Import: os")
import os

print("Import: load_dotenv")
from dotenv import load_dotenv

print("Import: LoggingHandler")
from app.utils.common.app.utils.logger.loggingWrapper import LoggingHandler

print("Import: LOGGING_CONFIG")
from app.utils.common.app.utils.logger.UvicornLoggingFilter import LOGGING_CONFIG  

print("Import: logging")
import logging



logging.getLogger("watchfiles.main").setLevel(logging.WARNING)


if __name__ == "__main__":


    logger = logging.getLogger(__name__)
    LoggingHandler(logging_level="DEBUG-2")
    

    load_dotenv()
    if os.getenv("DEBUG_MODE") == "True":
        GPU_WORKER_BASE_PORT = int(os.getenv("GPU_WORKER_BASE_PORT", 8000))
    else:
        # Default port for the GPU worker
        GPU_WORKER_BASE_PORT = int(8000)

    logging.info(f"Starting {os.getenv('GPU_WORKER_NAME')} on port {GPU_WORKER_BASE_PORT}...")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=GPU_WORKER_BASE_PORT,
        reload=False,
        log_config=LOGGING_CONFIG
    )



