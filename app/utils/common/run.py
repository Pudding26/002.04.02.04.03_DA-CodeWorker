import uvicorn
from dotenv import load_dotenv
import os
import logging

from app.utils.logger.UvicornLoggingFilter import LOGGING_CONFIG  




if __name__ == "__main__":
    load_dotenv()
    if os.getenv("DEBUG_MODE") == "True":
        BACKEND_ORCH_BASE_PORT = int(os.getenv("BACKEND_ORCH_BASE_PORT"))
    else:
        # Default port for the backend orchestrator
        BACKEND_ORCH_BASE_PORT = int(8000)
        

        
    BACKEND_ORCH_BASE_URL = os.getenv("BACKEND_ORCH_BASE_URL")
    BACKEND_ORCH_BASE_URL = "localhost"
    logging.info("CWD =", os.getcwd())
    print("HELLO")



    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=BACKEND_ORCH_BASE_PORT,
        reload=True,
        log_config=LOGGING_CONFIG
    )
