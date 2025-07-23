import os
from dotenv import load_dotenv


load_dotenv()
BACKEND_HOSTNAME_PUBLIC = os.getenv("BACKEND_HOSTNAME_PUBLIC")


ORCHESTRATOR_URL = "http://" + BACKEND_HOSTNAME_PUBLIC

