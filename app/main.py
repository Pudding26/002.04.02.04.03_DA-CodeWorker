import asyncio
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.utils.lobby.LobbyHandler import LobbyHandler
from contextlib import asynccontextmanager

# Instantiate the handler
lobby_handler = LobbyHandler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 🕐 wait for a bit to ensure the app is accepting connections
    asyncio.create_task(delayed_start())  
    yield
    # optional shutdown code here
logging.getLogger("watchfiles.main").setLevel(logging.WARNING)
app = FastAPI(lifespan=lifespan)

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


