"""FastAPI signaling server for the Pipecat WebRTC voice agent."""

from pathlib import Path

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection

from bot import run_bot

load_dotenv(override=True)

ICE_SERVERS = ["stun:stun.l.google.com:19302"]
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Embodied Chatbot Signaling Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

active_connections: dict[str, SmallWebRTCConnection] = {}


@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks) -> dict:
    """Accept an SDP offer, start a bot session, and return the SDP answer."""
    connection = SmallWebRTCConnection(ICE_SERVERS)
    await connection.initialize(sdp=request["sdp"], type=request["type"])

    @connection.event_handler("closed")
    async def _on_closed(conn: SmallWebRTCConnection) -> None:
        logger.info(f"Peer connection closed: {conn.pc_id}")
        active_connections.pop(conn.pc_id, None)

    active_connections[connection.pc_id] = connection
    background_tasks.add_task(run_bot, connection)

    return connection.get_answer()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "active_connections": len(active_connections)}


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="localhost", port=7860, reload=False)
