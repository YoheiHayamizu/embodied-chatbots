"""FastAPI signalling server for the SmallWebRTC voice agent.

The app exposes three surfaces:
  - ``POST /api/offer``: SDP offer/answer exchange. Each new connection spawns
    a pipecat pipeline via ``run_bot``. The pipecat-provided
    ``SmallWebRTCRequestHandler`` takes care of pc_id reuse, renegotiation,
    and single-connection enforcement.
  - ``GET /health``: liveness probe.
  - ``GET /``: static browser client (``smallwebrtc_bot/static``).
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    ConnectionMode,
    IceCandidate,
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)

from smallwebrtc_bot.bot import run_bot

load_dotenv(override=True)

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_ICE_SERVERS = [IceServer(urls="stun:stun.l.google.com:19302")]


@asynccontextmanager
async def _lifespan(app: FastAPI):
    app.state.bot_tasks = set()
    try:
        yield
    finally:
        tasks = list(app.state.bot_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Server shutdown complete")


app = FastAPI(title="Embodied Chatbot Signalling Server", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_handler = SmallWebRTCRequestHandler(
    ice_servers=_ICE_SERVERS,
    connection_mode=ConnectionMode.SINGLE,
)


def _spawn_bot(connection: SmallWebRTCConnection) -> None:
    """Schedule ``run_bot`` for a freshly initialised peer connection.

    Called by ``SmallWebRTCRequestHandler`` once per new connection. The task
    is tracked in app state so the lifespan teardown can cancel it.
    """
    task = asyncio.create_task(run_bot(connection), name=f"bot-{connection.pc_id}")
    app.state.bot_tasks.add(task)
    task.add_done_callback(app.state.bot_tasks.discard)


async def _connection_callback(connection: SmallWebRTCConnection) -> None:
    _spawn_bot(connection)


@app.post("/api/offer")
async def offer(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Accept an SDP offer from the browser and return the SDP answer."""
    try:
        request = SmallWebRTCRequest.from_dict(dict(payload))
    except TypeError as exc:
        raise HTTPException(status_code=400, detail=f"invalid offer payload: {exc}") from exc
    answer = await _handler.handle_web_request(
        request,
        webrtc_connection_callback=_connection_callback,
    )
    return answer


@app.patch("/api/offer")
async def offer_patch(payload: dict[str, Any]) -> dict[str, str]:
    """Accept trickled ICE candidates from the browser.

    The JS SmallWebRTC transport sends additional candidates via PATCH once
    the initial offer is accepted. Without this endpoint the browser logs a
    405 and renegotiation fails even though the first connection may still
    succeed.
    """
    pc_id = payload.get("pc_id")
    raw_candidates = payload.get("candidates") or []
    if not pc_id:
        raise HTTPException(status_code=400, detail="missing pc_id")
    try:
        candidates = [IceCandidate(**c) for c in raw_candidates]
    except TypeError as exc:
        raise HTTPException(status_code=400, detail=f"invalid candidate: {exc}") from exc
    await _handler.handle_patch_request(
        SmallWebRTCPatchRequest(pc_id=pc_id, candidates=candidates)
    )
    return {"status": "ok"}


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "active_bots": len(app.state.bot_tasks)}


app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import os

    import uvicorn

    uvicorn.run(
        "smallwebrtc_bot.app:app",
        host=os.getenv("HOST", "localhost"),
        port=int(os.getenv("PORT", "7860")),
        reload=os.getenv("RELOAD", "").lower() in {"1", "true", "yes"},
    )
