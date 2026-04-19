"""FastAPI signaling app for the REOH Daily voice bot.

Exposes:

* ``POST /api/start`` — create a Daily room, mint a bot meeting token, spawn a
  ``run_bot`` task, and return the room URL the user should open in Daily
  Prebuilt (or any Daily-compatible client). If ``DAILY_ROOM_URL`` is set in
  the environment, the bot joins that room instead of creating one.
* ``GET /health`` — liveness probe with the count of active bot tasks.
* ``GET /`` — minimal landing page so a casual ``curl`` (or browser hit)
  shows what the service is.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from loguru import logger

from reoh_bot.bot import run_bot
from reoh_bot.config import Settings
from reoh_bot.daily_session import DailyAPIError, DailyRESTClient

load_dotenv(override=True)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    app.state.settings = Settings.from_env()
    app.state.bot_tasks = set()
    logger.info(
        "REOH bot ready scenario_dir={} default_scenario={} model={}",
        app.state.settings.scenario.scenario_dir,
        app.state.settings.scenario.scenario_id or "<first available>",
        app.state.settings.llm.model,
    )
    try:
        yield
    finally:
        tasks = list(app.state.bot_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("REOH bot server shutdown complete")


app = FastAPI(title="REOH Daily Voice Bot", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "active_bots": len(app.state.bot_tasks),
        "scenario_id": app.state.settings.scenario.scenario_id,
    }


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (
        "<!doctype html><meta charset=utf-8><title>REOH Daily Voice Bot</title>"
        "<h1>REOH Daily Voice Bot</h1>"
        "<p><code>POST /api/start</code> to create a room and dispatch the agent."
        " Then open the returned <code>room_url</code> in any Daily client.</p>"
    )


@app.post("/api/start")
async def start(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Spawn a bot for a freshly-created (or pre-configured) Daily room.

    Optional JSON body fields:
      - ``scenario_id``: override the scenario for this session only.
    """
    settings: Settings = app.state.settings
    overridden_scenario = (payload or {}).get("scenario_id") if payload else None
    if overridden_scenario:
        # Build a per-request shadow of settings without mutating the global one.
        settings = _with_scenario(settings, str(overridden_scenario))

    room_url, token, expires_at = await _resolve_room(settings)

    task = asyncio.create_task(
        run_bot(settings=settings, room_url=room_url, token=token),
        name=f"reoh-bot-{room_url.rsplit('/', 1)[-1]}",
    )
    app.state.bot_tasks.add(task)
    task.add_done_callback(app.state.bot_tasks.discard)

    return {
        "room_url": room_url,
        "expires_at": expires_at,
        "scenario_id": settings.scenario.scenario_id,
    }


def _with_scenario(settings: Settings, scenario_id: str) -> Settings:
    from dataclasses import replace

    return replace(settings, scenario=replace(settings.scenario, scenario_id=scenario_id))


async def _resolve_room(settings: Settings) -> tuple[str, str | None, int | None]:
    """Return ``(room_url, token, expires_at)`` for the bot to join.

    Uses ``DAILY_ROOM_URL`` if it is set; otherwise creates a fresh room via
    the Daily REST API.
    """
    if settings.daily.room_url:
        return settings.daily.room_url, settings.daily.room_token, None

    client = DailyRESTClient(
        api_key=settings.daily.api_key,
        api_url=settings.daily.api_url,
    )
    try:
        room = await client.create_room(expiry_seconds=settings.daily.room_expiry_seconds)
    except DailyAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return room.url, room.bot_token, room.expires_at


if __name__ == "__main__":
    import os

    import uvicorn

    uvicorn.run(
        "reoh_bot.app:app",
        host=os.getenv("HOST", "localhost"),
        port=int(os.getenv("PORT", "7861")),
        reload=os.getenv("RELOAD", "").lower() in {"1", "true", "yes"},
    )
