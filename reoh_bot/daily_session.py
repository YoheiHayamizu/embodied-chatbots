"""Thin async wrapper over the Daily REST API.

We only need two operations: create a short-lived room and mint a meeting
token for the bot. Anything more should be done with the official
``daily-python`` SDK directly. The wrapper exists so that ``app.py`` can stay
free of HTTP plumbing and so the request/response shape is documented in one
place.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import aiohttp


class DailyAPIError(RuntimeError):
    """Raised when the Daily REST API returns a non-2xx response."""


@dataclass(frozen=True)
class DailyRoom:
    """A Daily room and a token good enough for the bot to join it."""

    url: str
    name: str
    bot_token: str
    expires_at: int


class DailyRESTClient:
    """Minimal async client for the bits of the Daily REST API we use."""

    def __init__(self, *, api_key: str, api_url: str = "https://api.daily.co/v1") -> None:
        if not api_key:
            raise ValueError("Daily API key is required")
        self._api_key = api_key
        self._api_url = api_url.rstrip("/")

    async def create_room(
        self,
        *,
        expiry_seconds: int = 3600,
        enable_chat: bool = False,
        enable_recording: bool = False,
    ) -> DailyRoom:
        """Create a short-lived Daily room and a bot token for it.

        Args:
            expiry_seconds: How long until the room (and token) expires.
            enable_chat: Whether the Daily Prebuilt chat panel is available.
            enable_recording: Whether participants can record the call.

        Returns:
            A :class:`DailyRoom` ready for ``DailyTransport`` to join.
        """
        expires_at = int(time.time()) + expiry_seconds

        async with aiohttp.ClientSession() as session:
            room_payload: dict[str, Any] = {
                "properties": {
                    "exp": expires_at,
                    "enable_chat": enable_chat,
                    "enable_recording": "cloud" if enable_recording else False,
                    "eject_at_room_exp": True,
                    "start_audio_off": False,
                    "start_video_off": True,
                }
            }
            async with session.post(
                f"{self._api_url}/rooms",
                headers=self._auth_headers(),
                json=room_payload,
            ) as response:
                room = await self._parse(response, "create room")

            token_payload = {
                "properties": {
                    "room_name": room["name"],
                    "is_owner": True,
                    "exp": expires_at,
                }
            }
            async with session.post(
                f"{self._api_url}/meeting-tokens",
                headers=self._auth_headers(),
                json=token_payload,
            ) as response:
                token = await self._parse(response, "create meeting token")

        return DailyRoom(
            url=str(room["url"]),
            name=str(room["name"]),
            bot_token=str(token["token"]),
            expires_at=expires_at,
        )

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    async def _parse(response: aiohttp.ClientResponse, op: str) -> dict[str, Any]:
        text = await response.text()
        if response.status >= 400:
            raise DailyAPIError(f"Daily REST {op} failed ({response.status}): {text}")
        try:
            return await response.json(content_type=None)
        except aiohttp.ContentTypeError as exc:
            raise DailyAPIError(f"Daily REST {op} returned non-JSON body: {text}") from exc
