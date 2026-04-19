"""Operator-driven gate that pauses the bot until the robot has arrived.

The E2LG agent narrates a guided tour over multiple rooms. When it asks the
visitor to "please follow me", the physical robot needs time to actually move
to the next room before the agent describes what is in there. There is no
arrival sensor on the robot, so we expose a manual gate: the operator presses
``Enter`` in the terminal once the robot has reached the next room, and the
agent is allowed to continue.

The gate is intentionally tiny:

* :class:`ArrivalGate` is an ``asyncio.Event`` wrapper with one-shot semantics
  (``wait()`` clears the event before returning, so the next ``wait()`` blocks
  again — one Enter press releases exactly one waiter).
* :func:`run_stdin_signaler` is the async task that reads ``stdin`` lines via
  ``asyncio.to_thread`` and signals the gate per line. It must be cancelled on
  shutdown; cancellation propagates as a ``CancelledError`` out of the
  ``to_thread`` call so the task ends cleanly.

Wiring is the caller's job — see ``reoh_bot.bot.run_bot``.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from typing import TextIO

from loguru import logger


@dataclass
class ArrivalGate:
    """One-shot async gate signalled by the operator.

    A single ``signal()`` releases the next pending ``wait()`` and only that
    one — the event is cleared inside ``wait`` before it returns so subsequent
    waiters block again until the operator signals once more.
    """

    _event: asyncio.Event = field(default_factory=asyncio.Event)

    async def wait(self) -> None:
        """Block until the next ``signal()`` call, then re-arm the gate."""
        await self._event.wait()
        self._event.clear()

    def signal(self) -> None:
        """Release the next pending ``wait()``."""
        self._event.set()


async def run_stdin_signaler(
    gate: ArrivalGate,
    *,
    prompt: str = "[arrival gate] press Enter when the robot has arrived... ",
    stream: TextIO | None = None,
) -> None:
    """Read lines from ``stdin`` and signal ``gate`` on each.

    Runs forever until cancelled. ``stream`` is exposed for tests; in
    production we always read from ``sys.stdin`` via ``input()``.
    """
    src = stream if stream is not None else sys.stdin
    is_tty = src.isatty() if hasattr(src, "isatty") else False

    while True:
        try:
            if src is sys.stdin and is_tty:
                # input() handles terminal echo + line editing; run it off the
                # event loop so we don't block the pipeline.
                await asyncio.to_thread(input, prompt)
            else:
                # Tests / piped input: a blocking readline keeps semantics
                # identical without depending on a TTY.
                line = await asyncio.to_thread(src.readline)
                if line == "":  # EOF — no more input will arrive.
                    logger.info("arrival-gate stdin reached EOF; signaler stopping")
                    return
        except asyncio.CancelledError:
            raise
        except EOFError:
            logger.info("arrival-gate stdin closed; signaler stopping")
            return
        except Exception:
            logger.exception("arrival-gate signaler crashed; stopping")
            return

        logger.info("arrival-gate: operator signalled arrival")
        gate.signal()
