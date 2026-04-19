"""Tests for the operator-driven arrival gate.

Covers the two pieces wired together at runtime:

* :class:`reoh_bot.arrival_gate.ArrivalGate` — one-shot async gate.
* :func:`reoh_bot.arrival_gate.run_stdin_signaler` — pumps stdin lines into
  the gate, with EOF and cancellation handled cleanly.

The signaler tests use an ``io.StringIO`` instead of real stdin so they stay
deterministic and avoid touching any TTY. We drive the coroutines through
``asyncio.run`` rather than introducing a ``pytest-asyncio`` dependency.
"""

from __future__ import annotations

import asyncio
import io
import time

import pytest

from reoh_bot.arrival_gate import ArrivalGate, run_stdin_signaler


def test_signal_releases_pending_wait() -> None:
    async def _run() -> None:
        gate = ArrivalGate()
        waiter = asyncio.create_task(gate.wait())

        await asyncio.sleep(0)
        assert not waiter.done()

        gate.signal()
        await asyncio.wait_for(waiter, timeout=1.0)

    asyncio.run(_run())


def test_wait_re_arms_after_signal() -> None:
    """Two consecutive signals should release two consecutive waits."""

    async def _run() -> None:
        gate = ArrivalGate()

        gate.signal()
        await asyncio.wait_for(gate.wait(), timeout=1.0)

        second = asyncio.create_task(gate.wait())
        await asyncio.sleep(0)
        assert not second.done(), "gate must re-arm; one signal cannot release two waits"

        gate.signal()
        await asyncio.wait_for(second, timeout=1.0)

    asyncio.run(_run())


def test_stdin_signaler_signals_per_line() -> None:
    async def _run() -> None:
        gate = ArrivalGate()
        stream = io.StringIO("\n\n")  # two Enter presses, then EOF

        signaler = asyncio.create_task(run_stdin_signaler(gate, stream=stream))

        await asyncio.wait_for(gate.wait(), timeout=1.0)
        await asyncio.wait_for(gate.wait(), timeout=1.0)

        # EOF after the second line should let the signaler exit on its own.
        await asyncio.wait_for(signaler, timeout=1.0)

    asyncio.run(_run())


def test_stdin_signaler_cancels_cleanly() -> None:
    class _Blocking(io.StringIO):
        """A pipe that never returns from ``readline``.

        Cancellation is the only way the signaler task can exit when reading
        from this stream, so it doubles as a regression test for the cleanup
        path in :func:`run_stdin_signaler`.
        """

        def readline(self, *_a, **_k) -> str:  # type: ignore[override]
            while True:
                time.sleep(0.05)

    async def _run() -> None:
        gate = ArrivalGate()
        signaler = asyncio.create_task(run_stdin_signaler(gate, stream=_Blocking()))
        await asyncio.sleep(0.1)

        signaler.cancel()
        with pytest.raises(asyncio.CancelledError):
            await signaler

    asyncio.run(_run())
