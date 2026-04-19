# reoh_bot — REOH Daily Voice Bot

A pipecat voice agent that runs an end-to-end Real Estate Open House (REOH) tour
over a [Daily](https://daily.co) room. Same pipeline shape as `smallwebrtc_bot/`
(Whisper STT → LLM → Piper TTS, with Silero VAD), but:

| | `smallwebrtc_bot/` | `reoh_bot/` |
| --- | --- | --- |
| Transport | SmallWebRTC (browser ⇄ FastAPI) | Daily (cloud room) |
| Client | Bundled static page | Any Daily client (Prebuilt, app, SDK) |
| LLM | Provider-agnostic factory | Anthropic Claude, scenario-bound |
| Dialog logic | Generic assistant | REOH **E2LG** agent (rewritten) |

## Layout

```
reoh_bot/
├── __init__.py
├── README.md
├── app.py              # FastAPI: POST /api/start, GET /health
├── bot.py              # Pipecat pipeline (Daily + Whisper + Claude + Piper + Silero)
├── config.py           # Frozen-dataclass settings, env-driven
├── e2lg_agent.py       # Reimplemented E2LG agent (Claude-backed)
├── scenarios.py        # Scenario JSON loader + property/goal renderers
├── daily_session.py    # Async REST helper for Daily room/token creation
├── run.sh              # uv run uvicorn launcher
└── prompts/
    └── e2lg_system_prompt.md
```

## How it differs from upstream `reoh.agents.e2lg_agent.E2LGAgent`

The original lives in [`reoh/reoh/agents/e2lg_agent.py`](https://github.com/YoheiHayamizu/reoh)
and is text-mode and OpenAI-only. This rewrite, in `e2lg_agent.py`:

- **LLM:** swaps OpenAI for Anthropic Claude via pipecat's
  `AnthropicLLMService`, with prompt caching enabled.
- **State:** removes the bespoke `State` dict and the `model.history` list;
  pipecat's `LLMContextAggregatorPair` owns dialog history.
- **Config:** frozen `E2LGModelSettings` dataclass instead of a mutable
  `dict` (also fixes the original's silent `tempature` typo).
- **Prompt:** rewritten for voice — drops the `</end>` token and bracketed
  room-movement commands that don't make sense over a phone call. Tour goals
  are rendered as natural-language directives instead of slot dumps.
- **Construction:** `E2LGAgent.from_scenario(...)` is a pure builder — no
  network I/O, no environment reads. The `Settings` boundary is `app.py`.
- **Performance:** streaming responses through pipecat (no per-turn
  buffering), and `enable_prompt_caching=True` keeps the static system
  instruction off the per-turn token bill.

## Prerequisites

- Python 3.12, `uv`, deps from the root `pyproject.toml`
  (`pipecat-ai[anthropic,daily,piper,silero,whisper,...]>=1.0.0`).
- A Daily account: set `DAILY_API_KEY` (server-side REST key) so the bot can
  create rooms on demand. Alternatively pre-create a room and pass it via
  `DAILY_ROOM_URL` (+ optional `DAILY_ROOM_TOKEN`).
- An Anthropic API key in `ANTHROPIC_API_KEY`.
- Scenario JSONs at `REOH_SCENARIO_DIR` (default: `<repo>/dataset/reoh/scenarios/`).

## Running

### 1. Set required env in `.env`

```bash
ANTHROPIC_API_KEY=sk-ant-...
DAILY_API_KEY=...                                  # Daily REST key
# Optional:
# DAILY_ROOM_URL=https://yourdomain.daily.co/test  # use a pre-created room instead
# REOH_SCENARIO_ID=13
# REOH_SCENARIO_DIR=dataset/reoh/scenarios         # default (relative to repo root)
```

### 2. Start the signaling server

```bash
./reoh_bot/run.sh
# Equivalent to:
# uv run uvicorn reoh_bot.app:app --host localhost --port 7861 --reload
```

### 3. Create a room and dispatch the bot (separate terminal)

```bash
curl -s -X POST http://localhost:7861/api/start | jq
# {
#   "room_url": "https://yourdomain.daily.co/abc123",
#   "expires_at": 1234567890,
#   "scenario_id": null
# }
```

To pin a particular scenario for this session:

```bash
curl -s -X POST http://localhost:7861/api/start \
  -H 'content-type: application/json' \
  -d '{"scenario_id":"13"}' | jq
```

`scenario_id` accepts either the bare index (`"13"`) or the full identifier
(`"scenario-0013"`).

### 4. Open the returned `room_url` in a browser

Daily Prebuilt loads automatically. Allow the microphone prompt; the bot
greets you the moment your participant joins.

### Health check

```bash
curl http://localhost:7861/health
# {"status":"ok","active_bots":1,"scenario_id":null}
```

### On Jetson (aarch64)

`run.sh` works on x86_64 with CUDA. On Jetson you'll need the same
`LD_PRELOAD` ordering as `smallwebrtc_bot/run-jetson.sh` (libgomp before
libc10.so) — adapt that script for `reoh_bot.app` if you deploy there.

## Environment variables

| Variable | Default | Notes |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | — | Required. |
| `DAILY_API_KEY` | — | Required unless `DAILY_ROOM_URL` is set. |
| `DAILY_ROOM_URL` | — | Skip auto-room-creation; bot joins this room instead. |
| `DAILY_ROOM_TOKEN` | — | Optional bot meeting token for the pre-existing room. |
| `DAILY_API_URL` | `https://api.daily.co/v1` | Override for self-hosted Daily. |
| `DAILY_ROOM_EXPIRY_SECONDS` | `3600` | TTL for auto-created rooms. |
| `REOH_BOT_NAME` | `REOH Agent` | Display name in the Daily call. |
| `LLM_MODEL` | `claude-haiku-4-5` | Override the Claude model. |
| `STT_MODEL` | `large-v3-turbo` | faster-whisper model id. |
| `STT_DEVICE` | `auto` | `auto`, `cuda`, or `cpu`. |
| `STT_COMPUTE_TYPE` | `int8` | quantisation for CTranslate2. |
| `PIPER_VOICE` | `en_US-ryan-high` | Piper voice id. |
| `PIPER_MODEL_DIR` | `<repo>/models/piper` | Where Piper voices are cached. |
| `REOH_SCENARIO_DIR` | `<repo>/dataset/reoh/scenarios` | Source of `scenario-*.json`. |
| `REOH_SCENARIO_ID` | `<first by sort>` | Default scenario for new sessions. |
| `REOH_PROMPT_PATH` | `reoh_bot/prompts/e2lg_system_prompt.md` | Override the system prompt. |
| `HOST` / `PORT` / `RELOAD_FLAG` | `localhost` / `7861` / `--reload` | Used by `run.sh`. |

## Tests

```bash
uv run pytest reoh_bot/tests
```

The tests are pure-functional: they exercise the prompt builder and scenario
loader without instantiating the Anthropic client or touching the network.
