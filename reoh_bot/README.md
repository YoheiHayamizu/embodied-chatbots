# reoh_bot — REOH Daily Voice Bot

A pipecat voice agent that runs an end-to-end Real Estate Open House (REOH) tour
over a [Daily](https://daily.co) room. Same pipeline shape as `smallwebrtc_bot/`
(Whisper STT → LLM → Piper TTS, with Silero VAD), but:

| | `smallwebrtc_bot/` | `reoh_bot/` |
| --- | --- | --- |
| Transport | SmallWebRTC (browser ⇄ FastAPI) | Daily (cloud room) |
| Client | Bundled static page | Any Daily client (Prebuilt, app, SDK) |
| LLM | Provider-agnostic factory | Anthropic Claude, scenario-bound |
| Dialog logic | Generic assistant | REOH **E2LG** or **SSLG** agent (selectable) |

## Layout

```
reoh_bot/
├── __init__.py
├── README.md
├── app.py                       # FastAPI: POST /api/start, GET /health
├── bot.py                       # Pipecat pipeline (Daily + Whisper + Claude + Piper + Silero)
├── config.py                    # Frozen-dataclass settings, env-driven
├── e2lg_agent.py                # End-to-End Language Generation agent
├── sslg_agent.py                # Strategy-conditioned SLG agent (persona-aware)
├── persona.py                   # Persona dataclass, strategies, selector, directive
├── persona_extractor.py         # Synchronous Claude extractor for persona deltas
├── persona_processor.py         # Pipecat FrameProcessor that injects per-turn directives
├── scenarios.py                 # Scenario JSON loader + property/goal renderers
├── daily_session.py             # Async REST helper for Daily room/token creation
├── arrival_gate.py              # Operator-driven wait_for_arrival gate
├── run.sh                       # uv run uvicorn launcher
└── prompts/
    ├── e2lg_system_prompt.md
    ├── sslg_system_prompt.md
    └── persona_extractor_prompt.md
```

## Agent kinds: E2LG vs SSLG

Select the dialog agent with `REOH_AGENT_KIND`:

- `e2lg` (default) — End-to-End Language Generation. One Claude call per turn.
  The system prompt is static (cached); the model's decisions about what to
  say are driven entirely by the prompt and the running dialog history.
- `sslg` — Strategy-conditioned SLG. Adds a persona-tracking layer. Before
  each reply, a second (smaller) Claude call reads the visitor's latest
  utterance and updates an in-memory persona snapshot. A weighted
  strategy selector then picks one of `logical_appeal`, `emotional_appeal`,
  `self_modeling`, `personal_story`, `highlight_unique_feature`,
  `personal_related_inquiry`, or `interest_level_inquiry`, and a short
  natural-language directive describing that approach is appended to the
  LLM context as a `developer`-role message. The main system prompt stays
  cached unchanged.

Extraction is synchronous with the voice turn — expect ~300–800 ms added
latency per turn when SSLG is enabled. Short utterances
(`< PERSONA_MIN_UTTERANCE_TOKENS`, default 3) skip extraction to avoid
burning a Claude call on backchannels like "uh-huh".

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

### On Jetson (aarch64, JetPack 5)

Use `./reoh_bot/run-jetson.sh` instead of `run.sh`. It does the same
`LD_PRELOAD` dance as `smallwebrtc_bot/run-jetson.sh` to work around two
aarch64-specific failures:

- ctranslate2's bundled libgomp throws *"cannot allocate memory in static TLS
  block"* when loaded lazily — preload it first.
- JetPack-built torch wheels need `libc10.so` made globally visible before
  torchaudio (or anything else) pulls torch in — preload it second.

STT defaults to CPU on aarch64 because the PyPI ctranslate2 wheels for
aarch64 are built without CUDA support. The script defaults to
`STT_MODEL=tiny.en` (~5x faster than `small`) and bumps
`USER_TURN_STOP_TIMEOUT=30` so the aggregator waits long enough for the
slow CPU inference to finish before declaring the turn over.

If the bot greets you but never replies after that, two failure modes are
common:

1. **Slow STT, late transcript.** Logs show `User stopped speaking
   (strategy: None)` with no LLM call after. STT finished after the turn
   timeout and the transcript was discarded. Lower `STT_MODEL` or raise
   `USER_TURN_STOP_TIMEOUT`.
2. **Empty transcript.** Logs show `WhisperSTTService#0 processing time:
   X.Xs` with no `Transcription: [...]` line after. Whisper flagged the
   audio as silence and emitted nothing, leaving the user-aggregator
   stuck. Raise `STT_NO_SPEECH_PROB` (toward 1.0) so more borderline
   segments are accepted.

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
| `STT_NO_SPEECH_PROB` | `0.6` | Whisper silence threshold. Raise (toward 1.0) if utterances are being dropped. |
| `PIPER_VOICE` | `en_US-ryan-high` | Piper voice id. |
| `PIPER_MODEL_DIR` | `<repo>/models/piper` | Where Piper voices are cached. |
| `REOH_SCENARIO_DIR` | `<repo>/dataset/reoh/scenarios` | Source of `scenario-*.json`. |
| `REOH_SCENARIO_ID` | `<first by sort>` | Default scenario for new sessions. |
| `USER_TURN_STOP_TIMEOUT` | `8.0` | Seconds the aggregator waits before declaring a turn over. **Must exceed STT processing time** — bump to 30 on Jetson. |
| `USER_SPEECH_TIMEOUT` | `0.6` | Extra seconds the strategy waits after VAD-stop before committing. |
| `VAD_STOP_SECS` | `0.8` | Seconds of continuous silence VAD requires before declaring "user stopped". Together with `USER_SPEECH_TIMEOUT` this is the total breath-pause budget between sentences. |
| `REOH_PROMPT_PATH` | `reoh_bot/prompts/e2lg_system_prompt.md` | Override the E2LG system prompt. |
| `REOH_AGENT_KIND` | `e2lg` | `e2lg` or `sslg`. Case-insensitive. |
| `REOH_SSLG_PROMPT_PATH` | `reoh_bot/prompts/sslg_system_prompt.md` | Override the SSLG system prompt. |
| `REOH_PERSONA_ENABLED` | `true` | Set to `false` to skip extraction entirely even when `REOH_AGENT_KIND=sslg` (useful for measuring the overhead). |
| `PERSONA_EXTRACTOR_MODEL` | `claude-haiku-4-5` | Model used for the persona-extraction sub-call. |
| `PERSONA_EXTRACTOR_MAX_TOKENS` | `256` | Cap on the extractor's JSON reply. |
| `PERSONA_EXTRACTOR_TIMEOUT_S` | `4.0` | Hard timeout on the extractor call. On timeout the prior persona is kept. |
| `PERSONA_MIN_UTTERANCE_TOKENS` | `3` | Utterances shorter than this skip extraction. |
| `PERSONA_EXTRACTOR_PROMPT_PATH` | `reoh_bot/prompts/persona_extractor_prompt.md` | Override the extractor system prompt. |
| `STRATEGY_WEIGHTS_JSON` | — | Optional JSON object of `{strategy_name: weight}` overriding the default weights. |
| `PERSONA_SELECTOR_SEED` | — | Optional integer seed for deterministic strategy selection (tests). |
| `HOST` / `PORT` / `RELOAD_FLAG` | `localhost` / `7861` / `--reload` | Used by `run.sh`. |

## Tests

```bash
uv run pytest reoh_bot/tests
```

The tests are pure-functional: they exercise the prompt builder and scenario
loader without instantiating the Anthropic client or touching the network.
