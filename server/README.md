# Embodied Chatbots — SmallWebRTC Voice Agent Server

A local development server that streams voice audio between a browser and a
pipecat pipeline (Whisper STT → LLM → Piper TTS) over WebRTC. The pipeline is
the same shape as the local-audio reference in `main.py`, but audio enters and
leaves through a browser-driven `SmallWebRTCTransport` instead of the host's
soundcard.

## Layout

```
server/
├── __init__.py
├── app.py              # FastAPI app: /api/offer, /health, static mount
├── bot.py              # Pipeline builder + run_bot(connection)
├── llm_factory.py      # Anthropic / OpenAI / Google switch + shared prompt
├── run.sh              # uv run uvicorn launcher
├── static/
│   ├── index.html      # Browser UI
│   └── app.js          # Pipecat JS client wiring (esm.sh, no build step)
└── tests/
    └── test_llm_factory.py
```

## Prerequisites

- Python 3.12, `uv` (already used at repo root).
- Dependencies installed via the root `pyproject.toml`
  (`pipecat-ai[webrtc,whisper,piper,silero,anthropic,openai,google,runner]`,
  `fastapi`, `uvicorn`, `piper-tts`, `torch`).
- A provider API key in `.env` at the repo root (see `env.example`), at least
  one of `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY` depending on
  `LLM_PROVIDER`.
- A Chromium-based browser (Chrome/Edge) or Firefox at `http://localhost:7860`.
  Browsers only grant microphone permission to secure origins; `localhost` is
  considered secure. Any other origin needs HTTPS.

## Running

```bash
./server/run.sh
# Equivalent to:
# uv run uvicorn server.app:app --host localhost --port 7860 --reload
```

Open `http://localhost:7860`, click **Connect**, accept the microphone prompt,
and the bot will greet you when the pipeline is ready. Transcript bubbles
appear as RTVI messages stream in over the data channel.

## Environment variables

| Variable | Default | Notes |
| --- | --- | --- |
| `LLM_PROVIDER` | `anthropic` | One of `anthropic`, `openai`, `google`. |
| `LLM_MODEL` | provider default | Overrides the default model for the chosen provider. |
| `ANTHROPIC_API_KEY` | — | Required when `LLM_PROVIDER=anthropic`. |
| `OPENAI_API_KEY` | — | Required when `LLM_PROVIDER=openai`. |
| `GOOGLE_API_KEY` | — | Required when `LLM_PROVIDER=google`. |
| `STT_MODEL` | `large-v3-turbo` | Faster-whisper model identifier. |
| `STT_DEVICE` | `auto` | `auto`, `cuda`, or `cpu`. |
| `STT_COMPUTE_TYPE` | `int8` | `int8`, `int8_float16`, `float16`, `float32`. |
| `PIPER_VOICE` | `en_US-ryan-high` | Piper voice identifier. |
| `PIPER_MODEL_DIR` | `../models/piper` | Where Piper voices are cached. |
| `HOST` / `PORT` / `RELOAD_FLAG` | `localhost` / `7860` / `--reload` | Consumed by `run.sh`. |

## Provider defaults

| Provider | Default model |
| --- | --- |
| `anthropic` | `claude-haiku-4-5` |
| `openai` | `gpt-4o-mini` |
| `google` | `gemini-2.0-flash` |

Override with `LLM_MODEL=<name>`.

## Architectural notes

- `SmallWebRTCRequestHandler` is configured in `ConnectionMode.SINGLE`. A second
  concurrent browser tab will be rejected with HTTP 400 until the first one
  closes. This matches the "one robot, one user" intent.
- The opening greeting is queued inside the RTVI `on_client_ready` handler. If
  we instead queued it at pipeline build time, the first audio frames could be
  produced before the data channel was open and the browser would miss them.
- The pipeline relies on the browser's built-in echo cancellation
  (`getUserMedia` default constraints). There is no server-side audio gate in
  this build. For far-field speaker/microphone setups (e.g. on-device robot
  deployment) a server-side gate will need to be reintroduced.
- Piper and faster-whisper both manage their own model caches. First run will
  download the selected voice/model; point `PIPER_MODEL_DIR` elsewhere if you
  want to bake models into an image.

## Tests

```bash
uv run pytest server/tests
```

The existing `test_llm_factory.py` covers provider dispatch without hitting any
network. Add FastAPI route tests here if the surface grows.

## Troubleshooting

- **No bot audio** — check browser console for RTVI errors. Confirm a valid API
  key for the selected provider; the bot fails to build if the factory raises.
- **Microphone blocked** — browser requires a secure origin. Use `localhost` or
  serve behind HTTPS via a reverse proxy.
- **`400 PC ID mismatch`** — you have two tabs open. Close one; single-connection
  mode is intentional.
- **STT latency is high** — try `STT_MODEL=distil-medium.en` or set
  `STT_DEVICE=cuda` if a compatible GPU is available on the host.
