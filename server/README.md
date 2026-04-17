# Voice Agent Server

A Pipecat-based voice agent exposed over WebRTC. Runs local Whisper STT and Piper TTS; delegates LLM work to a pluggable cloud provider.

## Components

```
Browser / Jetson (aiortc)
  ──WebRTC (audio, SDP via /api/offer)──▶
      FastAPI (app.py)
        └─ SmallWebRTCTransport
             └─ Pipeline
                  ├─ Whisper STT  (local, Faster Whisper)
                  ├─ LLM          (cloud: Claude / GPT / Gemini)
                  └─ Piper TTS    (local)
```

## Supported LLM providers

Controlled via the `LLM_PROVIDER`.

| `LLM_PROVIDER` | Default model         | API key env var    |
| -------------- | --------------------- | ------------------ |
| `anthropic`    | `claude-haiku-4-5`    | `ANTHROPIC_API_KEY`|
| `openai`       | `gpt-4o-mini`         | `OPENAI_API_KEY`   |
| `google`       | `gemini-2.0-flash`    | `GOOGLE_API_KEY`   |

Override the model with `LLM_MODEL` environment variable if needed.

## Setup

From the repository root:

```bash
uv add "pipecat-ai[silero,whisper,webrtc,anthropic,openai,google]" \
       piper-tts fastapi uvicorn python-dotenv
```

Copy the env template and fill in one provider's key:

```bash
cp env.example .env
# edit .env
```

## Run

```bash
cd server
uv run python app.py
```

Then open <http://localhost:7860> in a browser and click **Connect**.

On first launch:

- Whisper downloads the `distil-medium.en` weights.
- Piper downloads the `en_US-ryan-high` voice into `../models/piper/`.

Subsequent runs are fast.

## Switching providers

```bash
LLM_PROVIDER=openai uv run python app.py
LLM_PROVIDER=google LLM_MODEL=gemini-2.0-flash-exp uv run python app.py
```

## Health check

```bash
curl http://localhost:7860/health
```

## Next steps

1. Validate the flow from a laptop browser (browser AEC masks the echo problem during development).
2. Move the client to a Python `aiortc` implementation running on the Jetson.
3. Add PipeWire `module-echo-cancel` on the Jetson once a physical speaker replaces headphones.
