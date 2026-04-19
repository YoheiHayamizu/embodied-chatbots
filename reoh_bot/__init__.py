"""Daily-transport voice agent for the Real Estate Open House (REOH) scenarios.

Sister package to ``smallwebrtc_bot`` — same pipecat pipeline shape (Whisper STT
-> LLM -> Piper TTS with Silero VAD) but routed through ``DailyTransport`` and
backed by a Claude-powered, scenario-aware E2LG (End-to-End Language Generation)
agent. The REOH dialog logic is reimplemented here; this package does not import
from the upstream ``reoh`` project at runtime.
"""
