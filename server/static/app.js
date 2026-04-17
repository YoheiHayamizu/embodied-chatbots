// Browser client for the pipecat SmallWebRTC voice agent.
//
// Loads the Pipecat JS client and SmallWebRTC transport from esm.sh so the
// page has no build step. The server broadcasts RTVI events over the data
// channel; this script translates the events we care about into DOM updates:
// user transcript, bot output, speaking indicator, and a connection status
// pill.

const connectBtn = document.getElementById("connect-btn");
const statusEl = document.getElementById("status");
const statusLabelEl = statusEl.querySelector(".label");
const micEl = document.getElementById("mic-indicator");
const micLabelEl = document.getElementById("mic-label-state");
const transcriptEl = document.getElementById("transcript");
const greetStateEl = document.getElementById("greet-state");
const botAudioEl = document.getElementById("bot-audio");

const STATUS_LABELS = {
  idle: "Idle",
  connecting: "Connecting",
  ready: "Ready",
  speaking: "Bot speaking",
  error: "Error",
};

function setStatus(state, extra) {
  statusEl.dataset.state = state;
  statusLabelEl.textContent = extra
    ? `${STATUS_LABELS[state]} — ${extra}`
    : STATUS_LABELS[state];
}

function setMicActive(active) {
  micEl.dataset.active = active ? "true" : "false";
  micLabelEl.textContent = active ? "Listening" : "Inactive";
}

function setButtonState(mode) {
  if (mode === "idle") {
    connectBtn.textContent = "Connect";
    connectBtn.dataset.variant = "";
    connectBtn.disabled = false;
  } else if (mode === "connecting") {
    connectBtn.textContent = "Connecting…";
    connectBtn.dataset.variant = "";
    connectBtn.disabled = true;
  } else if (mode === "connected") {
    connectBtn.textContent = "Disconnect";
    connectBtn.dataset.variant = "danger";
    connectBtn.disabled = false;
  }
}

function scrollTranscriptToEnd() {
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

function makeBubble({ who, text, partial }) {
  const bubble = document.createElement("div");
  bubble.className = `bubble ${who}`;
  if (partial) bubble.dataset.partial = "true";
  const label = document.createElement("span");
  label.className = "who";
  label.textContent = who === "user" ? "You" : "Agent";
  const body = document.createElement("span");
  body.className = "text";
  body.textContent = text;
  bubble.append(label, body);
  transcriptEl.appendChild(bubble);
  scrollTranscriptToEnd();
  return bubble;
}

// Load the pipecat JS client modules dynamically so any import error is
// caught and surfaced in the UI instead of silently killing the script.
let PipecatClient = null;
let SmallWebRTCTransport = null;
let RTVIEvent = null;

const CLIENT_JS_URL =
  "https://esm.sh/@pipecat-ai/client-js@1.7.0?bundle";
const TRANSPORT_URL =
  "https://esm.sh/@pipecat-ai/small-webrtc-transport@1.10.0?bundle&deps=@pipecat-ai/client-js@1.7.0";

async function loadPipecatModules() {
  const [core, transport] = await Promise.all([
    import(CLIENT_JS_URL),
    import(TRANSPORT_URL),
  ]);
  PipecatClient = core.PipecatClient;
  RTVIEvent = core.RTVIEvent;
  SmallWebRTCTransport = transport.SmallWebRTCTransport;
  if (!PipecatClient || !RTVIEvent || !SmallWebRTCTransport) {
    throw new Error("Pipecat modules loaded but exports are missing.");
  }
}

let client = null;
let connecting = false;
let partialUserBubble = null;
let activeBotBubble = null;

function resetTranscriptState() {
  partialUserBubble = null;
  activeBotBubble = null;
}

function renderUserTranscript({ text, final }) {
  if (!text) return;
  if (partialUserBubble) {
    partialUserBubble.querySelector(".text").textContent = text;
    if (final) {
      delete partialUserBubble.dataset.partial;
      partialUserBubble = null;
    }
  } else {
    const bubble = makeBubble({ who: "user", text, partial: !final });
    if (!final) partialUserBubble = bubble;
  }
  scrollTranscriptToEnd();
}

function appendBotText(chunk) {
  if (!chunk) return;
  if (!activeBotBubble) {
    activeBotBubble = makeBubble({ who: "bot", text: chunk, partial: true });
    return;
  }
  const node = activeBotBubble.querySelector(".text");
  node.textContent = `${node.textContent}${chunk}`;
  scrollTranscriptToEnd();
}

function finaliseBotBubble() {
  if (!activeBotBubble) return;
  delete activeBotBubble.dataset.partial;
  activeBotBubble = null;
}

function attachRemoteAudio(track) {
  if (!track || track.kind !== "audio") return;
  const stream = new MediaStream([track]);
  botAudioEl.srcObject = stream;
  const play = botAudioEl.play();
  if (play && typeof play.catch === "function") {
    play.catch((err) => {
      console.warn("Autoplay blocked; will retry on next user gesture", err);
    });
  }
}

function detachRemoteAudio() {
  botAudioEl.srcObject = null;
}

function attachHandlers(pc) {
  pc.on(RTVIEvent.BotReady, () => {
    setStatus("ready");
    greetStateEl.textContent = "bot ready";
  });

  pc.on(RTVIEvent.Disconnected, () => {
    setStatus("idle");
    setMicActive(false);
    setButtonState("idle");
    greetStateEl.textContent = "—";
    resetTranscriptState();
    detachRemoteAudio();
  });

  pc.on(RTVIEvent.TrackStarted, (track, participant) => {
    if (participant?.local) return;
    if (track?.kind !== "audio") return;
    console.info("Attaching remote audio track", track.id);
    attachRemoteAudio(track);
  });

  pc.on(RTVIEvent.UserStartedSpeaking, () => setMicActive(true));
  pc.on(RTVIEvent.UserStoppedSpeaking, () => setMicActive(false));

  pc.on(RTVIEvent.BotStartedSpeaking, () => setStatus("speaking"));
  pc.on(RTVIEvent.BotStoppedSpeaking, () => {
    setStatus("ready");
    finaliseBotBubble();
  });

  pc.on(RTVIEvent.UserTranscript, (data) => renderUserTranscript(data || {}));

  // BotOutput fires once per source (TTS-spoken and LLM-aggregated), and the
  // deprecated BotTranscript event duplicates the same text again. Only
  // render the sentence-level text that is actually spoken to avoid 2x or 3x
  // duplicates in the transcript pane.
  pc.on(RTVIEvent.BotOutput, (data) => {
    if (!data?.text) return;
    if (data.aggregated_by !== "sentence") return;
    if (data.spoken === false) return;
    appendBotText(`${data.text} `);
  });

  pc.on(RTVIEvent.Error, (err) => {
    console.error("RTVI error", err);
    setStatus("error", err?.message || "see console");
  });
}

async function connect() {
  if (client || connecting) return;
  connecting = true;
  setStatus("connecting");
  setButtonState("connecting");

  try {
    if (!PipecatClient) await loadPipecatModules();

    client = new PipecatClient({
      transport: new SmallWebRTCTransport({
        iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
      }),
      enableCam: false,
      enableMic: true,
    });
    attachHandlers(client);
    await client.connect({ webrtcUrl: "/api/offer" });
    setButtonState("connected");
    greetStateEl.textContent = "waiting for bot";
  } catch (err) {
    console.error("Failed to connect", err);
    setStatus("error", err?.message || "connection failed");
    setButtonState("idle");
    if (client) {
      try {
        await client.disconnect();
      } catch {
        /* ignore */
      }
      client = null;
    }
  } finally {
    connecting = false;
  }
}

async function disconnect() {
  if (!client) return;
  setStatus("idle", "disconnecting");
  try {
    await client.disconnect();
  } catch (err) {
    console.warn("Disconnect raised", err);
  }
  client = null;
  setMicActive(false);
  setButtonState("idle");
  setStatus("idle");
  greetStateEl.textContent = "—";
  resetTranscriptState();
}

connectBtn.addEventListener("click", () => {
  if (client) {
    disconnect();
  } else {
    connect();
  }
});

// Preload modules after idle so the first click feels instant and any CDN
// failure surfaces in the status pill before the user clicks Connect.
(async () => {
  try {
    await loadPipecatModules();
  } catch (err) {
    console.error("Failed to load Pipecat client modules", err);
    setStatus("error", "client SDK load failed — see console");
    connectBtn.disabled = true;
  }
})();

window.addEventListener("beforeunload", () => {
  if (client) {
    client.disconnect().catch(() => {});
  }
});
