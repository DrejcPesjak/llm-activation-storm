const state = {
  payload: null,
  tokenIndex: 0,
  playing: false,
  timer: null,
};

const elements = {
  modelSelect: document.getElementById("model-select"),
  promptInput: document.getElementById("prompt-input"),
  analyzeButton: document.getElementById("analyze-button"),
  playButton: document.getElementById("play-button"),
  tokenSlider: document.getElementById("token-slider"),
  tokenLabel: document.getElementById("token-label"),
  tokenCounter: document.getElementById("token-counter"),
  tokenStrip: document.getElementById("token-strip"),
  statusPill: document.getElementById("status-pill"),
  modelMeta: document.getElementById("model-meta"),
  canvas: document.getElementById("storm-canvas"),
};

const FAMILY_COLORS = {
  resid: "#5eead4",
  attn: "#7dd3fc",
  mlp: "#f9a8d4",
};

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Request failed");
  }
  return payload;
}

function setStatus(label, className) {
  elements.statusPill.textContent = label;
  elements.statusPill.className = `pill ${className}`;
}

function resizeCanvas() {
  const ratio = window.devicePixelRatio || 1;
  const rect = elements.canvas.getBoundingClientRect();
  elements.canvas.width = Math.floor(rect.width * ratio);
  elements.canvas.height = Math.floor(rect.height * ratio);
  const context = elements.canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  render();
}

function renderTokenStrip() {
  elements.tokenStrip.innerHTML = "";
  if (!state.payload) {
    return;
  }

  state.payload.tokens.forEach((token, index) => {
    const chip = document.createElement("span");
    chip.className = `token-chip ${index === state.tokenIndex ? "active" : ""}`;
    chip.textContent = token === " " ? "␠" : token;
    chip.addEventListener("click", () => setToken(index));
    elements.tokenStrip.appendChild(chip);
  });
}

function setToken(index) {
  if (!state.payload) {
    return;
  }

  state.tokenIndex = Math.max(0, Math.min(index, state.payload.frames.length - 1));
  elements.tokenSlider.value = String(state.tokenIndex);
  const tokenText = state.payload.tokens[state.tokenIndex] || "—";
  elements.tokenLabel.textContent = `Token: ${tokenText}`;
  elements.tokenCounter.textContent = `${state.tokenIndex + 1} / ${state.payload.tokens.length}`;
  renderTokenStrip();
  render();
}

function stopPlayback() {
  state.playing = false;
  elements.playButton.textContent = "Play";
  if (state.timer) {
    window.clearInterval(state.timer);
    state.timer = null;
  }
}

function togglePlayback() {
  if (!state.payload) {
    return;
  }

  if (state.playing) {
    stopPlayback();
    return;
  }

  state.playing = true;
  elements.playButton.textContent = "Pause";
  state.timer = window.setInterval(() => {
    const nextIndex = (state.tokenIndex + 1) % state.payload.frames.length;
    setToken(nextIndex);
  }, 440);
}

function drawStormBackdrop(context, width, height) {
  const sky = context.createLinearGradient(0, 0, 0, height);
  sky.addColorStop(0, "rgba(8, 20, 34, 0.9)");
  sky.addColorStop(0.55, "rgba(2, 12, 22, 0.95)");
  sky.addColorStop(1, "rgba(1, 8, 15, 1)");
  context.fillStyle = sky;
  context.fillRect(0, 0, width, height);

  const horizon = context.createRadialGradient(width * 0.5, height * 1.08, 30, width * 0.5, height * 1.08, width * 0.65);
  horizon.addColorStop(0, "rgba(92, 135, 209, 0.45)");
  horizon.addColorStop(0.45, "rgba(36, 76, 127, 0.18)");
  horizon.addColorStop(1, "rgba(0, 0, 0, 0)");
  context.fillStyle = horizon;
  context.fillRect(0, 0, width, height);

  context.strokeStyle = "rgba(146, 204, 255, 0.08)";
  for (let i = 1; i <= 8; i += 1) {
    const y = (height / 9) * i;
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(width, y);
    context.stroke();
  }
}

function drawFamily(context, family, values, familyIndex, width, height) {
  const color = FAMILY_COLORS[family];
  const laneTop = (height / 3) * familyIndex;
  const laneHeight = height / 3;
  const xStep = width / Math.max(values.length, 1);

  context.save();
  context.fillStyle = color;
  context.shadowColor = color;
  context.shadowBlur = 24;

  values.forEach((value, layerIndex) => {
    const x = (layerIndex + 0.5) * xStep;
    const y = laneTop + laneHeight * (1 - value * 0.86) - laneHeight * 0.08;
    const radius = 8 + value * 18;
    const alpha = 0.08 + value * 0.75;

    context.globalAlpha = alpha;
    context.beginPath();
    context.arc(x, y, radius, 0, Math.PI * 2);
    context.fill();

    if (value > 0.2) {
      context.lineWidth = 1 + value * 2.4;
      context.strokeStyle = color;
      context.globalAlpha = alpha * 0.55;
      context.beginPath();
      context.moveTo(x, laneTop + laneHeight + 6);
      context.lineTo(x + (Math.sin(layerIndex) * 14), y);
      context.stroke();
    }
  });

  context.restore();

  context.fillStyle = "rgba(229, 243, 255, 0.7)";
  context.font = "600 14px IBM Plex Sans";
  context.fillText(family.toUpperCase(), 18, laneTop + 22);
}

function render() {
  const context = elements.canvas.getContext("2d");
  const width = elements.canvas.clientWidth;
  const height = elements.canvas.clientHeight;

  context.clearRect(0, 0, width, height);
  drawStormBackdrop(context, width, height);

  if (!state.payload) {
    context.fillStyle = "rgba(229, 243, 255, 0.55)";
    context.font = "500 18px IBM Plex Sans";
    context.fillText("Run a prompt to render the storm.", 24, 36);
    return;
  }

  const frame = state.payload.frames[state.tokenIndex];
  state.payload.families.forEach((family, index) => {
    drawFamily(context, family, frame.values[family], index, width, height);
  });
}

async function loadModels() {
  const payload = await fetchJson("/api/models");
  elements.modelSelect.innerHTML = "";
  payload.models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.id;
    option.textContent = model.label;
    elements.modelSelect.appendChild(option);
  });
  elements.modelSelect.value = payload.default_model;
  const active = payload.models.find((model) => model.id === payload.default_model);
  if (active) {
    elements.modelMeta.textContent = `${active.label} • ${active.layer_count} layers • width ${active.layer_width}`;
  }
}

async function analyzePrompt() {
  stopPlayback();
  setStatus("Analyzing", "busy");
  elements.analyzeButton.disabled = true;

  try {
    const payload = await fetchJson("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model_id: elements.modelSelect.value,
        prompt: elements.promptInput.value,
      }),
    });

    state.payload = payload;
    elements.tokenSlider.disabled = false;
    elements.playButton.disabled = false;
    elements.tokenSlider.max = String(Math.max(payload.frames.length - 1, 0));
    setToken(0);
    setStatus("Ready", "ready");
  } catch (error) {
    setStatus("Error", "busy");
    elements.tokenLabel.textContent = `Token: ${error.message}`;
    elements.tokenCounter.textContent = "0 / 0";
    state.payload = null;
    render();
  } finally {
    elements.analyzeButton.disabled = false;
  }
}

elements.analyzeButton.addEventListener("click", analyzePrompt);
elements.playButton.addEventListener("click", togglePlayback);
elements.tokenSlider.addEventListener("input", (event) => {
  setToken(Number(event.target.value));
});
window.addEventListener("resize", resizeCanvas);

loadModels()
  .then(() => {
    setStatus("Idle", "idle");
    resizeCanvas();
  })
  .catch((error) => {
    setStatus("Error", "busy");
    elements.modelMeta.textContent = error.message;
    resizeCanvas();
  });
