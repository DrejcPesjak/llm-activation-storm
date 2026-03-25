const state = {
  payload: null,
  textures: [],
  stepIndex: 0,
  playing: false,
  timer: null,
};

const elements = {
  modelSelect: document.getElementById("model-select"),
  promptInput: document.getElementById("prompt-input"),
  analyzeButton: document.getElementById("analyze-button"),
  playButton: document.getElementById("play-button"),
  stepSlider: document.getElementById("step-slider"),
  stepLabel: document.getElementById("step-label"),
  stepCounter: document.getElementById("step-counter"),
  tokenStrip: document.getElementById("token-strip"),
  analysisNote: document.getElementById("analysis-note"),
  statusPill: document.getElementById("status-pill"),
  modelMeta: document.getElementById("model-meta"),
  canvas: document.getElementById("storm-canvas"),
};

const POSITIVE = [94, 234, 212];
const NEGATIVE = [249, 168, 212];
const ACTIVE = "#fef08a";

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

function stopPlayback() {
  state.playing = false;
  elements.playButton.textContent = "Play";
  if (state.timer) {
    window.clearInterval(state.timer);
    state.timer = null;
  }
}

function base64ToBytes(base64) {
  const binary = window.atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

function lerpColor(base, magnitude) {
  const boost = Math.pow(magnitude, 0.8);
  return [
    Math.round(base[0] + (255 - base[0]) * boost),
    Math.round(base[1] + (255 - base[1]) * boost),
    Math.round(base[2] + (255 - base[2]) * boost),
  ];
}

function createTexture(step) {
  const bytes = base64ToBytes(step.encoded_field);
  const canvas = document.createElement("canvas");
  canvas.width = step.cols;
  canvas.height = step.rows;
  const context = canvas.getContext("2d", { willReadFrequently: false });
  const image = context.createImageData(step.cols, step.rows);
  const hotspots = [];

  for (let offset = 0; offset < bytes.length; offset += 1) {
    const byte = bytes[offset];
    const norm = (byte / 127.5) - 1;
    const magnitude = Math.abs(norm);
    const base = norm >= 0 ? POSITIVE : NEGATIVE;
    const [r, g, b] = lerpColor(base, magnitude);
    const pixel = offset * 4;
    image.data[pixel] = r;
    image.data[pixel + 1] = g;
    image.data[pixel + 2] = b;
    image.data[pixel + 3] = Math.max(18, Math.round(255 * magnitude));

    if (magnitude > 0.84 && offset % 11 === 0) {
      const x = offset % step.cols;
      const y = Math.floor(offset / step.cols);
      hotspots.push({ x, y, norm, magnitude });
    }
  }

  context.putImageData(image, 0, 0);
  return { ...step, canvas, hotspots, bytes };
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

  state.payload.tokens.forEach((token) => {
    const chip = document.createElement("span");
    chip.className = "token-chip";
    chip.textContent = token === " " ? "␠" : token;
    elements.tokenStrip.appendChild(chip);
  });
}

function setStep(index) {
  if (!state.payload) {
    return;
  }

  state.stepIndex = Math.max(0, Math.min(index, state.textures.length - 1));
  elements.stepSlider.value = String(state.stepIndex);
  const step = state.textures[state.stepIndex];
  elements.stepLabel.textContent = `Step: Layer ${step.layer_index + 1} • ${step.stage_label}`;
  elements.stepCounter.textContent = `${state.stepIndex + 1} / ${state.textures.length}`;
  render();
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
    const nextIndex = (state.stepIndex + 1) % state.textures.length;
    setStep(nextIndex);
  }, 260);
}

function drawBackdrop(context, width, height) {
  const sky = context.createLinearGradient(0, 0, 0, height);
  sky.addColorStop(0, "rgba(6, 16, 28, 0.96)");
  sky.addColorStop(0.55, "rgba(3, 11, 20, 0.98)");
  sky.addColorStop(1, "rgba(1, 6, 12, 1)");
  context.fillStyle = sky;
  context.fillRect(0, 0, width, height);

  const glow = context.createRadialGradient(width * 0.5, height * 1.05, 20, width * 0.5, height * 1.05, width * 0.55);
  glow.addColorStop(0, "rgba(105, 143, 224, 0.42)");
  glow.addColorStop(0.45, "rgba(47, 86, 146, 0.15)");
  glow.addColorStop(1, "rgba(0, 0, 0, 0)");
  context.fillStyle = glow;
  context.fillRect(0, 0, width, height);
}

function panelX(index, overview) {
  const stagesPerLayer = state.payload.model.stage_sequence.length;
  const layerIndex = Math.floor(index / stagesPerLayer);
  const gap = 2;
  const layerGap = 7;
  return overview.x + index * (overview.unitWidth + gap) + layerIndex * layerGap;
}

function drawOverview(context, overview) {
  const steps = state.textures;
  const stagesPerLayer = state.payload.model.stage_sequence.length;
  const totalGap = (steps.length - 1) * 2 + (state.payload.model.layer_count - 1) * 7;
  overview.unitWidth = Math.max((overview.width - totalGap) / steps.length, 3.5);

  context.save();
  context.strokeStyle = "rgba(155, 212, 255, 0.08)";
  context.strokeRect(overview.x, overview.y, overview.width, overview.height);

  steps.forEach((step, index) => {
    const x = panelX(index, overview);
    const y = overview.y + 24;
    const width = overview.unitWidth;
    const height = overview.height - 36;

    context.globalAlpha = index === state.stepIndex ? 1 : 0.3;
    context.imageSmoothingEnabled = false;
    context.drawImage(step.canvas, x, y, width, height);

    if ((index + 1) % stagesPerLayer === 0 && index < steps.length - 1) {
      context.globalAlpha = 1;
      context.fillStyle = "rgba(149, 192, 255, 0.08)";
      context.fillRect(x + width + 3, y - 10, 1, height + 20);
    }

    if (index % stagesPerLayer === 0) {
      context.globalAlpha = 1;
      context.fillStyle = "rgba(228, 243, 255, 0.52)";
      context.font = "11px IBM Plex Sans";
      context.fillText(`L${String(step.layer_index + 1).padStart(2, "0")}`, x, overview.y + 14);
    }
  });

  const activeX = panelX(state.stepIndex, overview);
  const activeWidth = overview.unitWidth;
  context.globalAlpha = 1;
  context.shadowColor = ACTIVE;
  context.shadowBlur = 20;
  context.strokeStyle = ACTIVE;
  context.lineWidth = 2;
  context.strokeRect(activeX - 2, overview.y + 20, activeWidth + 4, overview.height - 28);
  context.restore();
}

function drawHotspots(context, detail, step) {
  if (!step.hotspots.length) {
    return;
  }

  context.save();
  const count = Math.min(140, step.hotspots.length);

  for (let index = 0; index < count; index += 1) {
    const hotspot = step.hotspots[(index * 29 + state.stepIndex * 17) % step.hotspots.length];
    const baseX = detail.x + ((hotspot.x + 0.5) / step.cols) * detail.width;
    const baseY = detail.y + ((hotspot.y + 0.5) / Math.max(step.rows, 1)) * detail.height;
    const color = hotspot.norm >= 0 ? "rgba(94, 234, 212, 0.34)" : "rgba(249, 168, 212, 0.34)";
    const radius = 1.2 + hotspot.magnitude * 3.8;

    context.fillStyle = color;
    context.shadowColor = color;
    context.shadowBlur = 10 + hotspot.magnitude * 10;
    context.beginPath();
    context.arc(baseX, baseY, radius, 0, Math.PI * 2);
    context.fill();
  }

  context.restore();
}

function drawDetail(context, detail) {
  const step = state.textures[state.stepIndex];
  context.save();
  context.fillStyle = "rgba(10, 24, 41, 0.72)";
  context.fillRect(detail.x - 1, detail.y - 1, detail.width + 2, detail.height + 2);
  context.imageSmoothingEnabled = false;
  context.drawImage(step.canvas, detail.x, detail.y, detail.width, detail.height);
  drawHotspots(context, detail, step);

  context.fillStyle = "rgba(228, 243, 255, 0.92)";
  context.font = "600 18px IBM Plex Sans";
  context.fillText(`Layer ${step.layer_index + 1} • ${step.stage_label}`, detail.x, detail.y - 18);
  context.font = "12px IBM Plex Sans";
  context.fillStyle = "rgba(169, 207, 241, 0.72)";
  context.fillText(`${step.rows} tokens × ${step.cols} hidden dims`, detail.x, detail.y + detail.height + 18);

  const tokenLabelCount = Math.min(state.payload.tokens.length, 12);
  for (let index = 0; index < tokenLabelCount; index += 1) {
    const y = detail.y + ((index + 0.5) / tokenLabelCount) * detail.height;
    context.fillStyle = "rgba(212, 233, 255, 0.5)";
    context.fillText(state.payload.tokens[index], detail.x - 92, y + 4);
  }
  context.restore();
}

function render() {
  const context = elements.canvas.getContext("2d");
  const width = elements.canvas.clientWidth;
  const height = elements.canvas.clientHeight;

  context.clearRect(0, 0, width, height);
  drawBackdrop(context, width, height);

  if (!state.payload) {
    context.fillStyle = "rgba(228, 243, 255, 0.56)";
    context.font = "500 20px IBM Plex Sans";
    context.fillText("Run a prompt to map the layer-by-layer flow.", 28, 42);
    return;
  }

  const overview = {
    x: 24,
    y: 28,
    width: width - 48,
    height: height * 0.34,
    unitWidth: 0,
  };
  const detail = {
    x: 118,
    y: height * 0.47,
    width: width - 154,
    height: height * 0.4,
  };

  drawOverview(context, overview);
  drawDetail(context, detail);
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
    elements.modelMeta.textContent = `${active.label} • ${active.layer_count} layers • width ${active.layer_width} • ${active.stage_sequence.join(" → ")}`;
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
    state.textures = payload.steps.map(createTexture);
    elements.stepSlider.disabled = false;
    elements.playButton.disabled = false;
    elements.stepSlider.max = String(Math.max(state.textures.length - 1, 0));
    elements.analysisNote.textContent = payload.token_limit_applied
      ? `Showing the first ${payload.tokens.length} content tokens for the dense flow view.`
      : `${payload.tokens.length} content tokens across ${payload.steps.length} stage steps.`;
    renderTokenStrip();
    setStep(0);
    setStatus("Ready", "ready");
  } catch (error) {
    setStatus("Error", "busy");
    elements.stepLabel.textContent = `Step: ${error.message}`;
    elements.stepCounter.textContent = "0 / 0";
    elements.analysisNote.textContent = "";
    state.payload = null;
    state.textures = [];
    render();
  } finally {
    elements.analyzeButton.disabled = false;
  }
}

elements.analyzeButton.addEventListener("click", analyzePrompt);
elements.playButton.addEventListener("click", togglePlayback);
elements.stepSlider.addEventListener("input", (event) => {
  setStep(Number(event.target.value));
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
