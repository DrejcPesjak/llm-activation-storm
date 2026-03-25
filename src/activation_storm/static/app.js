const state = {
  payload: null,
  allTextures: [],
  visibleTextures: [],
  visibleIndex: 0,
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
  toggleAttn: document.getElementById("toggle-attn"),
  toggleResid: document.getElementById("toggle-resid"),
  toggleMlp: document.getElementById("toggle-mlp"),
  toggleSpecial: document.getElementById("toggle-special"),
};

const POSITIVE = [94, 234, 212];
const NEGATIVE = [249, 168, 212];
const ACTIVE = "#fef08a";
const STAGE_FAMILY = {
  attn_out: "attn",
  resid_after_attn: "resid",
  mlp_out: "mlp",
  resid_after_mlp: "resid",
};

async function fetchJson(url, options = {}) {
  let response;
  try {
    response = await fetch(url, options);
  } catch (error) {
    throw new Error(`Network request failed for ${url}. Check that the local server is still running.`);
  }

  const raw = await response.text();
  let payload = {};
  if (raw) {
    try {
      payload = JSON.parse(raw);
    } catch (_error) {
      if (!response.ok) {
        throw new Error(`Request failed with HTTP ${response.status}.`);
      }
      throw new Error(`Invalid JSON response from ${url}.`);
    }
  }

  if (!response.ok) {
    throw new Error(payload.error || `Request failed with HTTP ${response.status}.`);
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

function selectedFamilies() {
  return {
    attn: elements.toggleAttn.checked,
    resid: elements.toggleResid.checked,
    mlp: elements.toggleMlp.checked,
  };
}

function activeTexture() {
  if (!state.visibleTextures.length) {
    return null;
  }
  return state.visibleTextures[state.visibleIndex];
}

function familyForStep(step) {
  return STAGE_FAMILY[step.stage_id] || "resid";
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
  const eased = Math.pow(magnitude, 0.75);
  const floor = 8;
  return [
    Math.round(floor + (base[0] - floor) * eased),
    Math.round(floor + (base[1] - floor) * eased),
    Math.round(floor + (base[2] - floor) * eased),
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
  return { ...step, canvas, hotspots };
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

function updateVisibleSteps(preferredStepIndex = null) {
  const families = selectedFamilies();
  const active = activeTexture();
  const keepStepIndex = preferredStepIndex ?? (active ? active.step_index : null);

  state.visibleTextures = state.allTextures.filter((step) => families[familyForStep(step)]);

  if (!state.visibleTextures.length) {
    state.visibleIndex = 0;
    elements.stepSlider.disabled = true;
    elements.playButton.disabled = true;
    elements.stepLabel.textContent = "Step: Select at least one stage family";
    elements.stepCounter.textContent = "0 / 0";
    render();
    return;
  }

  const matchedIndex = state.visibleTextures.findIndex((step) => step.step_index === keepStepIndex);
  state.visibleIndex = matchedIndex >= 0 ? matchedIndex : 0;
  elements.stepSlider.disabled = false;
  elements.playButton.disabled = false;
  elements.stepSlider.max = String(Math.max(state.visibleTextures.length - 1, 0));
  elements.stepSlider.value = String(state.visibleIndex);
  updateStepLabel();
  render();
}

function updateStepLabel() {
  const step = activeTexture();
  if (!step) {
    elements.stepLabel.textContent = "Step: —";
    elements.stepCounter.textContent = "0 / 0";
    return;
  }
  elements.stepLabel.textContent = `Step: Layer ${step.layer_index + 1} • ${step.stage_label}`;
  elements.stepCounter.textContent = `${state.visibleIndex + 1} / ${state.visibleTextures.length}`;
}

function setVisibleStep(index) {
  if (!state.visibleTextures.length) {
    return;
  }

  state.visibleIndex = Math.max(0, Math.min(index, state.visibleTextures.length - 1));
  elements.stepSlider.value = String(state.visibleIndex);
  updateStepLabel();
  render();
}

function togglePlayback() {
  if (!state.visibleTextures.length) {
    return;
  }

  if (state.playing) {
    stopPlayback();
    return;
  }

  state.playing = true;
  elements.playButton.textContent = "Pause";
  state.timer = window.setInterval(() => {
    const nextIndex = (state.visibleIndex + 1) % state.visibleTextures.length;
    setVisibleStep(nextIndex);
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
  const steps = state.allTextures;
  const active = activeTexture();
  const families = selectedFamilies();
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
    const enabled = families[familyForStep(step)];

    context.globalAlpha = active && step.step_index === active.step_index ? 1 : enabled ? 0.32 : 0.08;
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

  if (active) {
    const activeX = panelX(active.step_index, overview);
    const activeWidth = overview.unitWidth;
    context.globalAlpha = 1;
    context.shadowColor = ACTIVE;
    context.shadowBlur = 20;
    context.strokeStyle = ACTIVE;
    context.lineWidth = 2;
    context.strokeRect(activeX - 2, overview.y + 20, activeWidth + 4, overview.height - 28);
  }
  context.restore();
}

function drawHotspots(context, detail, step) {
  if (!step.hotspots.length) {
    return;
  }

  context.save();
  const count = Math.min(140, step.hotspots.length);

  for (let index = 0; index < count; index += 1) {
    const hotspot = step.hotspots[(index * 29 + step.step_index * 17) % step.hotspots.length];
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
  const step = activeTexture();
  if (!step) {
    context.fillStyle = "rgba(228, 243, 255, 0.56)";
    context.font = "500 18px IBM Plex Sans";
    context.fillText("Enable at least one stage family to inspect a slice.", 28, detail.y + 18);
    return;
  }

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

  const tokenLabelCount = state.payload.tokens.length;
  const fontSize = Math.max(9, Math.min(13, 15 - tokenLabelCount * 0.18));
  context.font = `${fontSize}px IBM Plex Sans`;
  for (let index = 0; index < tokenLabelCount; index += 1) {
    const y = detail.y + ((index + 0.5) / tokenLabelCount) * detail.height;
    context.fillStyle = "rgba(212, 233, 255, 0.64)";
    context.fillText(state.payload.tokens[index], detail.x - 108, y + fontSize * 0.32);
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

  const tokenCount = state.payload.tokens.length;
  const detailHeight = Math.min(height * 0.58, Math.max(height * 0.4, tokenCount * 22));
  const detailTop = height - detailHeight - 54;
  const overview = {
    x: 24,
    y: 28,
    width: width - 48,
    height: Math.max(180, detailTop - 58),
    unitWidth: 0,
  };
  const detail = {
    x: 136,
    y: detailTop,
    width: width - 172,
    height: detailHeight,
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
        include_special_tokens: elements.toggleSpecial.checked,
      }),
    });

    state.payload = payload;
    state.allTextures = payload.steps.map(createTexture);
    renderTokenStrip();
    elements.analysisNote.textContent = payload.token_limit_applied
      ? `Showing the first ${payload.tokens.length} visible tokens${elements.toggleSpecial.checked ? " including special tokens" : ""}.`
      : `${payload.tokens.length} visible tokens across ${payload.steps.length} stage steps${elements.toggleSpecial.checked ? ", including special tokens" : ""}.`;
    updateVisibleSteps();
    setStatus("Ready", "ready");
  } catch (error) {
    setStatus("Error", "busy");
    elements.stepLabel.textContent = `Step: ${error.message}`;
    elements.stepCounter.textContent = "0 / 0";
    elements.analysisNote.textContent = "";
    state.payload = null;
    state.allTextures = [];
    state.visibleTextures = [];
    render();
  } finally {
    elements.analyzeButton.disabled = false;
  }
}

elements.analyzeButton.addEventListener("click", analyzePrompt);
elements.playButton.addEventListener("click", togglePlayback);
elements.stepSlider.addEventListener("input", (event) => {
  setVisibleStep(Number(event.target.value));
});
[elements.toggleAttn, elements.toggleResid, elements.toggleMlp].forEach((toggle) => {
  toggle.addEventListener("change", () => {
    stopPlayback();
    updateVisibleSteps();
  });
});
elements.toggleSpecial.addEventListener("change", () => {
  stopPlayback();
  if (state.payload) {
    analyzePrompt();
  }
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
