const state = {
  payload: null,
  layerAnalysis: [],
  models: [],
  allTextures: [],
  visibleTextures: [],
  visibleIndex: 0,
  playing: false,
  timer: null,
  lastPromptWasDefault: true,
  analysisRequestToken: 0,
};

const elements = {
  modelSelect: document.getElementById("model-select"),
  promptInput: document.getElementById("prompt-input"),
  analyzeButton: document.getElementById("analyze-button"),
  playButton: document.getElementById("play-button"),
  architectureButton: document.getElementById("architecture-button"),
  stepSlider: document.getElementById("step-slider"),
  stepLabel: document.getElementById("step-label"),
  stepCounter: document.getElementById("step-counter"),
  tokenStrip: document.getElementById("token-strip"),
  analysisNote: document.getElementById("analysis-note"),
  logitsMeta: document.getElementById("logits-meta"),
  logitsList: document.getElementById("logits-list"),
  metricsMeta: document.getElementById("metrics-meta"),
  metricsGrid: document.getElementById("metrics-grid"),
  attentionMeta: document.getElementById("attention-meta"),
  attentionGrid: document.getElementById("attention-grid"),
  depthMeta: document.getElementById("depth-meta"),
  depthList: document.getElementById("depth-list"),
  statusPill: document.getElementById("status-pill"),
  modelMeta: document.getElementById("model-meta"),
  canvas: document.getElementById("storm-canvas"),
  toggleEmb: document.getElementById("toggle-emb"),
  toggleAttn: document.getElementById("toggle-attn"),
  toggleResid: document.getElementById("toggle-resid"),
  toggleMlp: document.getElementById("toggle-mlp"),
  toggleSpecial: document.getElementById("toggle-special"),
  architectureDialog: document.getElementById("architecture-dialog"),
  architectureTitle: document.getElementById("architecture-title"),
  architectureContent: document.getElementById("architecture-content"),
  architectureClose: document.getElementById("architecture-close"),
};

const POSITIVE = [94, 234, 212];
const NEGATIVE = [249, 168, 212];
const ACTIVE = "#fef08a";
const OVERVIEW_HEIGHT = 208;
const TOKEN_ROW_HEIGHT = 24;
const DETAIL_TOP_GAP = 72;
const CANVAS_TOP_PADDING = 28;
const CANVAS_BOTTOM_PADDING = 54;
const DETAIL_META_SPACE = 34;
const DETAIL_LEFT_GUTTER = 152;
const PANEL_GAP = 2;
const GROUP_GAP = 7;
const EMBEDDING_GROUP_GAP = 12;
const EMBEDDING_LABEL_OFFSET = 18;

const STAGE_FAMILY = {
  embeddings: "emb",
  attn_out: "attn",
  resid_after_attn: "resid",
  mlp_out: "mlp",
  resid_after_mlp: "resid",
};

async function fetchJson(url, options = {}) {
  let response;
  try {
    response = await fetch(url, options);
  } catch (_error) {
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
    emb: elements.toggleEmb.checked,
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

function layerDisplayIndex(step) {
  return step.layer_index + 1;
}

function stepTitle(step) {
  return `Layer ${layerDisplayIndex(step)} • ${step.stage_label}`;
}

function base64ToBytes(base64) {
  const binary = window.atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

function visibleTokenMask() {
  if (!state.payload?.tokens?.length) {
    return [];
  }
  const mask = state.payload.visible_token_mask;
  if (Array.isArray(mask) && mask.length === state.payload.tokens.length) {
    return mask;
  }
  return state.payload.tokens.map(() => true);
}

function displayRowIndices() {
  if (!state.payload?.tokens?.length) {
    return [];
  }
  if (elements.toggleSpecial.checked) {
    return state.payload.tokens.map((_token, index) => index);
  }
  const mask = visibleTokenMask();
  return mask.flatMap((isVisible, index) => (isVisible ? [index] : []));
}

function displayTokens() {
  if (!state.payload?.tokens?.length) {
    return [];
  }
  const rowIndices = displayRowIndices();
  return rowIndices.map((index) => state.payload.tokens[index]);
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

function createTextureCanvas(step, bytes) {
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
  return { canvas, hotspots };
}

function createTexture(step) {
  const bytes = base64ToBytes(step.encoded_field);
  const { canvas, hotspots } = createTextureCanvas(step, bytes);
  return { ...step, canvas, hotspots, sourceBytes: bytes, fullRows: step.rows };
}

function filterTextureRows(step, rowIndices) {
  if (rowIndices.length === step.fullRows) {
    return step;
  }

  const filteredBytes = new Uint8Array(rowIndices.length * step.cols);
  rowIndices.forEach((rowIndex, filteredRowIndex) => {
    const sourceStart = rowIndex * step.cols;
    const targetStart = filteredRowIndex * step.cols;
    filteredBytes.set(step.sourceBytes.subarray(sourceStart, sourceStart + step.cols), targetStart);
  });

  const filteredStep = { ...step, rows: rowIndices.length };
  const { canvas, hotspots } = createTextureCanvas(filteredStep, filteredBytes);
  return { ...filteredStep, canvas, hotspots };
}

function desiredCanvasHeight() {
  if (!state.payload) {
    return 760;
  }

  const tokenCount = Math.max(displayTokens().length, 1);
  return CANVAS_TOP_PADDING + OVERVIEW_HEIGHT + DETAIL_TOP_GAP + (tokenCount * TOKEN_ROW_HEIGHT) + DETAIL_META_SPACE + CANVAS_BOTTOM_PADDING;
}

function resizeCanvas() {
  elements.canvas.style.height = `${desiredCanvasHeight()}px`;

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

  displayTokens().forEach((token) => {
    const chip = document.createElement("span");
    chip.className = "token-chip";
    chip.textContent = token === " " ? "␠" : token;
    elements.tokenStrip.appendChild(chip);
  });
}

function layerAnalysisMap() {
  if (!state.layerAnalysis?.length) {
    return new Map();
  }
  return new Map(state.layerAnalysis.map((entry) => [entry.layer_index, entry]));
}

function renderLogitList(container, topTokens, emptyText) {
  container.innerHTML = "";
  if (!topTokens?.length) {
    const empty = document.createElement("div");
    empty.className = "logit-empty";
    empty.textContent = emptyText;
    container.appendChild(empty);
    return;
  }

  topTokens.forEach((entry) => {
    const row = document.createElement("div");
    row.className = "logit-row";

    const token = document.createElement("span");
    token.className = "logit-token";
    token.textContent = entry.token === " " ? "␠" : entry.token;

    const value = document.createElement("span");
    value.className = "logit-value";
    value.textContent = entry.logit.toFixed(2);

    row.appendChild(token);
    row.appendChild(value);
    container.appendChild(row);
  });
}

function formatMetricValue(value, decimals = 2, suffix = "") {
  if (value == null || Number.isNaN(value)) {
    return "—";
  }
  return `${Number(value).toFixed(decimals)}${suffix}`;
}

function renderMetricGrid(container, metrics, specs, emptyText) {
  container.innerHTML = "";
  if (!metrics) {
    const empty = document.createElement("div");
    empty.className = "logit-empty";
    empty.textContent = emptyText;
    container.appendChild(empty);
    return;
  }

  specs.forEach((spec) => {
    const card = document.createElement("div");
    card.className = "metric-card";

    const labelRow = document.createElement("div");
    labelRow.className = "metric-label-row";

    const label = document.createElement("span");
    label.className = "metric-label";
    label.textContent = spec.label;

    const info = document.createElement("span");
    info.className = "metric-info";
    info.textContent = "i";
    info.tabIndex = 0;
    info.setAttribute("role", "img");
    info.setAttribute("aria-label", `${spec.label}: ${spec.description}`);
    info.dataset.tooltip = spec.description;

    const value = document.createElement("span");
    value.className = "metric-value";
    value.textContent = formatMetricValue(metrics[spec.key], spec.decimals, spec.suffix || "");

    labelRow.appendChild(label);
    labelRow.appendChild(info);
    card.appendChild(labelRow);
    card.appendChild(value);
    container.appendChild(card);
  });
}

function renderDepthList(container, entries, activeLayerIndex) {
  container.innerHTML = "";
  if (!entries?.length) {
    const empty = document.createElement("div");
    empty.className = "logit-empty";
    empty.textContent = "Run a prompt to inspect layer contribution shifts.";
    container.appendChild(empty);
    return;
  }

  const maxValue = Math.max(...entries.map((entry) => entry.contribution_metrics.logit_shift_rms), 1e-6);
  entries.forEach((entry) => {
    const row = document.createElement("div");
    row.className = `depth-row${entry.layer_index === activeLayerIndex ? " active" : ""}`;

    const value = document.createElement("span");
    value.className = "depth-value";
    value.textContent = formatMetricValue(entry.contribution_metrics.logit_shift_rms);

    const barWrap = document.createElement("div");
    barWrap.className = "depth-bar-wrap";

    const bar = document.createElement("div");
    bar.className = "depth-bar";
    bar.style.height = `${(entry.contribution_metrics.logit_shift_rms / maxValue) * 100}%`;
    barWrap.appendChild(bar);

    const label = document.createElement("span");
    label.className = "depth-label";
    label.textContent = `L${String(entry.layer_index + 1).padStart(2, "0")}`;

    row.appendChild(value);
    row.appendChild(barWrap);
    row.appendChild(label);
    container.appendChild(row);
  });
}

function renderAnalysisPanels() {
  if (!state.payload) {
    elements.logitsMeta.textContent = "Awaiting analysis";
    renderLogitList(elements.logitsList, [], "Run a prompt to inspect layer LogitLens results.");
    elements.metricsMeta.textContent = "Current layer";
    elements.attentionMeta.textContent = "Current layer";
    elements.depthMeta.textContent = "Logit shift by layer";
    renderMetricGrid(elements.metricsGrid, null, [], "Run a prompt to inspect activation metrics.");
    renderMetricGrid(elements.attentionGrid, null, [], "Run a prompt to inspect attention metrics.");
    renderDepthList(elements.depthList, [], -1);
    return;
  }

  const active = activeTexture();
  if (!active) {
    elements.logitsMeta.textContent = "Select a step";
    renderLogitList(elements.logitsList, [], "No LogitLens data available.");
    renderMetricGrid(elements.metricsGrid, null, [], "No activation metrics available.");
    renderMetricGrid(elements.attentionGrid, null, [], "No attention metrics available.");
    renderDepthList(elements.depthList, state.layerAnalysis || [], -1);
    return;
  }

  const layerIndex = Math.max(active.layer_index, 0);
  const layerEntry = layerAnalysisMap().get(layerIndex) || null;
  const topTokens = layerEntry?.top_tokens || [];
  const targetToken = state.payload.target_token === " " ? "␠" : state.payload.target_token;
  const isFinalLayer = state.payload.model && layerIndex === (state.payload.model.layer_count - 1);
  elements.logitsMeta.textContent = isFinalLayer
    ? `Final logits after ${targetToken}`
    : `Layer ${layerIndex + 1} logits after ${targetToken}`;
  renderLogitList(
    elements.logitsList,
    topTokens,
    state.layerAnalysis.length ? "No LogitLens data for the current layer." : "Computing LogitLens and metrics…",
  );
  elements.metricsMeta.textContent = `Layer ${layerIndex + 1}`;
  elements.attentionMeta.textContent = `Layer ${layerIndex + 1}`;
  elements.depthMeta.textContent = `Current: Layer ${layerIndex + 1}`;
  renderMetricGrid(
    elements.metricsGrid,
    layerEntry?.activation_metrics,
    [
      { key: "layer_variance", label: "Layer Variance", decimals: 2, description: "How much the layer residual values vary overall, which is a simple depth-growth proxy." },
      { key: "kurtosis", label: "Kurtosis", decimals: 2, description: "How uneven the channel magnitudes are, with higher values meaning stronger outlier channels." },
      { key: "top_energy_share", label: "Top 1% Energy", decimals: 1, description: "How much of the layer energy sits in the strongest 1% of channels." },
      { key: "participation_ratio", label: "Participation", decimals: 2, description: "Roughly how many dimensions are carrying meaningful signal here." },
    ],
    state.layerAnalysis.length ? "No activation metrics for the current layer." : "Computing activation metrics…",
  );
  renderMetricGrid(
    elements.attentionGrid,
    layerEntry?.attention_metrics,
    [
      { key: "mean_entropy", label: "Mean Entropy", decimals: 2, description: "How spread out attention is across source tokens, averaged over heads and query positions." },
      { key: "sink_mass", label: "Sink Mass", decimals: 2, description: "How much attention flows into the first token on average, a common sink location." },
      { key: "sink_head_ratio", label: "Sink Heads", decimals: 1, description: "Percent of attention rows whose top target is the first token." },
    ],
    state.layerAnalysis.length ? "No attention metrics for the current layer." : "Computing attention metrics…",
  );
  if (layerEntry?.activation_metrics) {
    const energyCard = elements.metricsGrid.querySelectorAll(".metric-value")[2];
    if (energyCard) {
      energyCard.textContent = formatMetricValue(layerEntry.activation_metrics.top_energy_share * 100, 1, "%");
    }
  }
  if (layerEntry?.attention_metrics) {
    const attentionValues = elements.attentionGrid.querySelectorAll(".metric-value");
    if (attentionValues[2]) {
      attentionValues[2].textContent = formatMetricValue(layerEntry.attention_metrics.sink_head_ratio * 100, 1, "%");
    }
  }
  renderDepthList(elements.depthList, state.layerAnalysis || [], layerIndex);
}

async function loadLayerAnalysis(requestToken) {
  try {
    const payload = await fetchJson("/api/layer-analysis", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model_id: elements.modelSelect.value,
        prompt: elements.promptInput.value,
        include_special_tokens: true,
      }),
    });

    if (requestToken !== state.analysisRequestToken || !state.payload) {
      return;
    }
    state.layerAnalysis = payload.layer_analysis || [];
    renderAnalysisPanels();
  } catch (_error) {
    if (requestToken !== state.analysisRequestToken) {
      return;
    }
    state.layerAnalysis = [];
    renderAnalysisPanels();
  }
}

function updateVisibleSteps(preferredStepIndex = null) {
  const families = selectedFamilies();
  const active = activeTexture();
  const keepStepIndex = preferredStepIndex ?? (active ? active.step_index : null);
  const rowIndices = displayRowIndices();

  state.visibleTextures = state.allTextures
    .filter((step) => families[familyForStep(step)])
    .map((step) => filterTextureRows(step, rowIndices));

  if (!state.visibleTextures.length) {
    state.visibleIndex = 0;
    elements.stepSlider.disabled = true;
    elements.playButton.disabled = true;
    elements.stepLabel.textContent = "Step: Select at least one stage family";
    elements.stepCounter.textContent = "0 / 0";
    renderAnalysisPanels();
    resizeCanvas();
    return;
  }

  const matchedIndex = state.visibleTextures.findIndex((step) => step.step_index === keepStepIndex);
  state.visibleIndex = matchedIndex >= 0 ? matchedIndex : 0;
  elements.stepSlider.disabled = false;
  elements.playButton.disabled = false;
  elements.stepSlider.max = String(Math.max(state.visibleTextures.length - 1, 0));
  elements.stepSlider.value = String(state.visibleIndex);
  updateStepLabel();
  renderAnalysisPanels();
  resizeCanvas();
}

function updateStepLabel() {
  const step = activeTexture();
  if (!step) {
    elements.stepLabel.textContent = "Step: —";
    elements.stepCounter.textContent = "0 / 0";
    return;
  }
  elements.stepLabel.textContent = `Step: ${stepTitle(step)}`;
  elements.stepCounter.textContent = `${state.visibleIndex + 1} / ${state.visibleTextures.length}`;
}

function describeModel(model) {
  if (!model) {
    return "Model metadata unavailable.";
  }
  if (model.layer_count > 0 && model.layer_width > 0) {
    return `${model.label} • ${model.layer_count} layers • width ${model.layer_width}`;
  }
  return `${model.label} • metadata unavailable until model config is accessible`;
}

function defaultPromptForModel(model) {
  return model?.default_prompt || "The capital of France is";
}

function updateSelectedModelUi({ resetPrompt = false } = {}) {
  const active = state.models.find((model) => model.id === elements.modelSelect.value) || null;
  elements.modelMeta.textContent = describeModel(active);

  if (!active) {
    return;
  }

  if (resetPrompt || state.lastPromptWasDefault) {
    elements.promptInput.value = defaultPromptForModel(active);
    state.lastPromptWasDefault = true;
  }
}

function setVisibleStep(index) {
  if (!state.visibleTextures.length) {
    return;
  }

  state.visibleIndex = Math.max(0, Math.min(index, state.visibleTextures.length - 1));
  elements.stepSlider.value = String(state.visibleIndex);
  updateStepLabel();
  renderAnalysisPanels();
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


async function showArchitecture() {
  const modelId = elements.modelSelect.value;
  if (!modelId) {
    return;
  }

  elements.architectureButton.disabled = true;
  elements.architectureTitle.textContent = "Loading architecture…";
  elements.architectureContent.textContent = "Loading…";
  if (typeof elements.architectureDialog.showModal === "function") {
    if (!elements.architectureDialog.open) {
      elements.architectureDialog.showModal();
    }
  } else {
    elements.architectureDialog.setAttribute("open", "open");
  }

  try {
    const payload = await fetchJson(`/api/architecture?model_id=${encodeURIComponent(modelId)}`);
    elements.architectureTitle.textContent = payload.model.label;
    elements.architectureContent.textContent = payload.architecture;
  } catch (error) {
    elements.architectureTitle.textContent = "Architecture";
    elements.architectureContent.textContent = error.message;
  } finally {
    elements.architectureButton.disabled = false;
  }
}

function closeArchitecture() {
  if (typeof elements.architectureDialog.close === "function") {
    elements.architectureDialog.close();
  } else {
    elements.architectureDialog.removeAttribute("open");
  }
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

function groupKey(step) {
  return step.stage_id === "embeddings" ? "embeddings" : `layer-${step.layer_index}`;
}

function groupLabel(step) {
  return step.stage_id === "embeddings" ? "EMB" : `L${String(layerDisplayIndex(step)).padStart(2, "0")}`;
}

function buildOverviewLayout(steps, overview) {
  const groupKeys = steps.map(groupKey);
  const groupStarts = [];
  let groupBreakCount = 0;

  steps.forEach((step, index) => {
    if (index === 0 || groupKeys[index] !== groupKeys[index - 1]) {
      groupStarts.push(index);
      if (index > 0) {
        groupBreakCount += 1;
      }
    }
  });

  const embeddingGapBonus = steps.some((step) => step.stage_id === "embeddings") ? (EMBEDDING_GROUP_GAP - GROUP_GAP) : 0;
  const totalGap = (Math.max(steps.length - 1, 0) * PANEL_GAP) + (groupBreakCount * GROUP_GAP) + embeddingGapBonus + EMBEDDING_LABEL_OFFSET;
  const unitWidth = Math.max((overview.width - totalGap) / Math.max(steps.length, 1), 3.5);
  const positions = [];
  let cursor = overview.x + EMBEDDING_LABEL_OFFSET;

  steps.forEach((step, index) => {
    if (index > 0) {
      cursor += PANEL_GAP;
      if (groupKeys[index] !== groupKeys[index - 1]) {
        cursor += groupKeys[index - 1] === "embeddings" ? EMBEDDING_GROUP_GAP : GROUP_GAP;
      }
    }
    positions.push(cursor);
    cursor += unitWidth;
  });

  return { unitWidth, positions, groupStarts };
}

function drawOverview(context, overview) {
  const steps = state.allTextures;
  const active = activeTexture();
  const families = selectedFamilies();
  const layout = buildOverviewLayout(steps, overview);

  context.save();
  context.strokeStyle = "rgba(155, 212, 255, 0.08)";
  context.strokeRect(overview.x, overview.y, overview.width, overview.height);

  steps.forEach((step, index) => {
    const x = layout.positions[index];
    const y = overview.y + 24;
    const width = layout.unitWidth;
    const height = overview.height - 36;
    const enabled = families[familyForStep(step)];

    context.globalAlpha = active && step.step_index === active.step_index ? 1 : enabled ? 0.32 : 0.08;
    context.imageSmoothingEnabled = false;
    context.drawImage(step.canvas, x, y, width, height);

    if (index < steps.length - 1 && groupKey(steps[index + 1]) !== groupKey(step)) {
      context.globalAlpha = 1;
      context.fillStyle = "rgba(149, 192, 255, 0.08)";
      context.fillRect(x + width + 3, y - 10, 1, height + 20);
    }

    if (index === 0 || groupKey(steps[index - 1]) !== groupKey(step)) {
      context.globalAlpha = 1;
      context.fillStyle = "rgba(228, 243, 255, 0.52)";
      context.font = "11px IBM Plex Sans";
      const labelX = step.stage_id === "embeddings" ? x - EMBEDDING_LABEL_OFFSET + 2 : x;
      context.fillText(groupLabel(step), labelX, overview.y + 14);
    }
  });

  if (active) {
    const activeIndex = steps.findIndex((step) => step.step_index === active.step_index);
    const activeX = layout.positions[activeIndex];
    const activeWidth = layout.unitWidth;
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
  context.fillText(stepTitle(step), detail.x, detail.y - 18);
  context.font = "12px IBM Plex Sans";
  context.fillStyle = "rgba(169, 207, 241, 0.72)";
  context.fillText(`${step.rows} tokens × ${step.cols} hidden dims`, detail.x, detail.y + detail.height + 18);

  const tokens = displayTokens();
  const tokenLabelCount = tokens.length;
  const fontSize = 12;
  context.font = `${fontSize}px IBM Plex Sans`;
  for (let index = 0; index < tokenLabelCount; index += 1) {
    const y = detail.y + ((index + 0.5) * TOKEN_ROW_HEIGHT);
    context.fillStyle = "rgba(212, 233, 255, 0.72)";
    context.fillText(tokens[index], detail.x - 116, y + 4);
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

  const tokenCount = displayTokens().length;
  const detailHeight = Math.max(tokenCount, 1) * TOKEN_ROW_HEIGHT;
  const overview = {
    x: 24,
    y: CANVAS_TOP_PADDING,
    width: width - 48,
    height: OVERVIEW_HEIGHT,
  };
  const detail = {
    x: DETAIL_LEFT_GUTTER,
    y: overview.y + overview.height + DETAIL_TOP_GAP,
    width: width - DETAIL_LEFT_GUTTER - 28,
    height: detailHeight,
  };

  drawOverview(context, overview);
  drawDetail(context, detail);
}

async function loadModels() {
  const payload = await fetchJson("/api/models");
  state.models = payload.models;
  elements.modelSelect.innerHTML = "";
  payload.models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.id;
    option.textContent = model.label;
    elements.modelSelect.appendChild(option);
  });
  elements.modelSelect.value = payload.default_model;
  updateSelectedModelUi({ resetPrompt: true });
}

async function analyzePrompt() {
  stopPlayback();
  setStatus("Analyzing", "busy");
  elements.analyzeButton.disabled = true;
  const requestToken = state.analysisRequestToken + 1;
  state.analysisRequestToken = requestToken;

  try {
    const payload = await fetchJson("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model_id: elements.modelSelect.value,
        prompt: elements.promptInput.value,
        include_special_tokens: true,
      }),
    });

    state.payload = payload;
    state.layerAnalysis = [];
    state.allTextures = payload.steps.map(createTexture);
    renderTokenStrip();
    renderAnalysisPanels();
    const visibleCount = visibleTokenMask().filter(Boolean).length;
    elements.analysisNote.textContent = elements.toggleSpecial.checked
      ? `${payload.tokens.length} total tokens across ${payload.steps.length} stage steps, including special tokens.`
      : `${visibleCount} visible prompt tokens shown out of ${payload.tokens.length} total tokens across ${payload.steps.length} stage steps.`;
    resizeCanvas();
    updateVisibleSteps();
    setStatus("Ready", "ready");
    loadLayerAnalysis(requestToken);
  } catch (error) {
    setStatus("Error", "busy");
    elements.stepLabel.textContent = `Step: ${error.message}`;
    elements.stepCounter.textContent = "0 / 0";
    elements.analysisNote.textContent = "";
    state.payload = null;
    state.layerAnalysis = [];
    state.allTextures = [];
    state.visibleTextures = [];
    renderAnalysisPanels();
    render();
  } finally {
    elements.analyzeButton.disabled = false;
  }
}

elements.analyzeButton.addEventListener("click", analyzePrompt);
elements.playButton.addEventListener("click", togglePlayback);
elements.architectureButton.addEventListener("click", showArchitecture);
elements.architectureClose.addEventListener("click", closeArchitecture);
elements.architectureDialog.addEventListener("click", (event) => {
  if (event.target === elements.architectureDialog) {
    closeArchitecture();
  }
});
elements.stepSlider.addEventListener("input", (event) => {
  setVisibleStep(Number(event.target.value));
});
[elements.toggleEmb, elements.toggleAttn, elements.toggleResid, elements.toggleMlp].forEach((toggle) => {
  toggle.addEventListener("change", () => {
    stopPlayback();
    updateVisibleSteps();
  });
});
elements.toggleSpecial.addEventListener("change", () => {
  stopPlayback();
  if (state.payload) {
    renderTokenStrip();
    const active = activeTexture();
    updateVisibleSteps(active ? active.step_index : null);
    const visibleCount = visibleTokenMask().filter(Boolean).length;
    elements.analysisNote.textContent = elements.toggleSpecial.checked
      ? `${state.payload.tokens.length} total tokens across ${state.payload.steps.length} stage steps, including special tokens.`
      : `${visibleCount} visible prompt tokens shown out of ${state.payload.tokens.length} total tokens across ${state.payload.steps.length} stage steps.`;
  }
});
window.addEventListener("resize", resizeCanvas);
elements.promptInput.addEventListener("input", () => {
  const active = state.models.find((model) => model.id === elements.modelSelect.value) || null;
  state.lastPromptWasDefault = elements.promptInput.value === defaultPromptForModel(active);
});
elements.modelSelect.addEventListener("change", () => {
  updateSelectedModelUi({ resetPrompt: true });
});

loadModels()
  .then(() => {
    setStatus("Idle", "idle");
    renderAnalysisPanels();
    resizeCanvas();
  })
  .catch((error) => {
    setStatus("Error", "busy");
    elements.modelMeta.textContent = error.message;
    renderAnalysisPanels();
    resizeCanvas();
  });
