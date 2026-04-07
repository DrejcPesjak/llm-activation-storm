const state = {
  runs: [],
  metricCatalog: [],
  selections: new Map(),
  result: null,
};

const elements = {
  statusPill: document.getElementById("status-pill"),
  runCount: document.getElementById("run-count"),
  runList: document.getElementById("run-list"),
  selectAllButton: document.getElementById("select-all-button"),
  clearSelectionButton: document.getElementById("clear-selection-button"),
  modeSelect: document.getElementById("mode-select"),
  analyzeButton: document.getElementById("analyze-button"),
  selectionSummary: document.getElementById("selection-summary"),
  selectionError: document.getElementById("selection-error"),
  resultsTitle: document.getElementById("results-title"),
  resultsMeta: document.getElementById("results-meta"),
  groupSummary: document.getElementById("group-summary"),
  metricSections: document.getElementById("metric-sections"),
  deltaSection: document.getElementById("delta-section"),
};

const GROUP_COLORS = {
  a: "#8ad3ee",
  b: "#f4b6d9",
  delta: "#fef08a",
};

async function fetchJson(url, options = {}) {
  let response;
  try {
    response = await fetch(url, options);
  } catch (_error) {
    throw new Error(`Network request failed for ${url}.`);
  }

  const raw = await response.text();
  const payload = raw ? JSON.parse(raw) : {};
  if (!response.ok) {
    throw new Error(payload.error || `Request failed with HTTP ${response.status}.`);
  }
  return payload;
}

function setStatus(label, className) {
  elements.statusPill.textContent = label;
  elements.statusPill.className = `pill ${className}`;
}

function runLookup(runId) {
  return state.runs.find((run) => run.run_id === runId) || null;
}

function ensureSelection(runId) {
  if (!state.selections.has(runId)) {
    state.selections.set(runId, { checked: false, group: "a" });
  }
  return state.selections.get(runId);
}

function selectedRunIds() {
  return state.runs.flatMap((run) => (ensureSelection(run.run_id).checked ? [run.run_id] : []));
}

function selectedGroupRunIds(group) {
  return state.runs.flatMap((run) => {
    const selection = ensureSelection(run.run_id);
    return selection.checked && selection.group === group ? [run.run_id] : [];
  });
}

function summarizeSelection() {
  const selected = selectedRunIds();
  const groupA = selectedGroupRunIds("a");
  const groupB = selectedGroupRunIds("b");
  const mode = elements.modeSelect.value;
  if (mode === "compare_groups") {
    return `Selected ${selected.length} runs • Group A ${groupA.length} • Group B ${groupB.length}`;
  }
  return `Selected ${selected.length} runs`;
}

function validationMessage() {
  const mode = elements.modeSelect.value;
  const selected = selectedRunIds();
  const selectedRuns = selected.map((runId) => runLookup(runId)).filter(Boolean);
  const selectedModelIds = new Set(selectedRuns.map((run) => run.model_id));

  if (mode === "aggregate_selected") {
    if (!selected.length) {
      return "Select at least one run to aggregate.";
    }
    if (selectedModelIds.size > 1) {
      return "Aggregate selected requires all runs to come from one model.";
    }
    return "";
  }

  if (mode === "compare_two_runs") {
    if (selected.length !== 2) {
      return "Compare two runs requires exactly two selected runs.";
    }
    if (selectedModelIds.size > 1) {
      return "Compare two runs requires both runs to come from one model.";
    }
    return "";
  }

  const groupA = selectedGroupRunIds("a").map((runId) => runLookup(runId)).filter(Boolean);
  const groupB = selectedGroupRunIds("b").map((runId) => runLookup(runId)).filter(Boolean);
  if (!groupA.length || !groupB.length) {
    return "Compare Group A vs Group B requires at least one run in each group.";
  }
  if (new Set(groupA.map((run) => run.model_id)).size > 1) {
    return "Group A must contain runs from exactly one model.";
  }
  if (new Set(groupB.map((run) => run.model_id)).size > 1) {
    return "Group B must contain runs from exactly one model.";
  }
  return "";
}

function updateSelectionMeta() {
  elements.selectionSummary.textContent = summarizeSelection();
  const message = validationMessage();
  elements.selectionError.textContent = message;
}

function renderRunList() {
  elements.runList.innerHTML = "";
  if (!state.runs.length) {
    const empty = document.createElement("p");
    empty.className = "cross-panel-note";
    empty.textContent = "No metrics logs found in the configured log directory.";
    elements.runList.appendChild(empty);
    return;
  }

  const compareGroups = elements.modeSelect.value === "compare_groups";
  state.runs.forEach((run) => {
    const selection = ensureSelection(run.run_id);
    const card = document.createElement("label");
    card.className = "run-card";
    if (selection.checked) {
      card.classList.add("selected");
    }

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = selection.checked;
    checkbox.addEventListener("change", () => {
      selection.checked = checkbox.checked;
      if (!selection.checked) {
        selection.group = "a";
      }
      renderRunList();
      updateSelectionMeta();
    });

    const body = document.createElement("div");
    body.className = "run-card-body";

    const top = document.createElement("div");
    top.className = "run-card-top";

    const model = document.createElement("strong");
    model.textContent = run.model_label;

    const timestamp = document.createElement("span");
    timestamp.className = "run-card-time";
    timestamp.textContent = run.timestamp.replace("T", " ");

    top.appendChild(model);
    top.appendChild(timestamp);

    const prompt = document.createElement("div");
    prompt.className = "run-card-prompt";
    prompt.textContent = run.prompt_preview;

    body.appendChild(top);
    body.appendChild(prompt);

    if (compareGroups && selection.checked) {
      const groupRow = document.createElement("div");
      groupRow.className = "group-toggle-row";

      ["a", "b"].forEach((groupId) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = `group-chip ${selection.group === groupId ? "active" : ""}`;
        button.textContent = groupId === "a" ? "Group A" : "Group B";
        button.addEventListener("click", (event) => {
          event.preventDefault();
          selection.group = groupId;
          renderRunList();
          updateSelectionMeta();
        });
        groupRow.appendChild(button);
      });

      body.appendChild(groupRow);
    }

    card.appendChild(checkbox);
    card.appendChild(body);
    elements.runList.appendChild(card);
  });
}

function clearResults() {
  state.result = null;
  elements.resultsTitle.textContent = "Awaiting analysis";
  elements.resultsMeta.textContent = "Select logged runs and choose an action.";
  elements.groupSummary.innerHTML = "";
  elements.metricSections.innerHTML = "";
  elements.deltaSection.innerHTML = "";
}

const CHART_LEFT = 38;
const CHART_RIGHT = 302;
const CHART_TOP = 18;
const CHART_BOTTOM = 125;
const X_AXIS_LABEL_Y = 140;
const X_AXIS_PADDING_RATIO = 0.04;
const Y_AXIS_PADDING_RATIO = 0.10;

function paddedBounds(minValue, maxValue, paddingRatio) {
  const range = maxValue - minValue;
  if (Math.abs(range) < 1e-9) {
    const base = Math.abs(maxValue) > 1e-9 ? Math.abs(maxValue) : 1;
    const padding = base * paddingRatio;
    return { minValue: minValue - padding, maxValue: maxValue + padding };
  }
  const padding = range * paddingRatio;
  return { minValue: minValue - padding, maxValue: maxValue + padding };
}

function xPosition(relativeDepth) {
  const paddedDepth = X_AXIS_PADDING_RATIO + (relativeDepth * (1 - (2 * X_AXIS_PADDING_RATIO)));
  return CHART_LEFT + (paddedDepth * (CHART_RIGHT - CHART_LEFT));
}

function yPosition(value, minValue, maxValue) {
  const range = maxValue - minValue || 1;
  return CHART_TOP + (1 - ((value - minValue) / range)) * (CHART_BOTTOM - CHART_TOP);
}

function linePath(points, valueKey, minValue, maxValue) {
  return points.map((point, index) => {
    const x = xPosition(point.relative_depth);
    const y = yPosition(point[valueKey], minValue, maxValue);
    return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
  }).join(" ");
}

function bandPath(points, minValue, maxValue) {
  const upper = points.map((point) => {
    const x = xPosition(point.relative_depth);
    const y = yPosition(point.mean + point.std, minValue, maxValue);
    return `${x.toFixed(2)} ${y.toFixed(2)}`;
  });
  const lower = [...points].reverse().map((point) => {
    const x = xPosition(point.relative_depth);
    const y = yPosition(point.mean - point.std, minValue, maxValue);
    return `${x.toFixed(2)} ${y.toFixed(2)}`;
  });
  return `M ${upper.join(" L ")} L ${lower.join(" L ")} Z`;
}

function chartValueBounds(seriesA, seriesB = null) {
  const values = [];
  [seriesA, seriesB].filter(Boolean).forEach((series) => {
    series.forEach((point) => {
      values.push(point.mean + point.std, point.mean - point.std);
    });
  });
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  if (!Number.isFinite(minValue) || !Number.isFinite(maxValue)) {
    return { minValue: 0, maxValue: 1 };
  }
  return { minValue, maxValue };
}

function formatNumber(value) {
  return Number(value).toFixed(Math.abs(value) >= 100 ? 1 : 3);
}

function tickValues(minValue, maxValue, count = 4) {
  if (!Number.isFinite(minValue) || !Number.isFinite(maxValue)) {
    return [0, 0.333, 0.667, 1];
  }
  if (Math.abs(maxValue - minValue) < 1e-9) {
    return [minValue];
  }
  return Array.from({ length: count }, (_value, index) => minValue + ((maxValue - minValue) * index) / (count - 1));
}

function createMetricCard(metric, sectionResult) {
  const card = document.createElement("article");
  card.className = "cross-metric-card";

  const header = document.createElement("div");
  header.className = "cross-metric-card-header";
  header.innerHTML = `<strong>${metric.label}</strong>`;

  const chart = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  chart.setAttribute("viewBox", "0 0 320 156");
  chart.setAttribute("class", "cross-chart");

  const seriesA = sectionResult.groupA.metric_trends[metric.key];
  const seriesB = sectionResult.groupB ? sectionResult.groupB.metric_trends[metric.key] : null;
  const tickBounds = chartValueBounds(seriesA, seriesB);
  const plotBounds = paddedBounds(tickBounds.minValue, tickBounds.maxValue, Y_AXIS_PADDING_RATIO);

  tickValues(tickBounds.minValue, tickBounds.maxValue).forEach((tickValue) => {
    const y = yPosition(tickValue, plotBounds.minValue, plotBounds.maxValue);
    const gridLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
    gridLine.setAttribute("x1", String(CHART_LEFT));
    gridLine.setAttribute("x2", String(CHART_RIGHT));
    gridLine.setAttribute("y1", y.toFixed(2));
    gridLine.setAttribute("y2", y.toFixed(2));
    gridLine.setAttribute("class", "cross-grid-line");
    chart.appendChild(gridLine);

    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", "32");
    label.setAttribute("y", (y + 4).toFixed(2));
    label.setAttribute("text-anchor", "end");
    label.setAttribute("class", "cross-tick-label");
    label.textContent = formatNumber(tickValue);
    chart.appendChild(label);
  });

  [0, 0.5, 1].forEach((tickValue) => {
    const x = xPosition(tickValue);
    const tick = document.createElementNS("http://www.w3.org/2000/svg", "line");
    tick.setAttribute("x1", x.toFixed(2));
    tick.setAttribute("x2", x.toFixed(2));
    tick.setAttribute("y1", yPosition(plotBounds.maxValue, plotBounds.minValue, plotBounds.maxValue).toFixed(2));
    tick.setAttribute("y2", yPosition(plotBounds.minValue, plotBounds.minValue, plotBounds.maxValue).toFixed(2));
    tick.setAttribute("class", "cross-grid-line vertical");
    chart.appendChild(tick);

    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", x.toFixed(2));
    label.setAttribute("y", String(X_AXIS_LABEL_Y));
    label.setAttribute("text-anchor", "middle");
    label.setAttribute("class", "cross-tick-label");
    label.textContent = `${Math.round(tickValue * 100)}%`;
    chart.appendChild(label);
  });

  const bandA = document.createElementNS("http://www.w3.org/2000/svg", "path");
  bandA.setAttribute("d", bandPath(seriesA, plotBounds.minValue, plotBounds.maxValue));
  bandA.setAttribute("fill", "rgba(138, 211, 238, 0.18)");
  chart.appendChild(bandA);

  const lineA = document.createElementNS("http://www.w3.org/2000/svg", "path");
  lineA.setAttribute("d", linePath(seriesA, "mean", plotBounds.minValue, plotBounds.maxValue));
  lineA.setAttribute("stroke", GROUP_COLORS.a);
  lineA.setAttribute("class", "cross-line");
  chart.appendChild(lineA);

  if (seriesB) {
    const bandB = document.createElementNS("http://www.w3.org/2000/svg", "path");
    bandB.setAttribute("d", bandPath(seriesB, plotBounds.minValue, plotBounds.maxValue));
    bandB.setAttribute("fill", "rgba(244, 182, 217, 0.18)");
    chart.appendChild(bandB);

    const lineB = document.createElementNS("http://www.w3.org/2000/svg", "path");
    lineB.setAttribute("d", linePath(seriesB, "mean", plotBounds.minValue, plotBounds.maxValue));
    lineB.setAttribute("stroke", GROUP_COLORS.b);
    lineB.setAttribute("class", "cross-line");
    chart.appendChild(lineB);
  }

  const footer = document.createElement("div");
  footer.className = "cross-metric-footer";

  const summaryA = sectionResult.groupA.metric_summaries[metric.key];
  const rows = [
    `A final ${formatNumber(summaryA.final_value)} • A peak ${formatNumber(summaryA.peak_value)} @ ${(summaryA.peak_depth * 100).toFixed(0)}%`,
  ];

  if (sectionResult.groupB) {
    const summaryB = sectionResult.groupB.metric_summaries[metric.key];
    const deltaSummary = sectionResult.delta.metric_summaries[metric.key];
    rows.push(`B final ${formatNumber(summaryB.final_value)} • B peak ${formatNumber(summaryB.peak_value)} @ ${(summaryB.peak_depth * 100).toFixed(0)}%`);
    rows.push(`Δ final ${formatNumber(deltaSummary.final_value)} • Δ peak ${formatNumber(deltaSummary.peak_value)} @ ${(deltaSummary.peak_depth * 100).toFixed(0)}%`);
    rows.push(`Δ mean ${formatNumber(deltaSummary.series_mean)} ± ${formatNumber(deltaSummary.series_std)}`);
  }

  footer.innerHTML = rows.map((row) => `<span>${row}</span>`).join("");

  if (sectionResult.groupB) {
    const legend = document.createElement("div");
    legend.className = "cross-inline-legend";
    legend.innerHTML = `
      <span><i class="legend-dot cross-legend-a"></i> Group A</span>
      <span><i class="legend-dot cross-legend-b"></i> Group B</span>
    `;
    card.appendChild(header);
    card.appendChild(legend);
    card.appendChild(chart);
    card.appendChild(footer);
    return card;
  }

  card.appendChild(header);
  card.appendChild(chart);
  card.appendChild(footer);
  return card;
}

function renderGroupSummary(result) {
  elements.groupSummary.innerHTML = "";
  const cards = [];
  const groupA = result.group_a;
  cards.push({
    title: "Group A",
    body: `${groupA.model.label} • ${groupA.run_count} run${groupA.run_count === 1 ? "" : "s"}`,
  });
  if (result.group_b) {
    const groupB = result.group_b;
    cards.push({
      title: "Group B",
      body: `${groupB.model.label} • ${groupB.run_count} run${groupB.run_count === 1 ? "" : "s"}`,
    });
  }
  cards.forEach((entry) => {
    const card = document.createElement("div");
    card.className = "cross-summary-card";
    card.innerHTML = `<strong>${entry.title}</strong><span>${entry.body}</span>`;
    elements.groupSummary.appendChild(card);
  });
}

function renderMetricSections(result) {
  elements.metricSections.innerHTML = "";
  const sectionResult = {
    groupA: result.group_a,
    groupB: result.group_b || null,
    delta: result.delta || null,
  };

  result.metric_catalog.forEach((group) => {
    const section = document.createElement("section");
    section.className = "cross-metric-section";
    section.dataset.groupId = group.group_id;

    const header = document.createElement("div");
    header.className = "cross-metric-section-header";
    header.innerHTML = `<h3>${group.group_label}</h3>`;
    section.appendChild(header);

    const grid = document.createElement("div");
    grid.className = "cross-metric-grid";
    group.metrics.forEach((metric) => {
      grid.appendChild(createMetricCard(metric, sectionResult));
    });
    section.appendChild(grid);
    elements.metricSections.appendChild(section);
  });
}

function renderDeltaSection(result) {
  elements.deltaSection.innerHTML = "";
}

function renderResults(result) {
  state.result = result;
  state.metricCatalog = result.metric_catalog || [];
  renderGroupSummary(result);
  renderMetricSections(result);
  renderDeltaSection(result);

  if (result.mode === "aggregate_selected") {
    elements.resultsTitle.textContent = "Aggregate result";
    elements.resultsMeta.textContent = `${result.group_a.model.label} averaged across ${result.group_a.run_count} runs`;
  } else if (result.mode === "compare_two_runs") {
    elements.resultsTitle.textContent = "Two-run comparison";
    elements.resultsMeta.textContent = `${result.group_a.model.label} prompt-to-prompt comparison`;
  } else {
    elements.resultsTitle.textContent = "Group comparison";
    elements.resultsMeta.textContent = `${result.group_a.model.label} vs ${result.group_b.model.label} on relative depth`;
  }
}

async function loadRuns() {
  setStatus("Loading", "busy");
  const payload = await fetchJson("/api/cross-inspect/runs");
  state.runs = payload.runs || [];
  elements.runCount.textContent = `${state.runs.length} logged run${state.runs.length === 1 ? "" : "s"} loaded`;
  renderRunList();
  updateSelectionMeta();
  setStatus("Ready", "ready");
}

function requestPayload() {
  const mode = elements.modeSelect.value;
  if (mode === "compare_groups") {
    return {
      mode,
      group_a_run_ids: selectedGroupRunIds("a"),
      group_b_run_ids: selectedGroupRunIds("b"),
    };
  }
  return {
    mode,
    run_ids: selectedRunIds(),
  };
}

async function analyzeSelection() {
  const message = validationMessage();
  updateSelectionMeta();
  if (message) {
    clearResults();
    return;
  }

  setStatus("Analyzing", "busy");
  elements.analyzeButton.disabled = true;
  try {
    const result = await fetchJson("/api/cross-inspect/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestPayload()),
    });
    renderResults(result);
    setStatus("Ready", "ready");
  } catch (error) {
    clearResults();
    elements.selectionError.textContent = error.message;
    setStatus("Idle", "idle");
  } finally {
    elements.analyzeButton.disabled = false;
  }
}

function selectAllVisible() {
  state.runs.forEach((run) => {
    const selection = ensureSelection(run.run_id);
    selection.checked = true;
  });
  renderRunList();
  updateSelectionMeta();
}

function clearSelections() {
  state.selections.clear();
  renderRunList();
  updateSelectionMeta();
}

elements.modeSelect.addEventListener("change", () => {
  renderRunList();
  updateSelectionMeta();
  clearResults();
});
elements.selectAllButton.addEventListener("click", selectAllVisible);
elements.clearSelectionButton.addEventListener("click", clearSelections);
elements.analyzeButton.addEventListener("click", analyzeSelection);

loadRuns().catch((error) => {
  setStatus("Idle", "idle");
  elements.runCount.textContent = error.message;
  elements.selectionError.textContent = error.message;
});
