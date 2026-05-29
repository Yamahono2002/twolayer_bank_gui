const state = {
  data: null,
  lang: "ja",
};

const els = {};

function $(id) {
  return document.getElementById(id);
}

function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

function dot(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    sum += a[i] * b[i];
  }
  return sum;
}

function normalizeCategoricalOption(feature, option) {
  if (option.startsWith(`${feature}_`)) {
    return option.slice(feature.length + 1);
  }
  if (feature === "day_of_week" && option.startsWith("of_week_")) {
    return option.split("of_week_")[1];
  }
  return option;
}

function featureLabel(name) {
  return state.data.feature_labels[state.lang][name] ?? name;
}

function subscaleLabel(name) {
  return state.data.subscale_labels[state.lang][name] ?? name;
}

function optionLabel(feature, option) {
  const langMap = state.data.categorical_option_labels[state.lang] ?? {};
  const featureMap = langMap[feature] ?? {};
  return featureMap[option] ?? normalizeCategoricalOption(feature, option);
}

function uiText(key) {
  return state.data.ui_text[state.lang][key] ?? key;
}

function formatNumber(value) {
  if (Number.isInteger(value)) {
    return String(value);
  }
  return value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

function buildLanguageSelector() {
  els.languageSelect.innerHTML = "";
  for (const lang of Object.keys(state.data.ui_text)) {
    const option = document.createElement("option");
    option.value = lang;
    option.textContent = lang === "ja" ? "日本語" : "English";
    if (lang === state.lang) {
      option.selected = true;
    }
    els.languageSelect.appendChild(option);
  }
}

function renderStaticLabels() {
  els.title.textContent = uiText("title");
  els.subtitle.textContent = uiText("hero_desc");
  els.inputTitle.textContent = uiText("input_panel");
  els.numericTitle.textContent = uiText("numeric_features");
  els.categoricalTitle.textContent = uiText("categorical_features");
  els.runButton.textContent = uiText("run_button");
  els.outputTitle.textContent = uiText("model_output");
  els.placeholderText.textContent = uiText("run_prompt");
  els.eyebrow.textContent = state.lang === "ja" ? "GitHub Pages デモ" : "GitHub Pages Demo";
  if (els.languageLabel) {
    els.languageLabel.textContent = uiText("language");
  }
  if (els.helperText) {
    els.helperText.textContent = uiText("layer1_note");
  }
}

function buildHeroStats() {
  const fold = state.data.fold;
  const ap = state.data.validation_ap == null ? "N/A" : Number(state.data.validation_ap).toFixed(5);
  els.heroStats.innerHTML = `Fold ${fold} | Validation AP ${ap} | Browser JS`;
}

function createControl(spec, type) {
  const wrapper = document.createElement("div");
  wrapper.className = "control";

  const label = document.createElement("label");
  label.textContent = featureLabel(spec.name);
  wrapper.appendChild(label);

  if (type === "numeric") {
    if (spec.kind === "flag") {
      const select = document.createElement("select");
      select.id = spec.name;
      const options = [
        { value: "0", text: state.lang === "ja" ? "0: あり" : "0: present" },
        { value: "1", text: state.lang === "ja" ? "1: なし" : "1: absent" },
      ];
      for (const item of options) {
        const option = document.createElement("option");
        option.value = item.value;
        option.textContent = item.text;
        if (item.value === String(spec.currentValue ?? spec.default)) {
          option.selected = true;
        }
        select.appendChild(option);
      }
      wrapper.classList.add("categorical-control");
      wrapper.appendChild(select);
      return wrapper;
    } else {
      wrapper.classList.add("numeric-control");

      const row = document.createElement("div");
      row.className = "control-label-row";

      const value = document.createElement("span");
      value.className = "value-pill";
      value.id = `${spec.name}_value`;
      value.textContent = formatNumber(Number(spec.currentValue ?? spec.default));

      row.appendChild(label);
      row.appendChild(value);
      wrapper.appendChild(row);

      const range = document.createElement("input");
      range.type = "range";
      range.id = `${spec.name}_range`;
      range.min = spec.min;
      range.max = spec.max;
      range.step = spec.step ?? "1";
      range.value = spec.currentValue ?? spec.default;
      wrapper.appendChild(range);

      const foot = document.createElement("div");
      foot.className = "range-foot";
      foot.innerHTML = `
        <span>${formatNumber(Number(spec.min))}</span>
        <input type="number" id="${spec.name}_num" value="${spec.currentValue ?? spec.default}" step="${spec.step ?? "1"}" min="${spec.min}" max="${spec.max}" />
        <span>${formatNumber(Number(spec.max))}</span>
      `;
      wrapper.appendChild(foot);

      return wrapper;
    }
  }

  const select = document.createElement("select");
  select.id = spec.name;
  for (const optionValue of spec.options) {
    const option = document.createElement("option");
    option.value = optionValue;
    option.textContent = optionLabel(spec.name, optionValue);
    if (optionValue === (spec.currentValue ?? spec.default)) {
      option.selected = true;
    }
    select.appendChild(option);
  }
  wrapper.appendChild(select);
  return wrapper;
}

function buildControls(currentValues = {}) {
  els.numericControls.innerHTML = "";
  els.categoricalControls.innerHTML = "";

  for (const spec of state.data.feature_spec.numeric) {
    els.numericControls.appendChild(createControl({ ...spec, currentValue: currentValues[spec.name] }, "numeric"));
  }

  for (const spec of state.data.feature_spec.categorical) {
    els.categoricalControls.appendChild(createControl({ ...spec, currentValue: currentValues[spec.name] }, "categorical"));
  }

  for (const spec of state.data.feature_spec.numeric) {
    if (spec.kind === "flag") {
      continue;
    }
    const range = $(`${spec.name}_range`);
    const num = $(`${spec.name}_num`);
    const value = $(`${spec.name}_value`);
    if (!range || !num || !value) {
      continue;
    }
    const sync = (raw) => {
      const parsed = Number(raw);
      const clamped = Math.min(Math.max(Number.isFinite(parsed) ? parsed : Number(range.value), Number(range.min)), Number(range.max));
      range.value = String(clamped);
      num.value = String(clamped);
      value.textContent = formatNumber(clamped);
    };
    range.addEventListener("input", () => sync(range.value));
    num.addEventListener("input", () => sync(num.value));
    sync(range.value);
  }
}

function readFormState() {
  const formData = {};
  for (const spec of state.data.feature_spec.numeric) {
    if (spec.kind === "flag") {
      formData[spec.name] = String($(spec.name).value);
    } else {
      formData[spec.name] = String($(`${spec.name}_range`).value);
    }
  }
  for (const spec of state.data.feature_spec.categorical) {
    formData[spec.name] = String($(spec.name).value);
  }
  return formData;
}

function digitize(value, thresholds) {
  let idx = 0;
  while (idx < thresholds.length && value >= thresholds[idx]) {
    idx += 1;
  }
  return idx;
}

function encodeRawRow(formData) {
  const encoded = {};
  const modelColumns = state.data.model_header.filter((name) => name !== "y");
  for (const col of modelColumns) {
    encoded[col] = 0;
  }

  const uiState = {};
  let pdaysEq999Value = null;
  let pdaysLt999Value = null;

  for (const spec of state.data.feature_spec.numeric) {
    const name = spec.name;
    if (spec.kind === "flag") {
      const value = Number(formData[name] ?? spec.default);
      uiState[name] = value;
      const cols = state.data.feature_spec.feature_to_model_columns[name] || [];
      if (cols.length > 0) {
        encoded[cols[0]] = value;
      }
      if (name === "pdays_eq_999") {
        pdaysEq999Value = value;
      }
      continue;
    }

    const rawValue = Number(formData[name] ?? spec.default);
    uiState[name] = rawValue;
    if (name === "pdays_lt_999") {
      pdaysLt999Value = rawValue;
    }
    const thresholds = state.data.feature_spec.numeric_thresholds[name] || [];
    const cols = state.data.feature_spec.feature_to_model_columns[name] || [];
    if (cols.length > 0) {
      const binIdx = Math.max(0, Math.min(digitize(rawValue, thresholds), cols.length - 1));
      encoded[cols[binIdx]] = 1;
    }
  }

  const pdaysLtCols = state.data.feature_spec.feature_to_model_columns.pdays_lt_999 || [];
  const pdaysEqCols = state.data.feature_spec.feature_to_model_columns.pdays_eq_999 || [];
  if (pdaysEq999Value === 1) {
    if (pdaysLtCols.length > 0) {
      encoded[pdaysLtCols[0]] = 0;
    }
    uiState.pdays_lt_999 = 0;
  } else if (pdaysLt999Value != null && pdaysLt999Value > 0) {
    if (pdaysEqCols.length > 0) {
      encoded[pdaysEqCols[0]] = 0;
    }
    uiState.pdays_eq_999 = 0;
  }

  for (const spec of state.data.feature_spec.categorical) {
    const name = spec.name;
    const selected = String(formData[name] ?? spec.default);
    uiState[name] = selected;
    const target = `${name}_${selected}`;
    const cols = state.data.feature_spec.feature_to_model_columns[name] || [];
    if (Object.prototype.hasOwnProperty.call(encoded, target)) {
      encoded[target] = 1;
    } else if (cols.length > 0) {
      encoded[cols[0]] = 1;
    }
  }

  return { encoded, uiState, modelColumns };
}

function predict(formData) {
  const { encoded, uiState, modelColumns } = encodeRawRow(formData);
  const values = modelColumns.map((col) => Number(encoded[col] ?? 0));

  const subscaleCounts = state.data.model.subscale_num_attributes;
  const subscaleScores = [];
  let offset = 0;
  for (let i = 0; i < subscaleCounts.length; i += 1) {
    const nAttr = Number(subscaleCounts[i]);
    const slice = values.slice(offset, offset + nAttr);
    const clf = state.data.model.subscale_clfs[i];
    const linear = dot(slice, clf.weights) + clf.intercept;
    const prob = sigmoid(linear);
    subscaleScores.push(prob);
    offset += nAttr;
  }

  const final = state.data.model.final_clf;
  const finalProb = sigmoid(dot(subscaleScores, final.weights) + final.intercept);
  return { uiState, subscaleScores, finalProb };
}

function riskColor(prob) {
  const p = Math.max(0, Math.min(1, Number(prob)));
  const low = [0xb8, 0xb8, 0xb8];
  const high = [0xe8, 0x4b, 0x4b];
  const mix = (a, b) => Math.round(a * (1 - p) + b * p);
  return `#${[mix(low[0], high[0]), mix(low[1], high[1]), mix(low[2], high[2])]
    .map((v) => v.toString(16).padStart(2, "0"))
    .join("")}`;
}

function textColor(bgHex) {
  const hex = bgHex.replace("#", "");
  const r = Number.parseInt(hex.slice(0, 2), 16);
  const g = Number.parseInt(hex.slice(2, 4), 16);
  const b = Number.parseInt(hex.slice(4, 6), 16);
  return 0.299 * r + 0.587 * g + 0.114 * b > 160 ? "#111111" : "#ffffff";
}

function blendHex(baseHex, targetHex, ratio) {
  const clamp = (v) => Math.max(0, Math.min(1, v));
  const p = clamp(ratio);
  const parse = (hex) => {
    const value = hex.replace("#", "");
    return [
      Number.parseInt(value.slice(0, 2), 16),
      Number.parseInt(value.slice(2, 4), 16),
      Number.parseInt(value.slice(4, 6), 16),
    ];
  };
  const [br, bg, bb] = parse(baseHex);
  const [tr, tg, tb] = parse(targetHex);
  const toHex = (n) => Math.round(n).toString(16).padStart(2, "0");
  return `#${toHex(br * (1 - p) + tr * p)}${toHex(bg * (1 - p) + tg * p)}${toHex(bb * (1 - p) + tb * p)}`;
}

function metricColor(value, scale = 1) {
  const magnitude = Math.min(1, Math.abs(Number(value)) / (scale > 0 ? scale : 1));
  return riskColor(magnitude);
}

function rawValueLabel(name, value, kind) {
  if (kind === "categorical") {
    return optionLabel(name, String(value));
  }
  if (typeof value === "number") {
    return formatNumber(value);
  }
  const n = Number(value);
  return Number.isFinite(n) ? formatNumber(n) : String(value);
}

function subscaleSourceNames(name) {
  return name === "pdays" ? ["pdays_eq_999", "pdays_lt_999"] : [name];
}

function renderProbabilityLegend() {
  return `
    <div class="output-legend">
      <div class="legend-title">${state.lang === "ja" ? "色と大きさの対応" : "Color-to-magnitude map"}</div>
      <div class="legend-bar" aria-hidden="true">
        <span class="legend-stop legend-low"></span>
        <span class="legend-stop legend-mid"></span>
        <span class="legend-stop legend-high"></span>
      </div>
      <div class="legend-labels">
        <span>${state.lang === "ja" ? "低" : "Low"}</span>
        <span>${state.lang === "ja" ? "中" : "Mid"}</span>
        <span>${state.lang === "ja" ? "高" : "High"}</span>
      </div>
    </div>
  `;
}

function renderNetworkDiagram(rawItems, subscaleRows, finalProb) {
  const leftX = 60;
  const middleX = 460;
  const rightX = 920;
  const originalWidth = 330;
  const subscaleWidth = 250;
  const outputWidth = 220;
  const originalHeight = 36;
  const subscaleHeight = 64;
  const outputHeight = 76;
  const rowGap = 72;
  const topMargin = 70;
  const pdaysRowOffset = -12;
  const laterRowsOffset = 10;

  const rowY = (idx) => topMargin + idx * rowGap + (idx === 0 ? pdaysRowOffset : laterRowsOffset);
  const originalNodeTop = (baseY, height) => baseY + (subscaleHeight - height) / 2;
  const originalMap = new Map(rawItems.map((row) => [row.rawName ?? row.name, row]));
  const pdaysSources = subscaleSourceNames("pdays");
  const originalHasPdaysPair = pdaysSources.every((name) => originalMap.has(name));

  const orderedOriginalRows = [];
  if (originalHasPdaysPair) {
    orderedOriginalRows.push({ name: "pdays", kind: "group", sources: pdaysSources });
    for (const row of rawItems) {
      if (pdaysSources.includes(row.rawName ?? row.name)) {
        continue;
      }
      orderedOriginalRows.push(row);
    }
  } else {
    orderedOriginalRows.push(...rawItems);
  }

  const orderedSubscaleRows = [...subscaleRows].sort((a, b) => ((a.rawName ?? a.name) === "pdays" ? -1 : (b.rawName ?? b.name) === "pdays" ? 1 : 0));
  const maxRows = Math.max(orderedOriginalRows.length, orderedSubscaleRows.length);
  const stageHeight = topMargin * 2 + (maxRows - 1) * rowGap + 130 + (laterRowsOffset - pdaysRowOffset);
  const outputCy = stageHeight / 2;
  const outputTop = outputCy - outputHeight / 2;

  const originalNodes = [];
  const subscaleNodes = [];
  const lines = [];
  const markerDefs = [];
  const originalPositions = new Map();

  orderedOriginalRows.forEach((row, idx) => {
    const y = rowY(idx);
    if (row.kind === "group") {
      const groupHeight = originalHeight * 2 + 10;
      const nodeTop = originalNodeTop(y, groupHeight);
      const cy = nodeTop + groupHeight / 2;
      for (const sourceName of row.sources) {
        originalPositions.set(sourceName, { cy });
      }
      const sourceBits = row.sources
        .map((sourceName) => {
          const sourceRow = originalMap.get(sourceName);
          if (!sourceRow) return "";
          return `
            <div class="group-line">
              <span class="group-label">${featureLabel(sourceName)}</span>
              <span class="group-value">${rawValueLabel(sourceName, sourceRow.value, sourceRow.kind)}</span>
            </div>
          `;
        })
        .join("");

      originalNodes.push(`
        <div class="diagram-node original-node group-node" style="left:${leftX}px;top:${nodeTop}px;width:${originalWidth}px;height:${groupHeight}px;">
          <div class="node-title">${featureLabel("pdays")}</div>
          <div class="group-lines">${sourceBits}</div>
        </div>
      `);
      return;
    }

    const nodeTop = originalNodeTop(y, originalHeight);
    const cy = nodeTop + originalHeight / 2;
    originalPositions.set(row.rawName ?? row.name, { cy });
    originalNodes.push(`
      <div class="diagram-node original-node" style="left:${leftX}px;top:${nodeTop}px;width:${originalWidth}px;height:${originalHeight}px;">
        <div class="node-title">${featureLabel(row.rawName ?? row.name)}</div>
        <div class="node-value">${rawValueLabel(row.rawName ?? row.name, row.value, row.kind)}</div>
      </div>
    `);
  });

  const scaleForWeight = Math.max(...orderedSubscaleRows.map((row) => Math.abs(Number(row.weight ?? 0))), 0) || 1;
  let combinedScale = 0;
  orderedSubscaleRows.forEach((row) => {
    combinedScale = Math.max(combinedScale, Math.abs(Number(row.probability) * Number(row.weight ?? 0)));
  });
  if (combinedScale <= 0) combinedScale = 1;

  orderedSubscaleRows.forEach((row, idx) => {
    const y = rowY(idx);
    const cy = y + subscaleHeight / 2;
    const scoreColor = riskColor(row.probability);
    const weight = Number(row.weight ?? 0);
    const weightColor = metricColor(weight, scaleForWeight);
    const combinedColor = metricColor(row.probability * weight, combinedScale);
    const scoreTextColor = textColor(scoreColor);
    const weightTextColor = "#111111";
    const combinedTextColor = textColor(combinedColor);
    const weightText = Number.isFinite(weight) ? weight.toFixed(3) : "N/A";
    const baseLine = blendHex(scoreColor, "#ffffff", 0.45);

    subscaleNodes.push(`
      <div class="diagram-node subscale-node" style="left:${middleX}px;top:${y}px;width:${subscaleWidth}px;height:${subscaleHeight}px;">
        <div class="node-title">${subscaleLabel(row.rawName ?? row.name)}</div>
        <div class="subscale-split">
          <div class="subscale-half subscale-half-score" style="background:${scoreColor};color:${scoreTextColor}">
            <div class="half-value">${(row.probability * 100).toFixed(1)}%</div>
          </div>
          <div class="subscale-half subscale-half-weight" style="background:${weightColor};color:${weightTextColor}">
            <div class="half-value">${weightText}</div>
          </div>
        </div>
      </div>
    `);

    subscaleSourceNames(row.rawName ?? row.name).forEach((sourceName, sourceIdx) => {
      const pos = originalPositions.get(sourceName);
      if (!pos) return;
      const x1 = leftX + originalWidth;
      const x2 = middleX;
      lines.push(`<line x1="${x1}" y1="${pos.cy}" x2="${x2}" y2="${cy}" stroke="${baseLine}" stroke-width="2.2" marker-end="url(#arrow-in-${idx}-${sourceIdx})" />`);
      markerDefs.push(`
        <marker id="arrow-in-${idx}-${sourceIdx}" markerWidth="8" markerHeight="8" refX="6" refY="3.5" orient="auto" markerUnits="strokeWidth">
          <polygon points="0 0, 7 3.5, 0 7" fill="${baseLine}" />
        </marker>
      `);
    });

    const x3 = middleX + subscaleWidth;
    const x4 = rightX;
    lines.push(`<line x1="${x3}" y1="${cy}" x2="${x4}" y2="${outputCy}" stroke="${combinedColor}" stroke-width="2.8" marker-end="url(#arrow-out-${idx})" />`);
    markerDefs.push(`
      <marker id="arrow-out-${idx}" markerWidth="8" markerHeight="8" refX="6" refY="3.5" orient="auto" markerUnits="strokeWidth">
        <polygon points="0 0, 7 3.5, 0 7" fill="${combinedColor}" />
      </marker>
    `);
  });

  const outputColor = riskColor(finalProb);
  const outputTextColor = textColor(outputColor);

  const subscaleHeaderHtml = `
    <div class="network-col-label" style="left:${middleX + subscaleWidth / 2}px;top:18px;width:${subscaleWidth}px;">
      <div style="display:flex;justify-content:space-between;gap:8px;width:100%;">
        <span style="margin-left:26px;">${uiText("subscale_probability")}</span>
        <span style="margin-right:28px;">${uiText("subscale_weight")}</span>
      </div>
    </div>
  `;

  const outputNode = `
      <div class="diagram-node output-node" style="left:${rightX}px;top:${outputTop}px;width:${outputWidth}px;height:${outputHeight}px;background:${outputColor};color:${outputTextColor};">
        <div class="node-title">${uiText("output")}</div>
        <div class="node-value">${uiText("predicted_probability")} ${(finalProb * 100).toFixed(1)}%</div>
      </div>
    `;

  const svgWidth = rightX + outputWidth + 40;
  return `
    <section class="network-card">
      <div class="section-head">
        <h2>${uiText("structure_title")}</h2>
        <p>${uiText("layer1_note")}</p>
      </div>
      <div class="network-stage" style="height:${stageHeight}px;">
        <div class="network-legend-anchor">
          ${renderProbabilityLegend()}
        </div>
        <svg class="network-links" width="${svgWidth}" height="${stageHeight}" viewBox="0 0 ${svgWidth} ${stageHeight}" preserveAspectRatio="none">
          <defs>${markerDefs.join("")}</defs>
          ${lines.join("")}
        </svg>
        ${subscaleHeaderHtml}
        ${originalNodes.join("")}
        ${subscaleNodes.join("")}
        ${outputNode}
      </div>
    </section>
  `;
}

function renderResult(formData) {
  const { uiState, subscaleScores, finalProb } = predict(formData);
  const summaryColor = riskColor(finalProb);
  const summaryTextColor = textColor(summaryColor);
  const predictionLabel = state.lang === "ja"
    ? "定期預金申し込み確率"
    : "Term deposit subscription probability";
  const summaryMeta = state.data.validation_ap == null ? "" : `Validation AP ${Number(state.data.validation_ap).toFixed(5)}`;

  const rawItems = state.data.feature_spec.numeric
    .map((spec) => ({
      rawName: spec.name,
      name: featureLabel(spec.name),
      value: spec.kind === "flag" ? Number(uiState[spec.name]) : Number(uiState[spec.name]).toFixed(4).replace(/\.?0+$/, ""),
      kind: spec.kind,
    }))
    .concat(
      state.data.feature_spec.categorical.map((spec) => ({
        rawName: spec.name,
        name: featureLabel(spec.name),
        value: optionLabel(spec.name, uiState[spec.name]),
        kind: "categorical",
      })),
    );

  const subscaleRows = state.data.subscale_names.map((name, index) => ({
    rawName: name,
    name: subscaleLabel(name),
    probability: subscaleScores[index],
    weight: state.data.model.final_clf.weights[index] ?? 0,
  }));

  els.resultRoot.innerHTML = `
    <div class="summary-card" style="background:${summaryColor};color:${summaryTextColor}">
      <div class="summary-kicker">${uiText("model_output")}</div>
      <div class="summary-value">${(finalProb * 100).toFixed(1)}%</div>
      <div class="summary-label">${predictionLabel}</div>
      <div class="summary-meta">${summaryMeta}</div>
    </div>
    ${renderNetworkDiagram(rawItems, subscaleRows, finalProb)}

    <section class="result-section">
      <div class="section-head">
        <h2>${uiText("original_features")}</h2>
        <p>${uiText("raw_desc")}</p>
      </div>
      <div class="encoded-grid">
        ${rawItems
          .map(
            (item) => `
              <div class="feature-card">
                <div class="feature-name">${item.name}</div>
                <div class="feature-value">${item.value}</div>
              </div>`,
          )
          .join("")}
      </div>
    </section>

    <section class="result-section">
      <div class="section-head">
        <h2>${uiText("subscale_features")}</h2>
        <p>${uiText("subscale_desc")}</p>
      </div>
      <div class="card-stack">
        ${subscaleRows
          .map(
            (item) => `
              <div class="subscale-card">
                <div class="subscale-name">${item.name}<span class="subscale-weight">${item.weight.toFixed(3)}</span></div>
                <div class="subscale-metric-head">
                  <span>${uiText("subscale_probability")}</span>
                  <span>${uiText("subscale_weight")}</span>
                </div>
                <div class="subscale-split">
                  <div class="subscale-half subscale-half-score" style="background:#f4ddd5;color:#111111">
                    <div class="half-value">${(item.probability * 100).toFixed(1)}%</div>
                  </div>
                  <div class="subscale-half" style="background:#d7e4ef;color:#111111">
                    <div class="half-value">${item.weight.toFixed(3)}</div>
                  </div>
                </div>
              </div>`,
          )
          .join("")}
      </div>
    </section>

    <section class="result-section">
      <div class="section-head">
        <h2>${uiText("output")}</h2>
        <p>${uiText("output_desc")}</p>
      </div>
      <div class="output-panel">
        <div class="output-title">${uiText("predicted_probability")}</div>
        <div class="output-bar">
          <div class="output-fill" style="width:${(finalProb * 100).toFixed(1)}%;background:${summaryColor}"></div>
        </div>
        <div class="output-readout">${uiText("predicted_probability")} ${(finalProb * 100).toFixed(1)}%</div>
        <div class="output-legend">
          <div class="legend-title">${state.lang === "ja" ? "色と大きさの対応" : "Color-to-magnitude map"}</div>
          <div class="legend-bar" aria-hidden="true">
            <span class="legend-stop legend-low"></span>
            <span class="legend-stop legend-mid"></span>
            <span class="legend-stop legend-high"></span>
          </div>
          <div class="legend-labels">
            <span>${state.lang === "ja" ? "低" : "Low"}</span>
            <span>${state.lang === "ja" ? "中" : "Mid"}</span>
            <span>${state.lang === "ja" ? "高" : "High"}</span>
          </div>
        </div>
      </div>
    </section>
  `;
}

function initEvents() {
  els.languageSelect.addEventListener("change", () => {
    const currentValues = readFormState();
    state.lang = els.languageSelect.value;
    renderStaticLabels();
    buildControls(currentValues);
    buildHeroStats();
  });

  els.form.addEventListener("submit", (event) => {
    event.preventDefault();
    renderResult(readFormState());
  });
}

async function init() {
  els.title = $("title");
  els.subtitle = $("subtitle");
  els.inputTitle = $("input-title");
  els.numericTitle = $("numeric-title");
  els.categoricalTitle = $("categorical-title");
  els.languageLabel = $("language-label");
  els.runButton = $("run-button");
  els.outputTitle = $("output-title");
  els.placeholderText = $("placeholder-text");
  els.eyebrow = $("eyebrow");
  els.heroStats = $("hero-stats");
  els.languageSelect = $("language-select");
  els.numericControls = $("numeric-controls");
  els.categoricalControls = $("categorical-controls");
  els.form = $("prediction-form");
  els.resultRoot = $("result-root");
  els.helperText = $("helper-text");
  els.outputMeta = $("output-meta");

  const response = await fetch("./model-data.json");
  state.data = await response.json();
  state.lang = "ja";

  buildLanguageSelector();
  renderStaticLabels();
  buildControls();
  buildHeroStats();
  initEvents();
}

init().catch((error) => {
  console.error(error);
  const root = $("result-root");
  if (root) {
    root.innerHTML = `<div class="error-banner">Failed to load model data: ${String(error)}</div>`;
  }
});
