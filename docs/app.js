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

function renderResult(formData) {
  const { uiState, subscaleScores, finalProb } = predict(formData);
  const riskLabel = finalProb >= 0.5 ? uiText("high_risk") : uiText("low_risk");
  const textColor = finalProb >= 0.5 ? "#ffffff" : "#111111";
  const bg = finalProb >= 0.5 ? "#e84b4b" : "#b8b8b8";
  const summaryMeta = state.data.validation_ap == null ? "" : `Validation AP ${Number(state.data.validation_ap).toFixed(5)}`;

  const rawItems = state.data.feature_spec.numeric
    .map((spec) => ({
      name: featureLabel(spec.name),
      value: spec.kind === "flag" ? Number(uiState[spec.name]) : Number(uiState[spec.name]).toFixed(4).replace(/\.?0+$/, ""),
    }))
    .concat(
      state.data.feature_spec.categorical.map((spec) => ({
        name: featureLabel(spec.name),
        value: optionLabel(spec.name, uiState[spec.name]),
      })),
    );

  const subscaleRows = state.data.subscale_names.map((name, index) => ({
    name: subscaleLabel(name),
    probability: subscaleScores[index],
    weight: state.data.model.final_clf.weights[index] ?? 0,
  }));

  els.resultRoot.innerHTML = `
    <div class="summary-card" style="background:${bg};color:${textColor}">
      <div class="summary-kicker">${uiText("model_output")}</div>
      <div class="summary-value">${(finalProb * 100).toFixed(1)}%</div>
      <div class="summary-label">${riskLabel}</div>
      <div class="summary-meta">${summaryMeta}</div>
    </div>

    <section class="network-card">
      <div class="section-head">
        <h2>${uiText("structure_title")}</h2>
        <p>${uiText("layer1_note")}</p>
      </div>
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
    </section>

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
          <div class="output-fill" style="width:${(finalProb * 100).toFixed(1)}%;background:${bg}"></div>
        </div>
        <div class="output-readout">${uiText("predicted_probability")} ${(finalProb * 100).toFixed(1)}%</div>
        <div class="output-caption">${riskLabel}</div>
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
