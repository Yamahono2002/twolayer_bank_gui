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
}

function buildHeroStats() {
  const fold = state.data.fold;
  const ap = state.data.validation_ap == null ? "N/A" : Number(state.data.validation_ap).toFixed(5);
  els.heroStats.innerHTML = `
    <div class="stat-card">
      <div class="stat-label">Fold</div>
      <div class="stat-value">${fold}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Validation AP</div>
      <div class="stat-value">${ap}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Runtime</div>
      <div class="stat-value">Browser JS</div>
    </div>
  `;
}

function createControl(spec, type) {
  const wrapper = document.createElement("div");
  wrapper.className = "control";

  const label = document.createElement("label");
  label.textContent = featureLabel(spec.name);
  wrapper.appendChild(label);

  if (type === "numeric") {
    if (spec.kind === "flag") {
      const input = document.createElement("input");
      input.type = "checkbox";
      input.id = spec.name;
      input.checked = Boolean(Number(spec.currentValue ?? spec.default));
      wrapper.appendChild(input);
    } else {
      const input = document.createElement("input");
      input.type = "number";
      input.id = spec.name;
      input.step = spec.step ?? "1";
      input.min = spec.min;
      input.max = spec.max;
      input.value = spec.currentValue ?? spec.default;
      wrapper.appendChild(input);
    }
    return wrapper;
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
}

function readFormState() {
  const formData = {};
  for (const spec of state.data.feature_spec.numeric) {
    const el = $(spec.name);
    if (spec.kind === "flag") {
      formData[spec.name] = el.checked ? "1" : "0";
    } else {
      formData[spec.name] = String(el.value);
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

  const riskColor = `rgba(${Math.round(232 * finalProb + 184 * (1 - finalProb))}, ${Math.round(75 * finalProb + 184 * (1 - finalProb))}, ${Math.round(75 * finalProb + 184 * (1 - finalProb))}, 1)`;

  els.resultRoot.innerHTML = `
    <div class="result-card">
      <div class="result-grid">
        <div class="result-top">
          <div class="metric">
            <div class="metric-label">${uiText("predicted_probability")}</div>
            <div class="metric-value">${(finalProb * 100).toFixed(1)}%</div>
          </div>
          <div class="metric">
            <div class="metric-label">${uiText("output")}</div>
            <div class="metric-value">${riskLabel}</div>
          </div>
          <div class="metric">
            <div class="metric-label">Fold</div>
            <div class="metric-value">${state.data.fold}</div>
          </div>
        </div>

        <div class="badge" style="border-color:${riskColor};">
          <span>Color</span>
          <strong>${riskColor}</strong>
        </div>
      </div>
    </div>

    <div class="result-card">
      <div class="section-title">${uiText("original_features")}</div>
      <div class="list">
        ${rawItems
          .map(
            (item) => `
              <div class="list-item">
                <span>${item.name}</span>
                <strong>${item.value}</strong>
              </div>`,
          )
          .join("")}
      </div>
    </div>

    <div class="result-card">
      <div class="section-title">${uiText("subscale_features")}</div>
      <div class="list">
        ${subscaleRows
          .map(
            (item) => `
              <div class="subscale-card">
                <div class="subscale-header">
                  <span class="subscale-name">${item.name}</span>
                  <span>${(item.weight ?? 0).toFixed(3)}</span>
                </div>
                <div class="subscale-values">
                  <div class="subscale-value" style="background:#f4ddd5;">${(item.probability * 100).toFixed(1)}%</div>
                  <div class="subscale-value" style="background:#d7e4ef;">${(item.weight ?? 0).toFixed(3)}</div>
                </div>
              </div>`,
          )
          .join("")}
      </div>
    </div>
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
    root.innerHTML = `<div class="result-card">Failed to load model data: ${String(error)}</div>`;
  }
});
