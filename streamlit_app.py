from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from new_bank.base_cluster_gui import (
    DEFAULT_LANG,
    _build_bundle,
    _default_state,
    _encode_raw_row,
    _format_number,
    _predict,
    categorical_option_label,
    feature_label,
    subscale_label,
    tr,
)


st.set_page_config(
    page_title="Bank Two-Layer Additive Model",
    page_icon="🏦",
    layout="wide",
)


APP_CSS = """
<style>
  .stApp {
    background:
      radial-gradient(circle at top left, rgba(255, 198, 153, 0.25), transparent 28%),
      radial-gradient(circle at top right, rgba(179, 214, 255, 0.22), transparent 24%),
      linear-gradient(180deg, #fcfcfb 0%, #f6f3ee 100%);
  }
  .app-shell {
    max-width: 1400px;
    margin: 0 auto;
  }
  .hero {
    padding: 1.4rem 1.4rem 1rem;
    border: 1px solid rgba(60, 60, 60, 0.08);
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.72);
    backdrop-filter: blur(8px);
    box-shadow: 0 12px 30px rgba(33, 33, 33, 0.06);
    margin-bottom: 1rem;
  }
  .hero h1 {
    margin: 0;
    font-size: 2rem;
    line-height: 1.15;
  }
  .hero p {
    margin: 0.5rem 0 0;
    color: rgba(35, 35, 35, 0.78);
  }
  .summary-row {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    margin-top: 0.9rem;
  }
  .summary-pill {
    padding: 0.5rem 0.85rem;
    border-radius: 999px;
    border: 1px solid rgba(0, 0, 0, 0.08);
    background: rgba(255, 255, 255, 0.9);
    font-size: 0.92rem;
  }
  .result-box {
    border-radius: 18px;
    border: 1px solid rgba(0, 0, 0, 0.08);
    background: rgba(255, 255, 255, 0.78);
    padding: 1rem;
    margin-top: 0.8rem;
  }
  .section-title {
    font-size: 1.05rem;
    font-weight: 700;
    margin: 0 0 0.5rem;
  }
  .hint {
    color: rgba(0, 0, 0, 0.55);
    font-size: 0.92rem;
  }
</style>
"""


def _bundle() -> Any:
    return _build_bundle(None)


@st.cache_resource(show_spinner=True)
def load_bundle() -> Any:
    return _bundle()


def _widget_key(name: str) -> str:
    return f"input_{name}"


def _ensure_defaults(bundle: Any) -> dict[str, Any]:
    default_state = _default_state(bundle)
    for name, value in default_state.items():
        key = _widget_key(str(name))
        if key not in st.session_state:
            st.session_state[key] = value
    return default_state


def _render_top_banner(bundle: Any, lang: str) -> None:
    ap_text = f"{bundle.ap:.5f}" if bundle.ap is not None else "N/A"
    ratio_text = "100%"
    st.markdown(
        f"""
        <div class="hero">
          <h1>{tr(lang, "title")}</h1>
          <p>{tr(lang, "hero_desc")}</p>
          <div class="summary-row">
            <span class="summary-pill">Fold: {bundle.fold}</span>
            <span class="summary-pill">Validation AP: {ap_text}</span>
            <span class="summary-pill">Train ratio: {ratio_text}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _format_raw_value(lang: str, name: str, value: Any, kind: str | None) -> str:
    if kind == "categorical":
        return categorical_option_label(lang, name, str(value))
    if isinstance(value, float):
        return _format_number(value)
    return str(value)


def _render_results(bundle: Any, lang: str, result: dict[str, Any]) -> None:
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    left, right = st.columns([1.1, 0.9])

    with left:
        st.markdown(f'<div class="section-title">{tr(lang, "model_output")}</div>', unsafe_allow_html=True)
        st.metric(tr(lang, "predicted_probability"), f"{result['probability']:.1%}")
        st.write(f"{tr(lang, 'output')}: {result['risk_label']}")
        st.caption(f"Color: {result['color']}")
        st.caption(f"Fold label: {result['fold_label']}")

    with right:
        st.markdown(f'<div class="section-title">{tr(lang, "legend_title")}</div>', unsafe_allow_html=True)
        legend = pd.DataFrame(
            [
                {"label": tr(lang, "legend_low"), "score": 0.0},
                {"label": tr(lang, "legend_mid"), "score": 0.5},
                {"label": tr(lang, "legend_high"), "score": 1.0},
            ]
        )
        st.dataframe(legend, use_container_width=True, hide_index=True)

    st.markdown(f'<div class="section-title">{tr(lang, "original_features")}</div>', unsafe_allow_html=True)
    raw_rows = []
    for row in result["raw_rows"]:
        raw_rows.append(
            {
                tr(lang, "original_features"): feature_label(lang, str(row["name"])),
                "value": _format_raw_value(lang, str(row["name"]), row["value"], row.get("kind")),
            }
        )
    st.dataframe(pd.DataFrame(raw_rows), use_container_width=True, hide_index=True)

    st.markdown(f'<div class="section-title">{tr(lang, "subscale_features")}</div>', unsafe_allow_html=True)
    subscale_rows = []
    for row in result["subscale_rows"]:
        subscale_rows.append(
            {
                tr(lang, "subscale_features"): subscale_label(lang, str(row["name"])),
                tr(lang, "subscale_probability"): f"{row['score'] * 100:.1f}%",
                tr(lang, "subscale_weight"): "N/A" if row.get("weight") is None else f"{row['weight']:.3f}",
            }
        )
    st.dataframe(pd.DataFrame(subscale_rows), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _build_input_form(bundle: Any, lang: str, default_state: dict[str, Any]) -> dict[str, str] | None:
    submitted = False
    form_data: dict[str, str] = {}

    with st.form("bank_model_form", clear_on_submit=False):
        st.subheader(tr(lang, "input_panel"))
        st.caption(tr(lang, "run_prompt"))

        left, right = st.columns(2)

        with left:
            st.markdown(f"**{tr(lang, 'numeric_features')}**")
            for spec in bundle.feature_spec.numeric:
                name = str(spec["name"])
                label = feature_label(lang, name)
                key = _widget_key(name)
                default = st.session_state.get(key, default_state.get(name, spec["default"]))
                if spec["kind"] == "flag":
                    checked = st.checkbox(label, value=bool(int(default)), key=key)
                    form_data[name] = "1" if checked else "0"
                else:
                    min_value = float(spec["min"])
                    max_value = float(spec["max"])
                    step = float(spec["step"])
                    value = st.number_input(
                        label,
                        min_value=min_value,
                        max_value=max_value,
                        value=float(default),
                        step=step,
                        key=key,
                    )
                    form_data[name] = str(value)

        with right:
            st.markdown(f"**{tr(lang, 'categorical_features')}**")
            for spec in bundle.feature_spec.categorical:
                name = str(spec["name"])
                label = feature_label(lang, name)
                options = list(spec["options"])
                key = _widget_key(name)
                default = st.session_state.get(key, default_state.get(name, spec["default"]))
                if default not in options:
                    default = spec["default"]
                index = options.index(default)
                selected = st.selectbox(
                    label,
                    options=options,
                    index=index,
                    format_func=lambda opt, n=name: categorical_option_label(lang, n, str(opt)),
                    key=key,
                )
                form_data[name] = str(selected)

        submitted = st.form_submit_button(tr(lang, "run_button"), type="primary", use_container_width=True)

    return form_data if submitted else None


def main() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)
    bundle = load_bundle()
    default_state = _ensure_defaults(bundle)

    current_lang = str(st.session_state.get("lang", DEFAULT_LANG))
    lang = st.sidebar.selectbox(
        "Language",
        options=["ja", "en"],
        index=0 if current_lang == "ja" else 1,
        key="lang_selector",
    )
    st.session_state["lang"] = lang

    _render_top_banner(bundle, lang)

    form_data = _build_input_form(bundle, lang, default_state)
    if form_data is not None:
        encoded, ui_state = _encode_raw_row(bundle, form_data)
        ui_state["lang"] = lang
        result = _predict(bundle, encoded, ui_state, lang)
        _render_results(bundle, lang, result)
    else:
        st.info(tr(lang, "run_prompt"))

    st.markdown(f'<p class="hint">{tr(lang, "layer1_note")}</p>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
