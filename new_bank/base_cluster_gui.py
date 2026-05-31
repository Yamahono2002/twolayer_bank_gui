#!/usr/bin/env python3
from __future__ import annotations

import ast
import colorsys
import html as html_lib
import json
import os
import re
import textwrap
from dataclasses import dataclass
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "new_bank" / "base_cluster.ipynb"
RAW_DATA_PATH = ROOT / "new_bank" / "data" / "bank_preprocessed.csv"
MODEL_TRAIN_TEMPLATE = ROOT / "new_bank" / "data" / "js_10" / "10_bank_js_train_fold{fold}.csv"
HOST = "127.0.0.1"
PORT = 8002

# 0.1 で 10%、1.0 で 100% に切り替えられる。
TRAIN_RATIO = 1.0
DATA_RATIO_TAG = "100pct" if TRAIN_RATIO < 1.0 else "100pct"

# GUI のデフォルト言語。
DEFAULT_LANG = "ja"

# GUI 側では明示的に使わない。
USE_SMOTE = False
USE_MILO_FINAL_LAYER = False

SUMMARY_CANDIDATES = [
    ROOT / "new_bank" / "AA_base_cluster.txt",
]

UI_TEXT = {
    "ja": {
        "title": "銀行 2 層加法モデル GUI",
        "language": "言語",
        "structure_title": "2 層加法構造",
        "hero_title": "グローバルモデル",
        "hero_desc": "下の入力パネルで銀行特徴量を設定し、ノートブック由来の 2 層モデルを実行すると、元の入力、サブスケール確率、最終的なリスク確率を確認できます。",
        # "model_tag": "ノートブック fold",
        "input_panel": "入力パネル",
        "numeric_features": "数値特徴量",
        "categorical_features": "カテゴリ特徴量",
        "run_button": "モデルを実行",
        # "helper_text": "この GUI は `new_bank/base_cluster.ipynb` を使い、学習は `new_bank/data/js_10/` の fold CSV、入力欄は `bank_preprocessed.csv` 由来の生の特徴量を使います。",
        "ready_label": "モデル準備完了",
        "run_prompt": "モデルを実行すると予測を表示します",
        "model_output": "モデル出力",
        "original_features": "元の特徴量",
        "raw_desc": "ユーザーが入力した元の特徴量です。",
        "subscale_features": "サブスケール特徴量",
        "subscale_desc": "ノートブックモデルが推定した 2 層目の確率です。特徴量選択は GUI では無効です。",
        "output": "出力",
        "output_desc": "確率が高いほど赤、低いほど灰色になります。",
        "subscale_probability": "予測確率",
        "subscale_weight": "重み",
        "predicted_probability": "予測確率",
        "low_risk": "低リスク",
        "mid_risk": "中間",
        "high_risk": "高リスク",
        "legend_title": "色と大きさの対応",
        "legend_low": "低",
        "legend_mid": "中",
        "legend_high": "高",
        # "legend_note": "大きいほど赤",
        "layer1_note": "第 1 層は元の特徴量です。第 2 層はサブスケールごとの予測確率です。最終層は定期預金を申し込む確率を示します。",
        # "status_text": "検証 AP",
    },
    "en": {
        "title": "Bank Two-Layer Additive Model GUI",
        "language": "Language",
        "structure_title": "Two-Layer Additive Structure",
        "hero_title": "Global Model",
        "hero_desc": "Use the input panel below to set bank features, then run the notebook-based two-layer model to inspect the raw inputs, subscale probabilities, and final risk score.",
        # "model_tag": "Notebook fold",
        "input_panel": "Input Panel",
        "numeric_features": "Numeric Features",
        "categorical_features": "Categorical Features",
        "run_button": "Run Model",
        # "helper_text": "This GUI uses `new_bank/base_cluster.ipynb`, trains on the fold CSVs under `new_bank/data/js_10/`, and builds the input panel from the raw preprocessed bank features.",
        "ready_label": "Model Ready",
        "run_prompt": "Run the model to see a prediction",
        "model_output": "Model Output",
        "original_features": "Original Features",
        "raw_desc": "Raw feature values entered by the user.",
        "subscale_features": "Subscale Features",
        "subscale_desc": "Layer-2 probabilities estimated by the notebook model. Feature selection is disabled in this GUI.",
        "output": "Output",
        "output_desc": "High risk is red, low risk is gray.",
        "risk_of_default": "Risk of Default",
        "subscale_probability": "Predicted Probability",
        "subscale_weight": "Weight",
        "predicted_probability": "Predicted Probability",
        "low_risk": "Low risk",
        "mid_risk": "Mid",
        "high_risk": "High risk",
        "legend_title": "Color-to-magnitude map",
        "legend_low": "Low",
        "legend_mid": "Mid",
        "legend_high": "High",
        # "legend_note": "higher magnitude is red",
        "layer1_note": "Layer 1 shows the raw bank features. Layer 2 shows the subscale scores estimated by the notebook model. The final card shows the integrated default probability.",
        # "status_text": "validation AP",
    },
}


def tr(lang: str, key: str) -> str:
    return UI_TEXT.get(lang, UI_TEXT["en"]).get(key, key)


FEATURE_LABELS = {
    "ja": {
        "age": "年齢",
        "campaign": "接触回数",
        "pdays_lt_999": "前回接触からの日数 < 999",
        "pdays_eq_999": "前回接触からの日数 の有無",
        "previous": "前回接触回数",
        "emp.var.rate": "雇用変化率",
        "cons.price.idx": "消費者物価指数",
        "cons.conf.idx": "消費者信頼感指数",
        "euribor3m": "3 か月 Euribor",
        "nr.employed": "就業者数",
        "education_encoded": "教育水準",
        "job": "職業",
        "marital": "婚姻状況",
        "default": "デフォルト履歴",
        "housing": "住宅ローン",
        "loan": "個人ローン",
        "contact": "連絡手段",
        "month": "月",
        "day_of_week": "曜日",
        "poutcome": "前回結果",
    },
    "en": {
        "age": "Age",
        "campaign": "Campaign",
        "pdays_lt_999": "pdays < 999",
        "pdays_eq_999": "Presence of previous contact days",
        "previous": "Previous",
        "emp.var.rate": "Employment variation rate",
        "cons.price.idx": "Consumer price index",
        "cons.conf.idx": "Consumer confidence index",
        "euribor3m": "Euribor 3m",
        "nr.employed": "Number employed",
        "education_encoded": "Education",
        "job": "Job",
        "marital": "Marital",
        "default": "Default history",
        "housing": "Housing",
        "loan": "Loan",
        "contact": "Contact",
        "month": "Month",
        "day_of_week": "Day of week",
        "poutcome": "Previous outcome",
    },
}


SUBSCALE_LABELS = {
    "ja": {
        "age": "年齢",
        "campaign": "接触回数",
        "pdays": "前回接触からの日数",
        "previous": "前回接触回数",
        "emp.var.rate": "雇用変化率",
        "cons.price.idx": "消費者物価指数",
        "cons.conf.idx": "消費者信頼感指数",
        "euribor3m": "3 か月 Euribor",
        "nr.employed": "就業者数",
        "education_encoded": "教育水準",
        "job": "職業",
        "marital": "婚姻状況",
        "default": "デフォルト履歴",
        "housing": "住宅ローン",
        "loan": "個人ローン",
        "contact": "連絡手段",
        "month": "月",
        "day_of_week": "曜日",
        "poutcome": "前回結果",
    },
    "en": {
        "age": "Age",
        "campaign": "Campaign",
        "pdays": "pdays",
        "previous": "Previous",
        "emp.var.rate": "Employment variation rate",
        "cons.price.idx": "Consumer price index",
        "cons.conf.idx": "Consumer confidence index",
        "euribor3m": "Euribor 3m",
        "nr.employed": "Number employed",
        "education_encoded": "Education",
        "job": "Job",
        "marital": "Marital",
        "default": "Default history",
        "housing": "Housing",
        "loan": "Loan",
        "contact": "Contact",
        "month": "Month",
        "day_of_week": "Day of week",
        "poutcome": "Previous outcome",
    },
}


def feature_label(lang: str, name: str) -> str:
    if lang == "ja" and name == "pdays":
        return "前回接触からの日数"
    labels = FEATURE_LABELS.get(lang, FEATURE_LABELS["en"])
    if name in labels:
        return labels[name]
    for prefix in ("day_of_week_", "of_week_"):
        if name.startswith(prefix):
            suffix = name.split(prefix, 1)[1]
            day_labels = {
                "ja": {
                    "mon": "月曜日",
                    "tue": "火曜日",
                    "wed": "水曜日",
                    "thu": "木曜日",
                    "fri": "金曜日",
                },
                "en": {
                    "mon": "Monday",
                    "tue": "Tuesday",
                    "wed": "Wednesday",
                    "thu": "Thursday",
                    "fri": "Friday",
                },
            }
            day_text = day_labels.get(lang, day_labels["en"]).get(suffix, suffix)
            return day_text if lang == "ja" else f"day_of_week_{suffix}"
    return name


def subscale_label(lang: str, name: str) -> str:
    labels = SUBSCALE_LABELS.get(lang, SUBSCALE_LABELS["en"])
    if name in labels:
        return labels[name]
    for prefix in ("day_of_week_", "of_week_"):
        if name.startswith(prefix):
            suffix = name.split(prefix, 1)[1]
            day_labels = {
                "ja": {
                    "mon": "月曜日",
                    "tue": "火曜日",
                    "wed": "水曜日",
                    "thu": "木曜日",
                    "fri": "金曜日",
                },
                "en": {
                    "mon": "Monday",
                    "tue": "Tuesday",
                    "wed": "Wednesday",
                    "thu": "Thursday",
                    "fri": "Friday",
                },
            }
            day_text = day_labels.get(lang, day_labels["en"]).get(suffix, suffix)
            return day_text if lang == "ja" else f"day_of_week_{suffix}"
    return name


CATEGORICAL_OPTION_LABELS = {
    "ja": {
        "job": {
            "admin.": "管理職",
            "blue-collar": "ブルーカラー",
            "entrepreneur": "起業家",
            "housemaid": "家政婦",
            "management": "管理職",
            "retired": "退職者",
            "self-employed": "自営業",
            "services": "サービス業",
            "student": "学生",
            "technician": "技術職",
            "unemployed": "無職",
            "unknown": "不明",
        },
        "marital": {
            "divorced": "離婚",
            "married": "既婚",
            "single": "独身",
            "unknown": "不明",
        },
        "default": {"no": "なし", "yes": "あり", "unknown": "不明"},
        "housing": {"no": "なし", "yes": "あり", "unknown": "不明"},
        "loan": {"no": "なし", "yes": "あり", "unknown": "不明"},
        "contact": {"cellular": "携帯", "telephone": "固定電話"},
        "month": {
            "jan": "1月",
            "feb": "2月",
            "mar": "3月",
            "apr": "4月",
            "may": "5月",
            "jun": "6月",
            "jul": "7月",
            "aug": "8月",
            "sep": "9月",
            "oct": "10月",
            "nov": "11月",
            "dec": "12月",
        },
        "day_of_week": {
            "mon": "月曜日",
            "tue": "火曜日",
            "wed": "水曜日",
            "thu": "木曜日",
            "fri": "金曜日",
            "sat": "土曜日",
            "sun": "日曜日",
        },
        "poutcome": {
            "failure": "失敗",
            "nonexistent": "なし",
            "success": "成功",
        },
    },
    "en": {
        "job": {
            "admin.": "admin.",
            "blue-collar": "blue-collar",
            "entrepreneur": "entrepreneur",
            "housemaid": "housemaid",
            "management": "management",
            "retired": "retired",
            "self-employed": "self-employed",
            "services": "services",
            "student": "student",
            "technician": "technician",
            "unemployed": "unemployed",
            "unknown": "unknown",
        },
        "marital": {
            "divorced": "divorced",
            "married": "married",
            "single": "single",
            "unknown": "unknown",
        },
        "default": {"no": "no", "yes": "yes", "unknown": "unknown"},
        "housing": {"no": "no", "yes": "yes", "unknown": "unknown"},
        "loan": {"no": "no", "yes": "yes", "unknown": "unknown"},
        "contact": {"cellular": "cellular", "telephone": "telephone"},
        "month": {
            "jan": "jan",
            "feb": "feb",
            "mar": "mar",
            "apr": "apr",
            "may": "may",
            "jun": "jun",
            "jul": "jul",
            "aug": "aug",
            "sep": "sep",
            "oct": "oct",
            "nov": "nov",
            "dec": "dec",
        },
        "day_of_week": {
            "mon": "mon",
            "tue": "tue",
            "wed": "wed",
            "thu": "thu",
            "fri": "fri",
            "sat": "sat",
            "sun": "sun",
        },
        "poutcome": {
            "failure": "failure",
            "nonexistent": "nonexistent",
            "success": "success",
        },
    },
}


def categorical_option_label(lang: str, feature: str, option: str) -> str:
    normalized = _normalize_categorical_option(feature, option)
    return CATEGORICAL_OPTION_LABELS.get(lang, CATEGORICAL_OPTION_LABELS["en"]).get(feature, {}).get(normalized, normalized)


def _normalize_categorical_option(feature: str, option: str) -> str:
    if option.startswith(feature + "_"):
        return option[len(feature) + 1 :]
    if feature == "day_of_week" and option.startswith("of_week_"):
        return option.split("of_week_", 1)[1]
    return option


NUMERIC_GROUPS = [
    "age",
    "campaign",
    "pdays_eq_999",
    "pdays_lt_999",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
    "education_encoded",
]

CATEGORICAL_GROUPS = [
    "job",
    "marital",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]


@dataclass(frozen=True)
class FeatureSpec:
    numeric: list[dict[str, Any]]
    categorical: list[dict[str, Any]]
    model_columns: list[str]
    raw_defaults: dict[str, Any]
    numeric_thresholds: dict[str, list[float]]
    categorical_map: dict[str, list[str]]
    feature_to_model_columns: dict[str, list[str]]


@dataclass
class ModelBundle:
    fold: int
    ap: float | None
    steps: dict[str, int]
    model: Any
    model_header: list[str]
    raw_df: pd.DataFrame
    raw_header: list[str]
    feature_spec: FeatureSpec
    notebook_ns: dict[str, Any]


def _set_runtime_env() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))


def _read_notebook_namespace() -> dict[str, Any]:
    _set_runtime_env()
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    ns: dict[str, Any] = {"__name__": "_notebook"}
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if src.strip():
            exec(compile(src, str(NOTEBOOK_PATH), "exec"), ns)
    return ns


def _groupby_columns(X_sub: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return pd.DataFrame(X_sub).T.groupby(labels).sum().T.values


def _patch_notebook_namespace(ns: dict[str, Any]) -> None:
    def aggregate_features_hierarchically(X_sub, y, n_steps_limit, use_weighted_ward=None):
        num_attributes = X_sub.shape[1]
        if num_attributes < 2:
            return X_sub, None

        y_01 = (y == 1).astype(int)
        if use_weighted_ward is None:
            use_weighted_ward = ns["USE_WEIGHTED_WARD"]

        if use_weighted_ward:
            Z = ns["compute_custom_linkage"](X_sub, y_01)
        else:
            rates = []
            for i in range(num_attributes):
                col_data = X_sub[:, i]
                pos = y_01[col_data == 1].sum()
                count = col_data.sum()
                rate = (pos + 1.0) / (count + 2.0)
                rates.append([rate])
            Z = ns["linkage"](rates, method="ward", metric="euclidean")

        num_clusters = max(1, num_attributes - n_steps_limit)
        cluster_labels = ns["fcluster"](Z, t=num_clusters, criterion="maxclust")
        X_agg = _groupby_columns(X_sub, cluster_labels)
        return X_agg, cluster_labels

    def _build_subscale_scores(self, X):
        scores_list = []
        subscale_start = 0
        for i, num_attr in enumerate(self.subscale_num_attributes):
            X_sub = X[:, subscale_start : subscale_start + num_attr]
            if hasattr(self, "agg_rules_") and i in self.agg_rules_:
                X_sub = _groupby_columns(X_sub, self.agg_rules_[i])
            prob_1 = self.subscale_clfs_[i].predict_proba(X_sub)[:, 1].reshape(-1, 1)
            scores_list.append(prob_1)
            subscale_start += num_attr
        return np.hstack(scores_list)

    ns["aggregate_features_hierarchically"] = aggregate_features_hierarchically
    ns["TwoLayerConstrainedLogisticRegression"]._build_subscale_scores = _build_subscale_scores  # type: ignore[attr-defined]

    def load_fixed_graph_payload(fold_id, base_dir='clu'):
        payload = {'graph_info': {}, 'graph_lambda': ns["load_graph_lambda_schedule"]().get(fold_id, 0.1)}
        edge_dir_candidates = [
            ROOT / "new_bank" / "graph_adaptive_plots" / f"graph_{fold_id}_adaptive",
            ROOT.parent / "new_bank" / "graph_adaptive_plots" / f"graph_{fold_id}_adaptive",
            Path("new_bank") / "graph_adaptive_plots" / f"graph_{fold_id}_adaptive",
            Path("..") / "new_bank" / "graph_adaptive_plots" / f"graph_{fold_id}_adaptive",
        ]
        for edge_dir in edge_dir_candidates:
            if not edge_dir.is_dir():
                continue
            for fname in edge_dir.iterdir():
                name = fname.name
                if not name.startswith(f"fold{fold_id}_") or not name.endswith("_graph_edges.txt"):
                    continue
                subscale = name[len(f"fold{fold_id}_"):-len("_graph_edges.txt")]
                payload["graph_info"][subscale] = ns["_parse_saved_graph_edges_txt"](str(fname))
            if payload["graph_info"]:
                return payload

        legacy_candidates = [
            Path(base_dir) / f"fold{fold_id}.txt",
            Path("new_bank") / base_dir / f"fold{fold_id}.txt",
            Path("./new_bank") / base_dir / f"fold{fold_id}.txt",
            ROOT / "new_bank" / base_dir / f"fold{fold_id}.txt",
        ]
        for path in legacy_candidates:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as f:
                legacy_payload = {"graph_info": {}}
                for raw_line in f:
                    line = raw_line.strip()
                    if not line or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    if key == "graph_lambda":
                        legacy_payload["graph_lambda"] = float(value)
                    elif key == "graph_info":
                        legacy_payload["graph_info"] = json.loads(value)
            if "graph_lambda" in legacy_payload and "graph_info" in legacy_payload:
                return legacy_payload

        raise FileNotFoundError(f"Could not find fixed graph payload for fold {fold_id}.")

    ns["load_fixed_graph_payload"] = load_fixed_graph_payload


def _safe_load_and_preprocess(filepath: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)
    data = np.genfromtxt(filepath, delimiter=",", skip_header=1)
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    y[y == 0] = -1
    header = Path(filepath).read_text(encoding="utf-8").splitlines()[0].split(",")
    return X, y, header


def _summary_candidates() -> list[Path]:
    return [path for path in SUMMARY_CANDIDATES if path.exists()]


def _parse_summary_file() -> tuple[dict[int, dict[str, int]], dict[int, float], int]:
    steps_by_fold: dict[int, dict[str, int]] = {}
    ap_by_fold: dict[int, float] = {}
    summary_candidates = _summary_candidates()

    # base_cluster.ipynb は fold ごとの結果を AA_base_cluster/fold*.txt に保存する。
    fold_result_dir = ROOT / "new_bank" / "base_graph_7_adaptive"
    fold_result_candidates = sorted(fold_result_dir.glob("fold*.txt"))

    for path in fold_result_candidates:
        try:
            payload = _load_fold_result_txt(path)
        except Exception:
            payload = None
        if not payload:
            continue
        fold = int(payload.get("fold", int(path.stem.replace("fold", ""))))
        metrics = payload.get("metrics", {})
        ap = metrics.get("ap")
        if ap is not None:
            ap_by_fold[fold] = float(ap)

    # 旧形式の要約ファイルが残っていれば AP の補完に使う。
    final_pattern = re.compile(r"Fold (\d+): n_steps=(\{.*\}), graph_lambda=([0-9.]+) \(Val AP: ([0-9.]+)\)")
    progress_fold_pattern = re.compile(r"^=== Fold (\d+) ===$")
    progress_steps_pattern = re.compile(r"^best_steps: (\{.*\})$")
    progress_ap_pattern = re.compile(r"^metrics: .*?ap=([0-9.]+)")

    for path in summary_candidates:
        lines = path.read_text(encoding="utf-8").splitlines()
        current_fold: int | None = None
        pending_steps: dict[str, int] | None = None
        pending_ap: float | None = None

        def finalize_pending() -> None:
            nonlocal current_fold, pending_steps, pending_ap
            if current_fold is None:
                return
            if pending_steps is not None:
                steps_by_fold[current_fold] = dict(pending_steps)
            if pending_ap is not None and current_fold not in ap_by_fold:
                ap_by_fold[current_fold] = pending_ap
            current_fold = None
            pending_steps = None
            pending_ap = None

        for raw_line in lines:
            line = raw_line.strip()
            final_match = final_pattern.match(line)
            if final_match:
                fold = int(final_match.group(1))
                steps = ast.literal_eval(final_match.group(2))
                ap = float(final_match.group(4))
                steps_by_fold[fold] = {str(k): int(v) for k, v in steps.items()}
                ap_by_fold.setdefault(fold, ap)
                continue

            progress_match = progress_fold_pattern.match(line)
            if progress_match:
                finalize_pending()
                current_fold = int(progress_match.group(1))
                pending_steps = None
                pending_ap = None
                continue

            if current_fold is None:
                continue

            steps_match = progress_steps_pattern.match(line)
            if steps_match:
                steps = ast.literal_eval(steps_match.group(1))
                pending_steps = {str(k): int(v) for k, v in steps.items()}
                continue

            ap_match = progress_ap_pattern.match(line)
            if ap_match:
                pending_ap = float(ap_match.group(1))

        finalize_pending()

    best_fold = 0
    if ap_by_fold:
        best_fold = max(ap_by_fold.items(), key=lambda item: item[1])[0]
    return steps_by_fold, ap_by_fold, best_fold


def _build_feature_spec(raw_df: pd.DataFrame, model_header: list[str]) -> FeatureSpec:
    feature_cols = [c for c in model_header if c != "y"]
    raw_defaults: dict[str, Any] = {}
    numeric_thresholds: dict[str, list[float]] = {}
    categorical_map: dict[str, list[str]] = {}
    feature_to_model_columns: dict[str, list[str]] = {}

    numeric_specs: list[dict[str, Any]] = []
    for name in NUMERIC_GROUPS:
        if name not in raw_df.columns:
            continue

        if name == "pdays_eq_999":
            raw_defaults[name] = int(raw_df[name].mode(dropna=True).iloc[0])
            cols = [c for c in feature_cols if c == "pdays_eq_999"]
            feature_to_model_columns[name] = cols
            numeric_specs.append({"name": name, "kind": "flag", "default": raw_defaults[name], "options": [0, 1]})
            continue

        values = pd.to_numeric(raw_df[name], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size == 0:
            values = np.array([0.0])

        default = float(np.nanmedian(values))
        raw_defaults[name] = default
        cols = [c for c in feature_cols if c.startswith(name + "_bin_")]
        feature_to_model_columns[name] = cols
        n_bins = max(1, len(cols))

        if n_bins > 1:
            qs = np.linspace(0, 1, n_bins + 1)[1:-1]
            ths = np.unique(np.quantile(values, qs).astype(float)).tolist()
            if len(ths) < n_bins - 1:
                pad = [ths[-1] if ths else default] * (n_bins - 1 - len(ths))
                ths = ths + pad
            numeric_thresholds[name] = ths[: max(0, n_bins - 1)]
        else:
            numeric_thresholds[name] = []

        numeric_specs.append(
            {
                "name": name,
                "kind": "numeric",
                "min": float(np.nanmin(values)),
                "max": float(np.nanmax(values)),
                "step": 0.1 if name in {"emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m"} else 1.0,
                "default": default,
            }
        )

    categorical_specs: list[dict[str, Any]] = []
    for name in CATEGORICAL_GROUPS:
        cols = [c for c in feature_cols if c.startswith(name + "_")]
        if not cols:
            continue
        categorical_map[name] = cols
        default = _normalize_categorical_option(name, cols[0])
        raw_defaults[name] = default
        categorical_specs.append(
            {
                "name": name,
                "options": [_normalize_categorical_option(name, c) for c in cols],
                "default": default,
            }
        )

    return FeatureSpec(
        numeric=numeric_specs,
        categorical=categorical_specs,
        model_columns=feature_cols,
        raw_defaults=raw_defaults,
        numeric_thresholds=numeric_thresholds,
        categorical_map=categorical_map,
        feature_to_model_columns=feature_to_model_columns,
    )


def _risk_color(prob: float) -> str:
    prob = max(0.0, min(1.0, float(prob)))
    # 低い値を灰色、高い値を赤にする。
    low = (0xb8, 0xb8, 0xb8)
    high = (0xe8, 0x4b, 0x4b)
    r = int(round(low[0] * (1.0 - prob) + high[0] * prob))
    g = int(round(low[1] * (1.0 - prob) + high[1] * prob))
    b = int(round(low[2] * (1.0 - prob) + high[2] * prob))
    return f"#{r:02x}{g:02x}{b:02x}"


def _metric_color(value: float, scale: float | None = None) -> str:
    magnitude = abs(float(value))
    if scale is not None and scale > 0:
        magnitude = min(1.0, magnitude / scale)
    else:
        magnitude = min(1.0, magnitude)
    return _risk_color(magnitude)


def _text_color(bg_hex: str) -> str:
    r = int(bg_hex[1:3], 16)
    g = int(bg_hex[3:5], 16)
    b = int(bg_hex[5:7], 16)
    return "#111111" if (0.299 * r + 0.587 * g + 0.114 * b) > 160 else "#ffffff"


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value)}"
    return f"{float(value):.4f}".rstrip("0").rstrip(".")


def _default_state(bundle: ModelBundle) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for spec in bundle.feature_spec.numeric:
        state[spec["name"]] = spec["default"]
    for spec in bundle.feature_spec.categorical:
        state[spec["name"]] = spec["default"]
    state["lang"] = DEFAULT_LANG
    return state


def _encode_raw_row(bundle: ModelBundle, form_data: dict[str, str]) -> tuple[np.ndarray, dict[str, Any]]:
    row = {c: 0.0 for c in bundle.model_header if c != "y"}
    ui_state: dict[str, Any] = {}
    pdays_eq_999_value: int | None = None
    pdays_lt_999_value: float | None = None

    for spec in bundle.feature_spec.numeric:
        name = spec["name"]
        if spec["kind"] == "flag":
            value = int(form_data.get(name, spec["default"]))
            ui_state[name] = value
            if name == "pdays_eq_999":
                pdays_eq_999_value = value
            col = bundle.feature_spec.feature_to_model_columns[name][0]
            row[col] = float(value)
            continue

        raw_value = float(form_data.get(name, spec["default"]))
        ui_state[name] = raw_value
        if name == "pdays_lt_999":
            pdays_lt_999_value = raw_value
        thresholds = bundle.feature_spec.numeric_thresholds.get(name, [])
        bin_idx = int(np.digitize([raw_value], thresholds, right=False)[0])
        cols = bundle.feature_spec.feature_to_model_columns[name]
        if cols:
            bin_idx = max(0, min(bin_idx, len(cols) - 1))
            row[cols[bin_idx]] = 1.0

    pdays_lt_col = bundle.feature_spec.feature_to_model_columns.get("pdays_lt_999", [])
    pdays_eq_col = bundle.feature_spec.feature_to_model_columns.get("pdays_eq_999", [])
    if pdays_eq_999_value == 1:
        if pdays_lt_col:
            row[pdays_lt_col[0]] = 0.0
        ui_state["pdays_lt_999"] = 0.0
    elif pdays_lt_999_value is not None and pdays_lt_999_value > 0:
        if pdays_eq_col:
            row[pdays_eq_col[0]] = 0.0
        ui_state["pdays_eq_999"] = 0

    for spec in bundle.feature_spec.categorical:
        name = spec["name"]
        selected = form_data.get(name, spec["default"])
        ui_state[name] = selected
        cols = bundle.feature_spec.categorical_map[name]
        target = f"{name}_{selected}"
        if target in row:
            row[target] = 1.0
        else:
            # fallback to the first available option
            row[cols[0]] = 1.0

    encoded = np.array([[row[c] for c in bundle.model_header if c != "y"]], dtype=float)
    return encoded, ui_state


def _load_raw_defaults() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH)
    df.columns = df.columns.str.strip()
    return df


@lru_cache(maxsize=4)
def _build_bundle(fold: int | None = None) -> ModelBundle:
    ns = _read_notebook_namespace()
    _patch_notebook_namespace(ns)
    ns["load_and_preprocess"] = _safe_load_and_preprocess

    steps_by_fold, ap_by_fold, best_fold = _parse_summary_file()
    if fold is None or fold not in ap_by_fold:
        fold = best_fold

    train_path = MODEL_TRAIN_TEMPLATE.with_name(MODEL_TRAIN_TEMPLATE.name.format(fold=fold))
    X, y, header = ns["load_and_preprocess"](str(train_path))
    if TRAIN_RATIO < 1.0:
        X, _, y, _ = train_test_split(
            X,
            y,
            train_size=TRAIN_RATIO,
            random_state=42,
            stratify=y,
        )
    raw_df = _load_raw_defaults()
    feature_spec = _build_feature_spec(raw_df, header)

    calculated_attrs = ns["calculate_subscale_attributes"](header)
    params = ns["get_optimized_parameters"](calculated_attrs, fold)
    steps = steps_by_fold.get(fold, {})

    model = ns["TwoLayerConstrainedLogisticRegression"](
        subscale_num_attributes=calculated_attrs,
        subscale_params_list=params,
        graph_lambda=0.0,
    )
    model.fit(X, y, feature_names=header[:-1], fold_id=fold)

    return ModelBundle(
        fold=fold,
        ap=ap_by_fold.get(fold),
        steps=steps,
        model=model,
        model_header=header,
        raw_df=raw_df,
        raw_header=list(raw_df.columns),
        feature_spec=feature_spec,
        notebook_ns=ns,
    )


def _blend_hex(base_hex: str, target_hex: str, ratio: float) -> str:
    ratio = max(0.0, min(1.0, ratio))
    br = int(base_hex[1:3], 16)
    bg = int(base_hex[3:5], 16)
    bb = int(base_hex[5:7], 16)
    tr = int(target_hex[1:3], 16)
    tg = int(target_hex[3:5], 16)
    tb = int(target_hex[5:7], 16)
    r = int(round(br * (1 - ratio) + tr * ratio))
    g = int(round(bg * (1 - ratio) + tg * ratio))
    b = int(round(bb * (1 - ratio) + tb * ratio))
    return f"#{r:02x}{g:02x}{b:02x}"


def _predict(bundle: ModelBundle, encoded: np.ndarray, ui_state: dict[str, Any], lang: str) -> dict[str, Any]:
    model = bundle.model
    proba = float(model.predict_proba(encoded)[0, 1])
    subscale_scores = model._build_subscale_scores(encoded)[0]
    final_weights = list(getattr(getattr(model, "final_clf_", None), "w_", []))
    weight_scale = max((abs(float(w)) for w in final_weights if w is not None), default=0.0)
    combined_scale = 0.0
    for idx, score in enumerate(subscale_scores):
        if idx < len(final_weights) and final_weights[idx] is not None:
            combined_scale = max(combined_scale, abs(float(score) * float(final_weights[idx])))
    if weight_scale <= 0:
        weight_scale = 1.0
    if combined_scale <= 0:
        combined_scale = 1.0

    raw_rows = []
    for spec in bundle.feature_spec.numeric:
        name = spec["name"]
        value = ui_state.get(name, spec["default"])
        if spec["kind"] == "flag":
            raw_rows.append({"name": name, "value": int(value), "kind": "flag"})
        else:
            raw_rows.append({"name": name, "value": float(value), "kind": "numeric"})
    for spec in bundle.feature_spec.categorical:
        name = spec["name"]
        value = ui_state.get(name, spec["default"])
        raw_rows.append({"name": name, "value": value, "kind": "categorical"})

    # base_cluster.ipynb の全サブスケール順をそのまま使い、選択済みサブスケール表示に落とさない。
    selected_names = ns_subscale_names(bundle.notebook_ns)
    subscale_rows = []
    for idx, name in enumerate(selected_names):
        score = float(subscale_scores[idx])
        weight = float(final_weights[idx]) if idx < len(final_weights) else None
        score_color = _risk_color(score)
        weight_color = _metric_color(weight if weight is not None else 0.0, weight_scale)
        combined_value = score * weight if weight is not None else 0.0
        combined_color = _metric_color(combined_value, combined_scale)
        subscale_rows.append(
            {
                "name": name,
                "score": score,
                "weight": weight,
                "score_color": score_color,
                "score_text_color": _text_color(score_color),
                "weight_color": weight_color,
                "weight_text_color": "#111111",
                "combined_color": combined_color,
                "combined_text_color": _text_color(combined_color),
            }
        )

    return {
        "probability": proba,
        "risk_label": tr(lang, "high_risk") if proba >= 0.5 else tr(lang, "low_risk"),
        "color": _risk_color(proba),
        "text_color": _text_color(_risk_color(proba)),
        "raw_rows": raw_rows,
        "subscale_rows": subscale_rows,
        "fold_label": f"Fold {bundle.fold}",
        "fold_ap": bundle.ap,
    }


def ns_subscale_names(ns: dict[str, Any]) -> list[str]:
    return list(ns.get("subscale_names", []))


def _subscale_source_names(name: str) -> list[str]:
    if name == "pdays":
        return ["pdays_eq_999", "pdays_lt_999"]
    return [name]


def _raw_value_label(lang: str, name: str, value: Any, kind: str | None = None) -> str:
    if kind == "categorical":
        return categorical_option_label(lang, name, str(value))
    return str(value)


def _render_raw_cards(rows: list[dict[str, Any]], lang: str) -> str:
    cards = []
    for row in rows:
        value = row["value"]
        if isinstance(value, float):
            value = _format_number(value)
        display_value = _raw_value_label(lang, row["name"], value, row.get("kind"))
        cards.append(
            f"""
            <div class="feature-card">
              <div class="feature-name">{html_lib.escape(feature_label(lang, row['name']))}</div>
              <div class="feature-value">{html_lib.escape(str(display_value))}</div>
            </div>
            """
        )
    return "".join(cards)


def _render_subscale_cards(rows: list[dict[str, Any]], lang: str) -> str:
    cards = []
    for row in rows:
        weight_text = f"{row['weight']:.3f}" if row.get("weight") is not None else "N/A"
        cards.append(
            f"""
            <div class="subscale-card">
              <div class="subscale-name">{html_lib.escape(subscale_label(lang, row['name']))}</div>
              <div class="subscale-metric-head">
                <span>{html_lib.escape(tr(lang, "subscale_probability"))}</span>
                <span>{html_lib.escape(tr(lang, "subscale_weight"))}</span>
              </div>
              <div class="subscale-split">
                <div class="subscale-half subscale-half-score" style="background:{row['score_color']};color:{row['score_text_color']}">
                  <div class="half-value">{row['score'] * 100:.1f}%</div>
                </div>
                <div class="subscale-half subscale-half-weight" style="background:{row['weight_color']};color:{row['weight_text_color']}">
                  <div class="half-value">{html_lib.escape(weight_text)}</div>
                </div>
              </div>
            </div>
            """
        )
    return "".join(cards)


def _render_probability_legend(lang: str) -> str:
    return f"""
    <div class="output-legend">
      <div class="legend-title">{html_lib.escape(tr(lang, "legend_title"))}</div>
      <div class="legend-bar" aria-hidden="true">
        <span class="legend-stop legend-low"></span>
        <span class="legend-stop legend-mid"></span>
        <span class="legend-stop legend-high"></span>
      </div>
      <div class="legend-labels">
        <span>{html_lib.escape(tr(lang, "legend_low"))}</span>
        <span>{html_lib.escape(tr(lang, "legend_mid"))}</span>
        <span>{html_lib.escape(tr(lang, "legend_high"))}</span>
      </div>
    </div>
    """


def _render_network_diagram(bundle: ModelBundle, result: dict[str, Any], lang: str) -> str:
    original_rows = result["raw_rows"]
    subscale_rows = result["subscale_rows"]
    if not original_rows or not subscale_rows:
        return ""

    row_gap = 72
    top_margin = 70
    left_x = 60
    middle_x = 460
    right_x = 920
    original_width = 330
    subscale_width = 250
    output_width = 220
    original_height = 36
    subscale_height = 64
    output_height = 76
    pdays_row_offset = -12
    later_rows_offset = 10

    def row_y(idx: int) -> float:
        return top_margin + idx * row_gap + (pdays_row_offset if idx == 0 else later_rows_offset)

    def original_node_top(base_y: float, height: float) -> float:
        # raw 側は subscale 側と中心位置を合わせる。
        return base_y + (subscale_height - height) / 2.0

    pdays_sources = _subscale_source_names("pdays")
    original_map = {str(row["name"]): row for row in original_rows}
    original_has_pdays_pair = all(name in original_map for name in pdays_sources)

    ordered_original_rows: list[dict[str, Any]] = []
    if original_has_pdays_pair:
        ordered_original_rows.append({"name": "pdays", "kind": "group", "sources": pdays_sources})
        for row in original_rows:
            if str(row["name"]) in pdays_sources:
                continue
            ordered_original_rows.append(row)
    else:
        ordered_original_rows = list(original_rows)

    ordered_subscale_rows = sorted(
        subscale_rows,
        key=lambda row: 0 if str(row["name"]) == "pdays" else 1,
    )

    max_rows = max(len(ordered_original_rows), len(ordered_subscale_rows))
    stage_height = top_margin * 2 + (max_rows - 1) * row_gap + 130 + (later_rows_offset - pdays_row_offset)
    output_y = stage_height / 2 - output_height / 2
    output_cy = stage_height / 2

    original_html = []
    subscale_html = []
    lines = []
    marker_defs = []
    original_positions: dict[str, tuple[float, float]] = {}

    for idx, row in enumerate(ordered_original_rows):
        y = row_y(idx)
        if row.get("kind") == "group" and row.get("sources"):
            group_height = original_height * 2 + 10
            node_top = original_node_top(y, group_height)
            cy = node_top + group_height / 2
            for source_name in row["sources"]:
                original_positions[source_name] = (node_top, cy)
            source_bits = []
            for source_name in row["sources"]:
                source_row = original_map.get(source_name)
                if source_row is None:
                    continue
                value = source_row["value"]
                if isinstance(value, float):
                    value = _format_number(value)
                source_bits.append(
                    f'<div class="group-line"><span class="group-label">{html_lib.escape(feature_label(lang, source_name))}</span>'
                    f'<span class="group-value">{html_lib.escape(str(_raw_value_label(lang, source_name, value, source_row.get("kind"))))}</span></div>'
                )
            original_html.append(
                f"""
                <div class="diagram-node original-node group-node" style="left:{left_x}px;top:{node_top}px;width:{original_width}px;height:{group_height}px;">
                  <div class="node-title">{html_lib.escape(feature_label(lang, "pdays"))}</div>
                  <div class="group-lines">
                    {''.join(source_bits)}
                  </div>
                </div>
                """
            )
            continue

        node_top = original_node_top(y, original_height)
        cy = node_top + original_height / 2
        original_positions[str(row["name"])] = (node_top, cy)
        value = row["value"]
        if isinstance(value, float):
            value = _format_number(value)
        label = html_lib.escape(feature_label(lang, str(row["name"])))
        val = html_lib.escape(str(_raw_value_label(lang, str(row["name"]), value, row.get("kind"))))
        original_html.append(
            f"""
            <div class="diagram-node original-node" style="left:{left_x}px;top:{node_top}px;width:{original_width}px;height:{original_height}px;">
              <div class="node-title">{label}</div>
              <div class="node-value">{val}</div>
            </div>
            """
        )

    for idx, row in enumerate(ordered_subscale_rows):
        y = row_y(idx)
        cy = y + subscale_height / 2
        score_color = row["score_color"]
        combined_color = row["combined_color"]
        base_line = _blend_hex(score_color, "#ffffff", 0.45)
        weight = row.get("weight")
        weight_text = f"{weight:.3f}" if weight is not None else "N/A"
        subscale_html.append(
            f"""
            <div class="diagram-node subscale-node" style="left:{middle_x}px;top:{y}px;width:{subscale_width}px;height:{subscale_height}px;">
              <div class="node-title">{html_lib.escape(subscale_label(lang, str(row['name'])))}</div>
              <div class="subscale-split">
                <div class="subscale-half subscale-half-score" style="background:{score_color};color:{row['score_text_color']}">
                  <div class="half-value">{row['score'] * 100:.1f}%</div>
                </div>
                <div class="subscale-half subscale-half-weight" style="background:{row['weight_color']};color:{row['weight_text_color']}">
                  <div class="half-value">{html_lib.escape(weight_text)}</div>
                </div>
              </div>
            </div>
            """
        )

        source_names = _subscale_source_names(str(row["name"]))
        y2 = y + subscale_height / 2
        for source_idx, source_name in enumerate(source_names):
            if source_name not in original_positions:
                continue
            source_y, source_cy = original_positions[source_name]
            x1 = left_x + original_width
            x2 = middle_x
            lines.append(
                f'<line x1="{x1}" y1="{source_cy}" x2="{x2}" y2="{y2}" stroke="{base_line}" stroke-width="2.2" marker-end="url(#arrow-in-{idx}-{source_idx})" />'
            )
            marker_defs.append(
                f'''
                <marker id="arrow-in-{idx}-{source_idx}" markerWidth="8" markerHeight="8" refX="6" refY="3.5" orient="auto" markerUnits="strokeWidth">
                  <polygon points="0 0, 7 3.5, 0 7" fill="{base_line}" />
                </marker>
                '''
            )

        x3 = middle_x + subscale_width
        x4 = right_x
        y3 = y2
        y4 = output_cy
        lines.append(
            f'<line x1="{x3}" y1="{y3}" x2="{x4}" y2="{y4}" stroke="{combined_color}" stroke-width="2.8" marker-end="url(#arrow-out-{idx})" />'
        )
        marker_defs.append(
            f'''
            <marker id="arrow-out-{idx}" markerWidth="8" markerHeight="8" refX="6" refY="3.5" orient="auto" markerUnits="strokeWidth">
              <polygon points="0 0, 7 3.5, 0 7" fill="{combined_color}" />
            </marker>
            '''
        )

    output_prob = result["probability"]
    output_color = result["color"]
    output_html = f"""
    <div class="diagram-node output-node" style="left:{right_x}px;top:{output_y}px;width:{output_width}px;height:{output_height}px;background:{output_color};color:{result['text_color']};">
      <div class="node-title">{html_lib.escape(tr(lang, "output"))}</div>
      <div class="node-value">{html_lib.escape(tr(lang, "predicted_probability"))} {output_prob * 100:.1f}%</div>
    </div>
    """

    subscale_header_html = f"""
    <div class="network-col-label" style="left:{middle_x + subscale_width / 2}px;top:18px;width:{subscale_width}px;">
      <div style="display:flex;justify-content:space-between;gap:8px;width:100%;">
        <span style="margin-left:26px;">{html_lib.escape(tr(lang, "subscale_probability"))}</span>
        <span style="margin-right:28px;">{html_lib.escape(tr(lang, "subscale_weight"))}</span>
      </div>
    </div>
    """

    svg_width = right_x + output_width + 40
    svg_height = stage_height

    return f"""
    <section class="network-card">
      <div class="section-head">
        <h2>{html_lib.escape(tr(lang, "structure_title"))}</h2>
        <p>{html_lib.escape(tr(lang, "layer1_note"))}</p>
      </div>
      <div class="network-stage" style="height:{stage_height}px;">
        <div class="network-legend-anchor">
          {_render_probability_legend(lang)}
        </div>
        <svg class="network-links" width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}" preserveAspectRatio="none">
          <defs>
            {''.join(marker_defs)}
          </defs>
          {"".join(lines)}
        </svg>
        {subscale_header_html}
        {"".join(original_html)}
        {"".join(subscale_html)}
        {output_html}
      </div>
    </section>
    """


def render_page(bundle: ModelBundle, state: dict[str, Any], result: dict[str, Any] | None = None, error: str | None = None) -> str:
    lang = str(state.get("lang", DEFAULT_LANG))
    t = lambda key: tr(lang, key)
    ap_text = f"{bundle.ap:.5f}" if bundle.ap is not None else "N/A"
    numeric_controls = []
    categorical_controls = []

    for spec in bundle.feature_spec.numeric:
        name = spec["name"]
        if spec["kind"] == "flag":
            flag_options = [(0, "あり" if lang == "ja" else "present"), (1, "なし" if lang == "ja" else "absent")]
            numeric_controls.append(
                f"""
                <div class="control categorical-control">
                  <label for="{name}">{html_lib.escape(feature_label(lang, name))}</label>
                  <select id="{name}" name="{name}">
                    {''.join(
                        f'<option value="{value}" {"selected" if int(state.get(name, spec["default"])) == value else ""}>'
                        f'{value}: {html_lib.escape(label)}</option>'
                        for value, label in flag_options
                    )}
                  </select>
                </div>
                """
            )
            continue

        current = float(state.get(name, spec["default"]))
        numeric_controls.append(
            f"""
            <div class="control numeric-control">
              <div class="control-label-row">
                <label for="{name}_range">{html_lib.escape(feature_label(lang, name))}</label>
                <span class="value-pill" id="{name}_value">{_format_number(current)}</span>
              </div>
              <input type="range" id="{name}_range" name="{name}" min="{spec['min']}" max="{spec['max']}" step="{spec['step']}" value="{current}" />
              <div class="range-foot">
                <span>{_format_number(spec['min'])}</span>
                <input type="number" id="{name}_num" value="{current}" step="{spec['step']}" min="{spec['min']}" max="{spec['max']}" />
                <span>{_format_number(spec['max'])}</span>
              </div>
            </div>
            """
        )

    for spec in bundle.feature_spec.categorical:
        name = spec["name"]
        current = str(state.get(name, spec["default"]))
        opts = []
        for option in spec["options"]:
            selected = " selected" if option == current else ""
            display = categorical_option_label(lang, name, option)
            opts.append(
                f'<option value="{html_lib.escape(option)}"{selected}>{html_lib.escape(display)}</option>'
            )
        categorical_controls.append(
            f"""
            <div class="control categorical-control">
              <label for="{name}">{html_lib.escape(feature_label(lang, name))}</label>
              <select id="{name}" name="{name}">
                {''.join(opts)}
              </select>
            </div>
            """
        )

    lang_controls = []
    for code, label in [("ja", "日本語"), ("en", "English")]:
        selected = " selected" if lang == code else ""
        lang_controls.append(f'<option value="{code}"{selected}>{label}</option>')

    if result is not None:
        result_ap_text = f"{result['fold_ap']:.5f}" if result.get("fold_ap") is not None else "N/A"
        summary_html = f"""
          <div class="summary-card" style="background:{result['color']};color:{result['text_color']}">
            <div class="summary-kicker">{html_lib.escape(t("model_output"))}</div>
            <div class="summary-value">{result['probability'] * 100:.1f}%</div>
            <div class="summary-label">{html_lib.escape(result['risk_label'])}</div>
          </div>
        """
        raw_html = f"""
          <section class="result-section">
            <div class="section-head">
              <h2>{html_lib.escape(t("original_features"))}</h2>
              <p>{html_lib.escape(t("raw_desc"))}</p>
            </div>
            <div class="card-grid encoded-grid">
              {_render_raw_cards(result['raw_rows'], lang)}
            </div>
          </section>
        """
        subscale_html = f"""
          <section class="result-section">
            <div class="section-head">
              <h2>{html_lib.escape(t("subscale_features"))}</h2>
              <p>{html_lib.escape(t("subscale_desc"))}</p>
            </div>
            <div class="card-stack">
              {_render_subscale_cards(result['subscale_rows'], lang)}
            </div>
          </section>
        """
        output_html = f"""
          <section class="result-section">
            <div class="output-head">
              <div class="section-head">
                <h2>{html_lib.escape(t("output"))}</h2>
                <p>{html_lib.escape(t("output_desc"))}</p>
              </div>
            </div>
            <div class="output-panel">
              <div class="output-title">{html_lib.escape(t("predicted_probability"))}</div>
              <div class="output-bar">
                <div class="output-fill" style="width:{result['probability'] * 100:.1f}%;background:{result['color']}"></div>
              </div>
              <div class="output-readout">{html_lib.escape(t("predicted_probability"))} {result['probability'] * 100:.1f}%</div>
              <div class="output-caption">{html_lib.escape(result['risk_label'])}</div>
            </div>
          </section>
        """
        network_html = _render_network_diagram(bundle, result, lang)
    else:
        summary_html = f"""
          <div class="summary-card summary-card-muted">
            <div class="summary-kicker">{html_lib.escape(t("ready_label"))}</div>
            <div class="summary-value">0.0%</div>
            <div class="summary-label">{html_lib.escape(t("run_prompt"))}</div>
          </div>
        """
        raw_html = f"""
          <section class="result-section">
            <div class="section-head">
              <h2>{html_lib.escape(t("original_features"))}</h2>
              <p>{html_lib.escape(t("raw_desc"))}</p>
            </div>
          </section>
        """
        subscale_html = f"""
          <section class="result-section">
            <div class="section-head">
              <h2>{html_lib.escape(t("subscale_features"))}</h2>
              <p>{html_lib.escape(t("subscale_desc"))}</p>
            </div>
          </section>
        """
        output_html = f"""
          <section class="result-section">
            <div class="output-head">
              <div class="section-head">
                <h2>{html_lib.escape(t("output"))}</h2>
                <p>{html_lib.escape(t("output_desc"))}</p>
              </div>
            </div>
          </section>
        """
        network_html = ""

    error_html = ""
    if error:
        error_html = f'<div class="error-banner">{html_lib.escape(error)}</div>'

    numeric_script = []
    for spec in bundle.feature_spec.numeric:
        if spec["kind"] == "flag":
            continue
        name = spec["name"]
        numeric_script.append(
            f"""
            (function() {{
              const range = document.getElementById('{name}_range');
              const num = document.getElementById('{name}_num');
              const value = document.getElementById('{name}_value');
              if (!range || !num || !value) return;
              const sync = (v) => {{
                const clamped = Math.min(Math.max(parseFloat(v || range.value), parseFloat(range.min)), parseFloat(range.max));
                range.value = clamped;
                num.value = clamped;
                value.textContent = clamped.toFixed(clamped % 1 === 0 ? 0 : 2).replace(/\\.0+$/, '');
              }};
              range.addEventListener('input', () => sync(range.value));
              num.addEventListener('input', () => sync(num.value));
              sync(range.value);
            }})();
            """
        )

    numeric_script.append(
        """
        (function() {
          const eq999 = document.getElementById('pdays_eq_999');
          const lt999Range = document.getElementById('pdays_lt_999_range');
          const lt999Num = document.getElementById('pdays_lt_999_num');
          const lt999Control = lt999Range ? lt999Range.closest('.control') : null;
          if (!eq999 || !lt999Range || !lt999Num) return;
          const syncDisabledState = () => {
            const disabled = String(eq999.value) === '1';
            lt999Range.disabled = disabled;
            lt999Num.disabled = disabled;
            if (lt999Control) {
              lt999Control.classList.toggle('is-disabled', disabled);
            }
            if (disabled) {
              const zeroValue = String(lt999Range.min ?? '0');
              lt999Range.value = zeroValue;
              lt999Num.value = zeroValue;
            }
          };
          eq999.addEventListener('change', syncDisabledState);
          syncDisabledState();
        })();
        """
    )

    css = textwrap.dedent(
        """
        :root {
          --bg: #f6f2ea;
          --panel: rgba(255, 255, 255, 0.86);
          --text: #171717;
          --muted: #6c6258;
          --border: rgba(31, 26, 22, 0.12);
          --shadow: 0 18px 40px rgba(44, 29, 17, 0.10);
        }

        * { box-sizing: border-box; }
        body {
          margin: 0;
          color: var(--text);
          background:
            radial-gradient(circle at top left, rgba(230, 204, 170, 0.24), transparent 35%),
            radial-gradient(circle at 80% 0%, rgba(190, 219, 203, 0.24), transparent 38%),
            linear-gradient(180deg, #fbfaf7 0%, #f6f2ea 100%);
          font-family: "Avenir Next", "Helvetica Neue", Helvetica, Arial, sans-serif;
        }

        .shell {
          max-width: 1800px;
          margin: 0 auto;
          padding: 18px 20px 28px;
        }

        .hero {
          display: flex;
          align-items: flex-end;
          justify-content: space-between;
          gap: 24px;
          margin-bottom: 18px;
        }

        .hero h1 {
          margin: 0;
          font-family: Georgia, "Times New Roman", serif;
          font-size: 34px;
          font-weight: 600;
          letter-spacing: -0.02em;
        }

        .hero p {
          margin: 6px 0 0;
          color: var(--muted);
          max-width: none;
          line-height: 1.45;
          white-space: nowrap;
        }

        .model-tag {
          padding: 10px 14px;
          border: 1px solid var(--border);
          border-radius: 999px;
          background: var(--panel);
          box-shadow: var(--shadow);
          font-size: 13px;
          color: #40362b;
          white-space: nowrap;
        }

        .grid {
          display: grid;
          grid-template-columns: 440px minmax(0, 1fr);
          gap: 20px;
          align-items: start;
        }

        .panel {
          border: 1px solid var(--border);
          border-radius: 22px;
          background: var(--panel);
          box-shadow: var(--shadow);
          backdrop-filter: blur(10px);
        }

        .input-panel {
          position: sticky;
          top: 18px;
          padding: 18px 18px 16px;
          max-height: calc(100vh - 36px);
          overflow: auto;
        }

        .input-panel form {
          display: flex;
          flex-direction: column;
          gap: 18px;
        }

        .panel-title {
          margin: 0 0 14px;
          font-size: 18px;
          font-family: Georgia, "Times New Roman", serif;
        }

        .control-group {
          display: grid;
          gap: 14px;
        }

        .lang-control {
          margin-bottom: 4px;
        }

        .group-title {
          margin: 0;
          font-size: 15px;
          letter-spacing: 0.02em;
          text-transform: uppercase;
          color: #5e5347;
        }

        .control {
          padding: 12px 12px 10px;
          border: 1px solid rgba(31, 26, 22, 0.08);
          border-radius: 16px;
          background: rgba(255, 255, 255, 0.72);
        }

        .control label {
          display: block;
          margin-bottom: 8px;
          font-size: 12px;
          color: #65594b;
          text-transform: uppercase;
          letter-spacing: 0.03em;
        }

        .control-label-row {
          display: flex;
          justify-content: space-between;
          gap: 12px;
          align-items: center;
          margin-bottom: 8px;
        }

        .value-pill {
          font-family: "SFMono-Regular", ui-monospace, Menlo, Consolas, monospace;
          font-size: 12px;
          padding: 4px 8px;
          border-radius: 999px;
          background: rgba(0, 0, 0, 0.06);
        }

        .numeric-control input[type="range"] { width: 100%; }

        .range-foot {
          margin-top: 8px;
          display: grid;
          grid-template-columns: 40px 1fr 52px;
          gap: 8px;
          align-items: center;
          font-size: 11px;
          color: var(--muted);
        }

        .range-foot input[type="number"] {
          width: 100%;
          padding: 6px 8px;
          border-radius: 10px;
          border: 1px solid rgba(31, 26, 22, 0.12);
          background: #fff;
          font-family: "SFMono-Regular", ui-monospace, Menlo, Consolas, monospace;
          font-size: 12px;
        }

        .control.is-disabled {
          opacity: 0.55;
        }

        .control.is-disabled input,
        .control.is-disabled select {
          cursor: not-allowed;
        }

        select {
          width: 100%;
          padding: 9px 10px;
          border-radius: 12px;
          border: 1px solid rgba(31, 26, 22, 0.12);
          background: #fff;
          font-size: 13px;
        }

        .run-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 12px;
          margin-top: 4px;
        }

        .run-button {
          border: 0;
          border-radius: 14px;
          background: linear-gradient(135deg, #1f6fb2, #3c8dd7);
          color: #fff;
          font-weight: 700;
          padding: 13px 18px;
          font-size: 14px;
          box-shadow: 0 10px 24px rgba(31, 111, 178, 0.26);
          cursor: pointer;
        }

        .run-button:hover { filter: brightness(1.03); }

        .helper-text {
          color: var(--muted);
          font-size: 12px;
          line-height: 1.45;
        }

        .results-panel {
          padding: 18px;
          background: linear-gradient(180deg, rgba(255,255,255,0.9), rgba(255,255,255,0.72));
        }

        .summary-card {
          border-radius: 22px;
          padding: 18px 18px 16px;
          margin-bottom: 18px;
          box-shadow: var(--shadow);
        }

        .summary-card-muted {
          background: linear-gradient(135deg, #536569, #7a8b7f);
          color: #fff;
        }

        .summary-kicker {
          font-size: 12px;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          opacity: 0.86;
          margin-bottom: 8px;
        }

        .summary-value {
          font-family: Georgia, "Times New Roman", serif;
          font-size: 46px;
          font-weight: 700;
          letter-spacing: -0.03em;
          line-height: 1;
        }

        .summary-label {
          margin-top: 6px;
          font-size: 18px;
          font-weight: 700;
        }

        .summary-meta {
          margin-top: 8px;
          font-size: 12px;
          opacity: 0.85;
        }

        .error-banner {
          margin-bottom: 16px;
          padding: 12px 14px;
          border-radius: 14px;
          background: #f6d7d6;
          border: 1px solid #e5a9a6;
          color: #6e1f1f;
        }

        .result-grid {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 16px;
          align-items: start;
        }

        .network-card {
          margin-bottom: 18px;
          padding: 16px;
          border-radius: 20px;
          border: 1px solid var(--border);
          background: rgba(255, 255, 255, 0.78);
          box-shadow: 0 10px 28px rgba(41, 28, 16, 0.07);
          overflow: hidden;
        }

        .network-stage {
          position: relative;
          min-height: 520px;
          margin-top: 8px;
        }

        .network-legend-anchor {
          position: absolute;
          top: 8px;
          right: 8px;
          width: 260px;
          z-index: 4;
        }

        .network-links {
          position: absolute;
          inset: 0;
          z-index: 1;
          pointer-events: none;
        }

        .network-col-label {
          position: absolute;
          top: 0;
          transform: translateX(-50%);
          font-size: 12px;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: #6d6255;
          font-weight: 700;
          z-index: 3;
        }

        .diagram-node {
          position: absolute;
          z-index: 2;
          border-radius: 16px;
          border: 1px solid rgba(31, 26, 22, 0.10);
          display: flex;
          flex-direction: column;
          justify-content: center;
          padding: 8px 12px;
          box-shadow: 0 10px 20px rgba(0, 0, 0, 0.05);
        }

        .original-node {
          background: linear-gradient(180deg, #ffffff, #f7f5f1);
        }

        .group-node {
          justify-content: flex-start;
          gap: 4px;
          padding-top: 8px;
          padding-bottom: 8px;
        }

        .group-lines {
          display: flex;
          flex-direction: column;
          gap: 4px;
          margin-top: 2px;
        }

        .group-line {
          display: flex;
          justify-content: space-between;
          gap: 10px;
          font-size: 11px;
          line-height: 1.1;
        }

        .group-label {
          font-weight: 700;
        }

        .group-value {
          font-weight: 600;
          color: #5e5144;
          text-align: right;
          white-space: nowrap;
        }

        .subscale-node {
          background: linear-gradient(180deg, #fffdf8, #f8f4ee);
          box-shadow: 0 10px 20px rgba(0, 0, 0, 0.05), inset 0 0 0 1px rgba(255,255,255,0.18);
          gap: 4px;
          padding-top: 8px;
          padding-bottom: 8px;
          justify-content: flex-start;
          align-items: stretch;
        }

        .subscale-metric-head {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 0;
          font-size: 11px;
          font-weight: 800;
          letter-spacing: 0.03em;
          text-align: center;
          color: rgba(31, 26, 22, 0.72);
          padding: 0 2px;
        }

        .output-node {
          align-items: center;
          text-align: center;
          font-weight: 700;
        }

        .node-title {
          font-size: 10px;
          text-transform: uppercase;
          letter-spacing: 0.04em;
          line-height: 1.0;
          margin-bottom: 1px;
        }

        .node-value {
          margin-top: 4px;
          font-size: 18px;
          font-weight: 700;
          line-height: 1.2;
        }

        .subscale-split {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 0;
          margin-top: 0;
          overflow: hidden;
          border-radius: 12px;
          border: 1px solid rgba(31, 26, 22, 0.08);
        }

        .subscale-half {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          min-height: 28px;
          padding: 4px 5px 5px;
          text-align: center;
        }

        .subscale-half-score {
          border-right: 1px solid rgba(255, 255, 255, 0.32);
        }

        .half-value {
          margin-top: 2px;
          font-size: 12px;
          font-weight: 800;
          line-height: 1.0;
          font-variant-numeric: tabular-nums;
        }

        .result-section {
          min-width: 0;
          padding: 16px;
          border-radius: 20px;
          border: 1px solid var(--border);
          background: rgba(255, 255, 255, 0.78);
          box-shadow: 0 10px 28px rgba(41, 28, 16, 0.07);
        }

        .section-head h2 {
          margin: 0;
          font-family: Georgia, "Times New Roman", serif;
          font-size: 22px;
        }

        .section-head p {
          margin: 6px 0 14px;
          color: var(--muted);
          font-size: 13px;
          line-height: 1.4;
        }

        .output-head {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          gap: 16px;
          margin-bottom: 8px;
        }

        .output-legend {
          width: 100%;
          padding: 10px 12px;
          border-radius: 16px;
          background: rgba(255, 255, 255, 0.78);
          border: 1px solid rgba(31, 26, 22, 0.08);
          box-shadow: inset 0 0 0 1px rgba(255,255,255,0.35);
        }

        .legend-title {
          font-size: 11px;
          text-transform: uppercase;
          letter-spacing: 0.04em;
          color: #6d6255;
          margin-bottom: 8px;
          font-weight: 700;
        }

        .legend-bar {
          display: grid;
          grid-template-columns: 1fr 1fr 1fr;
          height: 12px;
          border-radius: 999px;
          overflow: hidden;
          margin-bottom: 8px;
          border: 1px solid rgba(31, 26, 22, 0.08);
        }

        .legend-stop { display: block; }
        .legend-low { background: #b8b8b8; }
        .legend-mid { background: #d19898; }
        .legend-high { background: #e84b4b; }

        .legend-labels {
          display: flex;
          justify-content: space-between;
          gap: 8px;
          font-size: 11px;
          color: var(--muted);
        }

        .encoded-grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 10px;
          max-height: 760px;
          overflow: auto;
          padding-right: 4px;
        }

        .feature-card {
          border-radius: 14px;
          padding: 10px 12px;
          background: linear-gradient(180deg, #ffffff, #f8f6f2);
          border: 1px solid rgba(31, 26, 22, 0.08);
        }

        .feature-name {
          font-size: 11px;
          text-transform: uppercase;
          letter-spacing: 0.04em;
          color: #6d6255;
        }

        .feature-value {
          margin-top: 6px;
          font-size: 20px;
          font-weight: 700;
          font-family: "SFMono-Regular", ui-monospace, Menlo, Consolas, monospace;
        }

        .card-stack {
          display: grid;
          gap: 10px;
        }

        .subscale-card {
          display: flex;
          flex-direction: column;
          gap: 8px;
          padding: 12px 14px;
          border-radius: 16px;
          background: linear-gradient(180deg, #ffffff, #f8f6f2);
          border: 1px solid rgba(31, 26, 22, 0.08);
          box-shadow: inset 0 0 0 1px rgba(255,255,255,0.18);
        }

        .subscale-name {
          font-size: 13px;
          font-weight: 700;
          text-transform: none;
          letter-spacing: 0.04em;
        }

        .subscale-weight {
          font-family: "SFMono-Regular", ui-monospace, Menlo, Consolas, monospace;
          font-size: 12px;
          font-weight: 600;
          margin-left: 6px;
          opacity: 0.9;
        }

        .output-panel {
          display: grid;
          gap: 10px;
        }

        .output-title {
          font-size: 14px;
          text-transform: uppercase;
          letter-spacing: 0.06em;
          color: #62584c;
        }

        .output-bar {
          height: 22px;
          background: rgba(0, 0, 0, 0.08);
          border-radius: 999px;
          overflow: hidden;
        }

        .output-fill {
          height: 100%;
          border-radius: inherit;
        }

        .output-readout {
          font-family: Georgia, "Times New Roman", serif;
          font-size: 38px;
          font-weight: 700;
          line-height: 1;
        }

        .output-caption {
          font-size: 16px;
          font-weight: 700;
        }

        .footnote {
          margin-top: 12px;
          font-size: 12px;
          color: var(--muted);
        }

        @media (max-width: 1400px) {
          .grid { grid-template-columns: 1fr; }
          .input-panel { position: static; max-height: none; }
          .result-grid { grid-template-columns: 1fr; }
          .output-head { flex-direction: column; }
          .output-legend { flex-basis: auto; width: 100%; }
          .hero p { white-space: normal; }
        }
        """
    ).strip()

    html = f"""<!doctype html>
<html lang="{html_lib.escape(lang)}">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html_lib.escape(t("title"))}</title>
  <style>{css}</style>
</head>
<body>
  <div class="shell">
    <div class="hero">
      <div>
        <h1>{html_lib.escape(t("hero_title"))}</h1>
        <p>{html_lib.escape(t("hero_desc"))}</p>
      </div>
      <div class="model-tag">
        Fold {bundle.fold}
      </div>
    </div>

    {error_html}
    {summary_html}

    <div class="grid">
      <section class="panel input-panel">
        <h2 class="panel-title">{html_lib.escape(t("input_panel"))}</h2>
        <form method="post" action="/">
          <div class="control lang-control">
            <label for="lang">{html_lib.escape(t("language"))}</label>
            <select id="lang" name="lang" onchange="this.form.submit()">
              {''.join(lang_controls)}
            </select>
          </div>
          <div class="control-group">
            <div>
              <p class="group-title">{html_lib.escape(t("numeric_features"))}</p>
              {"".join(numeric_controls)}
            </div>
            <div>
              <p class="group-title">{html_lib.escape(t("categorical_features"))}</p>
              {"".join(categorical_controls)}
            </div>
          </div>
          <div class="run-row">
            <button class="run-button" type="submit">{html_lib.escape(t("run_button"))}</button>
            <div class="helper-text">{html_lib.escape(t("helper_text"))}</div>
          </div>
        </form>
      </section>

      <section class="panel results-panel">
        {network_html}
        <div class="result-grid">
          {raw_html}
          {subscale_html}
          {output_html}
        </div>
        <div class="footnote">
          {html_lib.escape(t("layer1_note"))}
        </div>
      </section>
    </div>
  </div>

  <script>
  {"".join(numeric_script)}
  </script>
</body>
</html>
"""
    return html


class _Handler(BaseHTTPRequestHandler):
    bundle: ModelBundle | None = None

    def _send_html(self, content: str, status: int = 200) -> None:
        payload = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _parse_form(self) -> dict[str, str]:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        from urllib.parse import parse_qs

        parsed = parse_qs(body, keep_blank_values=True)
        return {k: v[-1] for k, v in parsed.items()}

    def do_GET(self) -> None:  # noqa: N802
        if self.path not in {"/", "/index.html"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        assert self.bundle is not None
        state = _default_state(self.bundle)
        encoded, ui_state = _encode_raw_row(self.bundle, state)
        lang = str(ui_state.get("lang", DEFAULT_LANG))
        ui_state["lang"] = lang
        result = _predict(self.bundle, encoded, ui_state, lang)
        self._send_html(render_page(self.bundle, ui_state, result=result))

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        assert self.bundle is not None
        form = self._parse_form()
        try:
            encoded, ui_state = _encode_raw_row(self.bundle, form)
            lang = str(form.get("lang", DEFAULT_LANG))
            ui_state["lang"] = lang
            result = _predict(self.bundle, encoded, ui_state, lang)
            page = render_page(self.bundle, ui_state, result=result)
            self._send_html(page)
        except Exception as exc:  # pragma: no cover
            self._send_html(render_page(self.bundle, form, error=str(exc)), status=HTTPStatus.BAD_REQUEST)

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def main() -> None:
    bundle = _build_bundle(None)
    _Handler.bundle = bundle
    ap_text = f"{bundle.ap:.5f}" if bundle.ap is not None else "N/A"
    ratio_text = f"{TRAIN_RATIO:.2%}"
    print(f"Serving notebook GUI on http://{HOST}:{PORT} (fold {bundle.fold}, ratio {ratio_text}, validation AP {ap_text})")
    server = ThreadingHTTPServer((HOST, PORT), _Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
