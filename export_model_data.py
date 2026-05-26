from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from new_bank.base_cluster_gui import (
    UI_TEXT,
    _build_bundle,
    categorical_option_label,
    feature_label,
    subscale_label,
)


def _to_float_list(values: Any) -> list[float]:
    if values is None:
        return []
    return [float(v) for v in list(values)]


def _model_to_dict(model: Any) -> dict[str, Any]:
    subscale_clfs = []
    for clf in getattr(model, "subscale_clfs_", []):
        subscale_clfs.append(
            {
                "weights": _to_float_list(getattr(clf, "w_", [])),
                "intercept": float(getattr(clf, "c_", 0.0)),
            }
        )

    final_clf = getattr(model, "final_clf_", None)
    final_payload = {
        "weights": _to_float_list(getattr(final_clf, "w_", [])) if final_clf is not None else [],
        "intercept": float(getattr(final_clf, "c_", 0.0)) if final_clf is not None else 0.0,
    }

    return {
        "subscale_num_attributes": [int(v) for v in getattr(model, "subscale_num_attributes", [])],
        "subscale_clfs": subscale_clfs,
        "final_clf": final_payload,
    }


def build_payload() -> dict[str, Any]:
    bundle = _build_bundle(None)
    lang_codes = sorted(UI_TEXT.keys())

    feature_labels: dict[str, dict[str, str]] = {lang: {} for lang in lang_codes}
    subscale_labels: dict[str, dict[str, str]] = {lang: {} for lang in lang_codes}
    categorical_labels: dict[str, dict[str, dict[str, str]]] = {lang: {} for lang in lang_codes}

    for spec in bundle.feature_spec.numeric:
        name = str(spec["name"])
        for lang in lang_codes:
            feature_labels[lang][name] = feature_label(lang, name)

    for spec in bundle.feature_spec.categorical:
        name = str(spec["name"])
        options = [str(option) for option in spec["options"]]
        for lang in lang_codes:
            feature_labels[lang][name] = feature_label(lang, name)
            categorical_labels[lang][name] = {
                option: categorical_option_label(lang, name, option) for option in options
            }

    for name in bundle.notebook_ns.get("subscale_names", []):
        for lang in lang_codes:
            subscale_labels[lang][str(name)] = subscale_label(lang, str(name))

    feature_spec = {
        "numeric": [],
        "categorical": [],
        "model_columns": list(bundle.feature_spec.model_columns),
        "raw_defaults": bundle.feature_spec.raw_defaults,
        "numeric_thresholds": bundle.feature_spec.numeric_thresholds,
        "categorical_map": bundle.feature_spec.categorical_map,
        "feature_to_model_columns": bundle.feature_spec.feature_to_model_columns,
    }

    for spec in bundle.feature_spec.numeric:
        item = dict(spec)
        item["name"] = str(item["name"])
        feature_spec["numeric"].append(item)

    for spec in bundle.feature_spec.categorical:
        item = dict(spec)
        item["name"] = str(item["name"])
        item["options"] = [str(opt) for opt in item["options"]]
        feature_spec["categorical"].append(item)

    payload = {
        "meta": {
            "source_repo": "Yamahono2002/twolayer_bank_gui",
            "entrypoint": "docs/index.html",
        },
        "ui_text": UI_TEXT,
        "feature_labels": feature_labels,
        "subscale_labels": subscale_labels,
        "categorical_option_labels": categorical_labels,
        "feature_spec": feature_spec,
        "subscale_names": [str(name) for name in bundle.notebook_ns.get("subscale_names", [])],
        "model_header": list(bundle.model_header),
        "model": _model_to_dict(bundle.model),
        "fold": int(bundle.fold),
        "validation_ap": bundle.ap,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Export model data for GitHub Pages")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/model-data.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    payload = build_payload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
