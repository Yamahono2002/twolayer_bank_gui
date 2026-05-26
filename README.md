# twolayer_bank_gui

GitHub Pages 用の静的デモと、Python/Streamlit 用の実行版を同じリポジトリで管理するための構成です。

## Files

- `new_bank/base_cluster_gui.py`: 既存の Python GUI / モデル実装
- `streamlit_app.py`: Streamlit 用の入口
- `export_model_data.py`: 学習済みモデルを `docs/model-data.json` に書き出すスクリプト
- `docs/`: GitHub Pages 用の静的サイト

## GitHub Pages

`docs/index.html` を Pages の公開元に指定してください。

## Regenerate data

```bash
.venv/bin/python export_model_data.py --output docs/model-data.json
```

## Streamlit

```bash
streamlit run streamlit_app.py
```
