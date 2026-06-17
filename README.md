# twolayer_bank_gui

Bank two-layer additive model repository.

## Layout

- `docs/`: GitHub Pages static demo
- `export_model_data.py`: exports trained model parameters to `docs/model-data.json`
- `new_bank/base_cluster_gui.py`: Python GUI / runtime that loads the notebook-based model
- `new_bank/hier.ipynb`: notebook containing the model definitions and training logic
- `requirements.txt`: development dependencies

## Static demo

The browser demo reads `docs/model-data.json` and runs predictions client-side with JavaScript.

## Regenerate the JSON

```bash
.venv/bin/python export_model_data.py --output docs/model-data.json
