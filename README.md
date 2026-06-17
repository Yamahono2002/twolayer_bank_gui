# twolayer_bank_gui

Bank two-layer additive model repository.

This repository is organized for a static browser demo.

## Contents

- `docs/`: GitHub Pages static demo
- `docs/index.html`: entry point for the browser GUI
- `docs/app.js`: client-side prediction logic
- `docs/style.css`: visual styling
- `docs/model-data.json`: pre-exported model parameters and labels
- `docs/.nojekyll`: disables Jekyll on GitHub Pages

## How it works

The browser GUI loads `docs/model-data.json` and runs predictions client-side with JavaScript.

This means users can change input features and see predictions without installing Python or running model training.

## Run locally

To preview the GUI locally, serve the `docs/` directory with a simple HTTP server:

```bash
cd /path/to/twolayer_bank_gui-main
python3 -m http.server 8000 --directory docs
