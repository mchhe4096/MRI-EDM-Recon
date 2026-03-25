# Gradio Inference Demo

This demo is a local web app for single-slice MRI reconstruction:

- Input: one `.pt` file containing `kspace_full`
- Output: reconstructed MRI image (`Recon`)

## 1. Install

```bash
cd D:\MRI\MRI-EDM-Recon
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Run

```bash
python app.py
```

Open `http://127.0.0.1:7860`.

## 3. Notes

- Default checkpoint path is `outputs/ckpts/last.pt`.
- If your checkpoint has no `ema_model`, uncheck `使用 EMA 权重`.
- `GT` will only be shown when uploaded `.pt` includes `img_gt`.

