# DTLMUW Non-Sinusoidal Waveform Discovery

## Setup
pip install -r requirements.txt

## Run Order
# Step 1 — Generate synthetic waveforms
python src/fourier.py

# Step 2 — Verify dataset
python src/dataset.py

# Step 3 — Train VAE
python src/vae.py

## Data
Place `dns_labeled.csv` in `data/raw/` before running.

## MI300X / ROCm Setup
Install PyTorch with ROCm support:
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0

## Model Notes

**gpr_R.pkl** — Drag reduction (R) predictor. R²=0.76, MAE=0.042. 
Used in optimization pipeline.

**gpr_S.pkl** — Net power saving (S) predictor. R²=-0.71. 
Unreliable due to insufficient labeled data (39 training samples, 
59 missing S values in dataset). NOT used in optimization. 
Retained for reference only.

**vae.pt** — Trained VAE. Good reconstruction quality.