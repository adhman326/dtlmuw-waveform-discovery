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