# DTLMUW Non-Sinusoidal Waveform Discovery
### VAE + Bayesian Optimization Pipeline for Turbulent Drag Reduction

A machine learning pipeline to discover novel non-sinusoidal 
Downstream-Traveling Longitudinal Micro-Ultrasonic Wave (DTLMUW) 
waveform shapes predicted to outperform sinusoidal wall oscillations 
for turbulent drag reduction — inspired by dolphin skin biomechanics.

---

## Setup

```bash
git clone https://github.com/adhman326/dtlmuw-project.git
cd dtlmuw-project
pip install -r requirements.txt
```

### GPU Verification
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Tested on:
- NVIDIA RTX A6000 (48GB, CUDA)
- AMD Instinct MI300X (192GB, ROCm) — VAE trained here

---

## Run Order

### Step 1 — Generate Synthetic Waveforms
```bash
python src/generate.py
```
Generates 10,000 non-sinusoidal Fourier superposition waveforms for
VAE shape training. Runs in under 1 minute on CPU. No GPU needed.

Output: `synthetic/synthetic_waveforms.csv`

---

### Step 2 — Verify Dataset
```bash
python src/dataset.py
```
Loads `dns_labeled.csv`, generates waveform shapes from family names,
combines with synthetic waveforms, and reports dataset statistics.

Output: Console summary of dataset splits and missing values.

---

### Step 3 — Train VAE (or load existing)
```bash
# train from scratch (500 epochs, ~10-30 min on GPU)
python src/vae.py

# load existing trained model and evaluate only
python src/vae.py --eval-only
```

Architecture: 3-layer encoder/decoder, HIDDEN_DIM=256, LATENT_DIM=8.
Training uses KL annealing (100 epoch warmup) to prevent posterior collapse.

Output: `models/vae.pt`, `results/vae_loss_curve.png`,
`results/vae_reconstructions.png`, `results/vae_metrics.txt`

---

### Step 4 — Train GPR Surrogate
```bash
python src/surrogate.py
```

Trains two GPR models (R and S) using physics-grounded shape descriptors
(skewness, kurtosis, zero-crossing rate, max acceleration) combined with
physical parameters (period, wavenumber, reynolds, omega). Dataset filtered
to amplitude ≈ 4.5 to isolate shape signal from amplitude dominance.

Output: `models/gpr_R.pkl`, `models/gpr_S.pkl`, `models/scaler.pkl`,
`models/feature_names.pkl`, `results/gpr_metrics.txt`

---

### Step 5 — Run Bayesian Optimizer
```bash
python src/optimize.py
```

Searches VAE latent space using Expected Improvement acquisition function.
Each candidate is decoded to a waveform, shape descriptors computed, then
queried through GPR. Amplitude fixed at A+=4.5 — optimizer finds advantage
through waveform shape alone.

Output: `results/optimizer/proposed_waveforms.csv`,
`results/optimizer/proposed_waveforms.png`,
`results/optimizer/optimization_history.png`

---

## Key Results

### VAE
- Reconstruction MAE: reported in `results/vae_metrics.txt`
- Good reconstruction quality confirmed visually

### GPR (R — Drag Reduction)
- Mean R² = 0.60 ± 0.21 across 5 random seeds
- Mean MAE = 0.038
- Feature importance: Reynolds > wavenumber > period > kurtosis > skewness
- Amplitude excluded from features (fixed at 4.5 in filtered dataset)

### Optimizer
- Operating conditions: A+=4.5, T+=125.0, Re=180
- Sinusoidal baseline R: 0.1809
- Best proposed R: 0.2345 (+29.6% relative improvement)
- Proposals beating baseline: 2/10
- Best proposal shape: near-symmetric, flat-topped
  (skewness=0.003, kurtosis=-1.971)

---

## Model Notes

**vae.pt** — Trained VAE. Good reconstruction quality confirmed.
Used to encode/decode waveforms for optimizer.

**gpr_R.pkl** — Drag reduction (R) predictor.
Mean R²=0.60 ± 0.21, MAE=0.038. Used in optimization pipeline.

**gpr_S.pkl** — Net power saving (S) predictor.
R²=-0.83. Unreliable due to insufficient labeled data
(39 training samples, 59 missing S values in dataset).
NOT used in optimization. Retained for reference only.

**scaler.pkl** — StandardScaler fitted on GPR training features.
Must be used to transform features before any GPR query.

**feature_names.pkl** — Exact column order expected by GPR.
Used by optimizer to build feature vectors correctly.

---

## Dataset

### dns_labeled.csv
111 labeled DNS rows from two sources:

| Source | Rows | Waveform Families | R Available | S Available |
|---|---|---|---|---|
| Cimarelli et al. 2013 | ~41 | sine, square, revsawtooth, doublepeak, asymmetric | Yes | Partial |
| Gatti & Quadrio 2016 | ~70 | sine only | Yes | Partial |

Columns: source, waveform_family, amplitude_plus, amplitude_star,
period_plus, wavenumber_plus, reynolds, R, S, omega_plus,
R_uncertainty, simulation_type, notes, w_00...w_63

Note: w_00–w_63 columns are generated at runtime from waveform_family
labels using mathematical reconstruction — not stored in the CSV.

### Known Limitations
- Small dataset (111 rows) — primary constraint throughout
- S labels missing for 59/111 rows — S GPR excluded from pipeline
- Cimarelli data covers spanwise oscillations, not out-of-plane DTLMUW
- Only 4 DNS data points at maximum amplitude (A+=9.0)
- Sinusoidal baseline at matched conditions based on only 2 DNS rows

---

## Dependencies
numpy
scipy
pandas
matplotlib
scikit-learn
torch
joblib
Install via: `pip install -r requirements.txt`

For ROCm (AMD GPU):
```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
```

---

## Citation

If using this pipeline, please cite:
- Cimarelli et al. (2013). Physics of Fluids, 25(7), 075102
- Gatti & Quadrio (2016). Journal of Fluid Mechanics