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
python src/fourier.py
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

Evaluation uses stratified train/test splitting (consistent labeled-sample
counts per fold, not left to the luck of the random permutation) plus two
cross-validated R² estimates: a 5-seed **pooled** CV (every out-of-fold
prediction combined into one R², not an average of 5 individually unstable
per-fold R² values) and a full **leave-one-out** CV across all labeled rows.
See `results/gpr_metrics.txt` for both.

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

Global EI search is followed by an anchor-seeded local search: the VAE
latent position of each of the 5 known training waveform families (sine,
square, revsawtooth, doublepeak, asymmetric) is encoded, and a local
search refines around each one — global random/EI search alone reliably
finds only one such region by luck. Final proposals are gated on GPR
prediction uncertainty (only confident predictions are reported; the
proposal count is not padded with low-confidence candidates to force a
fixed number), with diversity enforced in shape-descriptor space rather
than raw latent distance.

Output: `results/optimizer/proposed_waveforms.csv`,
`results/optimizer/proposed_waveforms.png`,
`results/optimizer/optimization_history.png`

---

## Key Results

### VAE
- Reconstruction MAE: reported in `results/vae_metrics.txt`
- Good reconstruction quality confirmed visually

### GPR (R — Drag Reduction)
- Pooled 5-seed CV R² = 0.8526, MAE = 0.0233 (15 out-of-fold predictions
  pooled into one R², not averaged per-fold — see note below)
- Leave-one-out CV R² = 0.9078, MAE = 0.0166 (all 24 labeled rows,
  each held out and predicted exactly once)
- Feature importance: Reynolds > wavenumber > period > kurtosis > skewness
  (Reynolds' and max_acceleration/omega_plus's rankings are partly a
  kernel length-scale bound artifact, not a fully converged estimate —
  see `results/gpr_metrics.txt` and treat the ranking as directional)
- Amplitude excluded from features (fixed at 4.5 in filtered dataset)

> An earlier draft of this README reported Mean R² = 0.60 ± 0.21 from
> averaging 5 independently-computed per-fold R² values. That statistic
> is unstable at this sample size (~3 test points per fold) and was
> replaced with the pooled/LOOCV numbers above. Separately, that specific
> 0.60 ± 0.21 figure traces to an earlier, pre-amplitude-filter version of
> this evaluation (full 110-row / 10-feature dataset, before shape-signal
> isolation) — not a bug, but a materially different methodology than
> what this pipeline runs today, and not reproducible against the current
> amplitude-isolated setup.

### Optimizer
- Operating conditions: A+=4.5, T+=125.0, Re=180
- Sinusoidal baseline R: 0.1809
- Best proposed R: 0.2327
- Proposals beating baseline: 6/10 (all 10 reported proposals clear the
  confidence bar — see Step 5)
- Best proposal shape: near-symmetric, flat-topped
  (skewness≈-0.001, kurtosis≈-1.995)

---

## Model Notes

**vae.pt** — Trained VAE. Good reconstruction quality confirmed.
Used to encode/decode waveforms for optimizer.

**gpr_R.pkl** — Drag reduction (R) predictor.
Pooled CV R²=0.8526, LOOCV R²=0.9078 — see Key Results above for what
these mean and how they differ from a naive per-fold average.
Used in optimization pipeline (single fit on the main 80/10/10 split,
not itself an ensemble — `optimize.py` prints a reminder of this).

**gpr_S.pkl** — Net power saving (S) predictor.
R²=0.9137 on a single 2-sample test split — unlike R, S is not
stratified or cross-validated, so treat this number as illustrative
only, not a validated estimate. Trained on fewer labeled rows than R
(23/35 filtered rows have an S label vs. 24/35 for R).
NOT used in optimization. Retained for reference only.

**scaler.pkl** — StandardScaler fitted on GPR training features.
Must be used to transform features before any GPR query.

**feature_names.pkl** — Exact column order expected by GPR.
Used by optimizer to build feature vectors correctly.

---

## Dataset

### dns_labeled.csv
110 labeled DNS rows from two sources:

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
- Small dataset (110 rows, 35 after the amplitude filter, 24 R-labeled)
  — primary constraint throughout; addressed for evaluation stability via
  stratified splits and pooled/leave-one-out CV, not by adding data
- S labels missing for 59/110 rows — S GPR excluded from pipeline, and
  unlike R, S evaluation is not stratified or cross-validated
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