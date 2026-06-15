import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
import sys
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import load_labeled, build_gpr_dataset, split_labeled
from vae import VAE, N_POINTS, HIDDEN_DIM, LATENT_DIM, SAVE_PATH

# ── Configuration ────────────────────────────────────────────────────────────
GPR_SAVE_R = "models/gpr_R.pkl" # saved GPR for drag reduction
GPR_SAVE_S = "models/gpr_S.pkl" # saved GPR for net power saving
SCALER_SAVE = "models/scaler.pkl" # saved input scaler 
RANDOM_SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load Trained VAE ──────────────────────────────────────────────────────────
def load_vae():
    """Load the trained VAE and return it in eval mode"""
    model = VAE(
        n_points = N_POINTS,
        hidden_dim = HIDDEN_DIM,
        latent_dim = LATENT_DIM,
    ).to(device)
    model.load_state_dict(torch.load(SAVE_PATH, map_location = device))
    model.eval()
    print(f"VAE loaded from {SAVE_PATH}")
    return model

# ── Encode Labeled Waveforms to Latent Space ──────────────────────────────────
def encode_labeled_waveforms(vae, labeled_df):
    """
    Take all labeled DNS rows that have waveform shape data and encode them to latent 
    coordinates using the trained VAE encoder. Returns latent coordinates and the 
    corresponding dataframe rows.
    """
    from dataset import VELOCITY_COLS

    #get rows that have waveform data
    has_waveform = labeled_df[VELOCITY_COLS].notnull().all(axis = 1)
    df_waves     = labeled_df[has_waveform].copy()

    waveforms = df_waves[VELOCITY_COLS].values.astype(np.float32)
    X_tensor  = torch.tensor(waveforms).to(device)

    with torch.no_grad():
        latent = vae.encode(X_tensor).cpu().numpy()

    print(f"Encoded {len(latent)} waveforms to latent space "
          f"(shape: {latent.shape})")
    return latent, df_waves

# ── Build GPR Feature Matrix ──────────────────────────────────────────────────
def build_gpr_features(latent, df):
    """
    Combine latent coordinates with physical parameters to build full feature matrix for GPR
    
    Features:
    - 8 latent dimensions (waveform shape)
    - amplitude_plus or amplitude_star (operating condition)
    - reynolds (flow condition)
    - period_plus (operating condition — where available)
    """

    # latent coordinates
    latent_df = pd.DataFrame(
        latent,
        columns=[f"z_{i}" for i in range(latent.shape[1])],
        index=df.index
    )

    # physical parameters - use what's available
    phys_cols = []
    for col in ["amplitude_plus", "amplitude_star", "period_plus", "reynolds"]:
        if col in df.columns:
            phys_cols.append(col)

    phys_df = df[phys_cols].copy()

    # fill missing physical params with median
    for col in phys_cols:
        median = phys_df[col].median()
        n_miss = phys_df[col].isnull().sum()
        if n_miss > 0:
            print(f"  Filling {n_miss} missing {col} values with median "
                  f"{median:.4f}")
        phys_df[col] = phys_df[col].fillna(median)

    # combine latent + physical
    X = pd.concat([latent_df, phys_df], axis=1).values.astype(np.float32)
    print(f"GPR feature matrix shape: {X.shape}")
    return X

# ── Train GPR ─────────────────────────────────────────────────────────────────
def train_gpr(X_train, y_train, target_name = "R"):
    """
    Train a Gaussian Process Regressor on the training set. Uses a Matern kernel which is 
    well suited to smooth physical functions. Returns the fitted GPR.
    """
    # remove rows where target is NaN
    valid = np.logical_not(np.isnan(y_train))
    X_tr = X_train[valid]
    y_tr = y_train[valid]
    print(f"\nTraining GPR for {target_name} on {len(y_tr)} samples "
          f"({(~valid).sum()} skipped — missing labels)")
    
    # kernel: ConstantKernel * Matern + WhiteKernel(noise)
    kernel = (
        ConstantKernel(1.0, constant_value_bounds = (1e-3, 1e3)) *
        Matern(length_scale = 1.0, length_scale_bounds = (1e-2, 1e2), nu = 2.5) +
        WhiteKernel(noise_level = 0.01, noise_level_bounds = (1e-5, 1.0))
    )

    gpr = GaussianProcessRegressor (
        kernel = kernel,
        n_restarts_optimizer = 5, # try 5 random starts to find best kernel params
        normalize_y = True, # normalize target — helps with small datasets
        random_state = RANDOM_SEED,
    )

    gpr.fit(X_tr, y_tr)
    print(f"Optimized kernel: {gpr.kernel_}")
    return gpr, X_tr, y_tr

# ── Evaluate GPR ──────────────────────────────────────────────────────────────
def evaluate_gpr(gpr, X_test, y_test, target_name="R"):
    """
    Evaluate GPR on held-out test set.
    Reports MAE, R², and calibration of uncertainty estimates.
    """
    # remove NaN targets
    valid  = np.logical_not(np.isnan(y_test))
    X_te   = X_test[valid]
    y_te   = y_test[valid]

    if len(y_te) == 0:
        print(f"No valid test samples for {target_name} — skipping evaluation")
        return None, None

    # predict with uncertainty
    y_pred, y_std = gpr.predict(X_te, return_std=True)

    mae = mean_absolute_error(y_te, y_pred)
    r2  = r2_score(y_te, y_pred)

    # calibration check — what fraction of true values fall within 95% CI?
    lower = y_pred - 1.96 * y_std
    upper = y_pred + 1.96 * y_std
    coverage = np.mean((y_te >= lower) & (y_te <= upper))

    print(f"\n── GPR Evaluation: {target_name} ──────────────────────")
    print(f"Test samples:        {len(y_te)}")
    print(f"MAE:                 {mae:.4f}")
    print(f"R²:                  {r2:.4f}")
    print(f"95% CI coverage:     {coverage:.2%}  (target: ~95%)")
    print(f"Mean uncertainty:    {np.mean(y_std):.4f}")
    print("────────────────────────────────────────────────────")

    return mae, r2
    
# ── Plot GPR Predictions ──────────────────────────────────────────────────────
def plot_gpr_predictions(gpr, X_test, y_test, target_name="R"):
    """
    Predicted vs actual scatter plot with uncertainty bars.
    A perfect predictor would show all points on the diagonal.
    """
    valid  = ~np.isnan(y_test)
    X_te   = X_test[valid]
    y_te   = y_test[valid]

    if len(y_te) == 0:
        return

    y_pred, y_std = gpr.predict(X_te, return_std=True)

    plt.figure(figsize=(6, 6))
    plt.errorbar(y_te, y_pred, yerr=1.96*y_std,
                 fmt="o", color="#2d2d6e", alpha=0.7,
                 ecolor="#aaaacc", capsize=3, label="Predictions ± 95% CI")

    # perfect prediction line
    lims = [min(y_te.min(), y_pred.min()) - 0.02,
            max(y_te.max(), y_pred.max()) + 0.02]
    plt.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")

    plt.xlabel(f"Actual {target_name}")
    plt.ylabel(f"Predicted {target_name}")
    plt.title(f"GPR: Predicted vs Actual {target_name}")
    plt.legend()
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    path = f"results/gpr_predictions_{target_name}.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Plot saved to {path}")

# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # 1. load VAE
    vae = load_vae()

    # 2. load labeled data
    labeled_df = load_labeled()

    # 3. encode waveforms to latent space
    latent, df_waves = encode_labeled_waveforms(vae, labeled_df)

    # 4. build feature matrix
    X = build_gpr_features(latent, df_waves)

    # 5. get targets
    y_R = df_waves["R"].values.astype(np.float32)
    y_S = df_waves["S"].values.astype(np.float32)

    # 6. train/val/test split
    splits = split_labeled(X, y_R, y_S)
    X_train, yR_train, yS_train = splits[0], splits[1], splits[2]
    X_val,   yR_val,   yS_val   = splits[3], splits[4], splits[5]
    X_test,  yR_test,  yS_test  = splits[6], splits[7], splits[8]

    # 7. scale features — GPR is sensitive to feature scale
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # save scaler — needed later for optimizer
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, SCALER_SAVE)
    print(f"Scaler saved to {SCALER_SAVE}")

    # 8. train GPR for R
    gpr_R, X_tr_R, y_tr_R = train_gpr(X_train, yR_train, target_name="R")
    joblib.dump(gpr_R, GPR_SAVE_R)
    print(f"GPR (R) saved to {GPR_SAVE_R}")

    # 9. train GPR for S
    gpr_S, X_tr_S, y_tr_S = train_gpr(X_train, yS_train, target_name="S")
    joblib.dump(gpr_S, GPR_SAVE_S)
    print(f"GPR (S) saved to {GPR_SAVE_S}")

    # 10. evaluate both GPRs on test set
    mae_R, r2_R = evaluate_gpr(gpr_R, X_test, yR_test, target_name="R")
    mae_S, r2_S = evaluate_gpr(gpr_S, X_test, yS_test, target_name="S")

    # 11. plots
    plot_gpr_predictions(gpr_R, X_test, yR_test, target_name="R")
    plot_gpr_predictions(gpr_S, X_test, yS_test, target_name="S")

    # 12. save metrics
    os.makedirs("results", exist_ok=True)
    with open("results/gpr_metrics.txt", "w") as f:
        f.write("GPR Evaluation Metrics\n")
        f.write("=" * 40 + "\n\n")
        if mae_R is not None:
            f.write(f"Drag Reduction (R):\n")
            f.write(f"  MAE: {mae_R:.4f}\n")
            f.write(f"  R²:  {r2_R:.4f}\n\n")
        if mae_S is not None:
            f.write(f"Net Power Saving (S):\n")
            f.write(f"  MAE: {mae_S:.4f}\n")
            f.write(f"  R²:  {r2_S:.4f}\n")

    print("\nGPR training complete.")
    print("Next step: run src/optimize.py to find novel waveforms.")