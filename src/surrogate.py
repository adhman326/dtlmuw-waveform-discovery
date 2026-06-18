import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
import sys
from scipy.stats import skew, kurtosis
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import load_labeled, split_labeled, VELOCITY_COLS
from vae import VAE, N_POINTS, HIDDEN_DIM, LATENT_DIM, SAVE_PATH

# ── Configuration ────────────────────────────────────────────────────────────
GPR_SAVE_R   = "models/gpr_R.pkl"
GPR_SAVE_S   = "models/gpr_S.pkl"
SCALER_SAVE  = "models/scaler.pkl"
RANDOM_SEED  = 42
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Shape Descriptor Computation ──────────────────────────────────────────────
def compute_shape_descriptors(waveform):
    """
    Compute physically meaningful shape descriptors from a 64-point
    wall velocity waveform. These replace raw latent coordinates as
    GPR input features — compact, interpretable, and physically grounded.

    Returns a dict with: skewness, kurtosis, zero_crossing_rate,
    max_acceleration.
    """
    n = len(waveform)

    # skewness — rise/fall asymmetry
    sk = skew(waveform)

    # kurtosis — peak sharpness vs flatness
    ku = kurtosis(waveform)

    # zero-crossing rate — how oscillatory the shape is
    signs           = np.sign(waveform)
    signs[signs == 0] = 1  # avoid zero-sign edge case
    crossings       = np.sum(np.diff(signs) != 0)
    zc_rate         = crossings / n

    # max acceleration — key physical scaling parameter (Cimarelli)
    dt              = 1.0 / n
    acceleration    = np.diff(waveform) / dt
    max_accel       = np.max(np.abs(acceleration))

    return {
        "skewness":          sk,
        "kurtosis":           ku,
        "zero_crossing_rate": zc_rate,
        "max_acceleration":   max_accel,
    }


def compute_descriptors_batch(waveforms):
    """
    Compute shape descriptors for a batch of waveforms.
    waveforms: array of shape (n_samples, 64)
    Returns a DataFrame with one row per waveform.
    """
    rows = []
    for w in waveforms:
        rows.append(compute_shape_descriptors(w))
    return pd.DataFrame(rows)


# ── Load Trained VAE (still used for sanity checks / future steps) ──────────
def load_vae():
    model = VAE(
        n_points   = N_POINTS,
        hidden_dim = HIDDEN_DIM,
        latent_dim = LATENT_DIM,
    ).to(device)
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    model.eval()
    print(f"VAE loaded from {SAVE_PATH}")
    return model


# ── Get Labeled Rows With Waveform Data ───────────────────────────────────────
def get_labeled_waveforms(labeled_df):
    """
    Extract rows that have full 64-point waveform shape data.
    Returns the waveform array and the corresponding dataframe rows.
    """
    has_waveform = labeled_df[VELOCITY_COLS].notnull().all(axis=1)
    df_waves     = labeled_df[has_waveform].copy()
    waveforms    = df_waves[VELOCITY_COLS].values.astype(np.float32)

    print(f"Found {len(waveforms)} labeled rows with waveform shape data")
    return waveforms, df_waves


# ── Build GPR Feature Matrix ──────────────────────────────────────────────────
def build_gpr_features(waveforms, df):
    """
    Build GPR feature matrix using shape descriptors + physical parameters.
    This replaces the latent-coordinate approach — more interpretable
    and better suited to a small dataset.
    """
    # compute shape descriptors for every waveform
    descriptors_df = compute_descriptors_batch(waveforms)
    descriptors_df.index = df.index

    # physical parameters
    phys_cols = []
    for col in ["amplitude_plus", "amplitude_star", "period_plus",
                "wavenumber_plus", "reynolds", "omega_plus"]:
        if col in df.columns:
            phys_cols.append(col)

    phys_df = df[phys_cols].copy()
    for col in phys_cols:
        median = phys_df[col].median()
        n_miss = phys_df[col].isnull().sum()
        if n_miss > 0:
            print(f"  Filling {n_miss} missing {col} with median {median:.4f}")
        phys_df[col] = phys_df[col].fillna(median)

    # combine shape descriptors + physical parameters
    X = pd.concat([descriptors_df, phys_df], axis=1)
    feature_names = list(X.columns)
    X = X.values.astype(np.float32)

    print(f"GPR feature matrix shape: {X.shape}")
    print(f"Features used: {feature_names}")
    return X, feature_names


# ── Train GPR ─────────────────────────────────────────────────────────────────
def train_gpr(X_train, y_train, target_name="R"):
    valid = np.logical_not(np.isnan(y_train))
    X_tr  = X_train[valid]
    y_tr  = y_train[valid]
    print(f"\nTraining GPR for {target_name} on {len(y_tr)} samples "
          f"({(~valid).sum()} skipped — missing labels)")

    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) *
        Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5) +
        WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 1.0))
    )

    gpr = GaussianProcessRegressor(
        kernel               = kernel,
        n_restarts_optimizer = 5,
        normalize_y          = True,
        random_state         = RANDOM_SEED,
    )

    gpr.fit(X_tr, y_tr)
    print(f"Optimized kernel: {gpr.kernel_}")
    return gpr


# ── Evaluate GPR ──────────────────────────────────────────────────────────────
def evaluate_gpr(gpr, X_test, y_test, target_name="R"):
    valid = np.logical_not(np.isnan(y_test))
    X_te  = X_test[valid]
    y_te  = y_test[valid]

    if len(y_te) == 0:
        print(f"No valid test samples for {target_name} — skipping")
        return None, None

    y_pred, y_std = gpr.predict(X_te, return_std=True)

    mae = mean_absolute_error(y_te, y_pred)
    r2  = r2_score(y_te, y_pred)

    lower    = y_pred - 1.96 * y_std
    upper    = y_pred + 1.96 * y_std
    coverage = np.mean((y_te >= lower) & (y_te <= upper))

    print(f"\n── GPR Evaluation: {target_name} ──────────────────────")
    print(f"Test samples:        {len(y_te)}")
    print(f"MAE:                 {mae:.4f}")
    print(f"R²:                  {r2:.4f}")
    print(f"95% CI coverage:     {coverage:.2%}  (target: ~95%)")
    print(f"Mean uncertainty:    {np.mean(y_std):.4f}")
    print("────────────────────────────────────────────────────")

    return mae, r2


# ── Feature Importance via Length Scales ──────────────────────────────────────
def print_feature_importance(gpr, feature_names):
    """
    The Matern kernel's length_scale per dimension tells us how
    sensitive R is to each feature — shorter length scale means
    the GPR needs finer resolution in that dimension, i.e. it
    matters more.
    """
    try:
        # navigate kernel structure to find Matern component
        matern_kernel = gpr.kernel_.k1.k2  # ConstantKernel * Matern + White
        length_scales = matern_kernel.length_scale

        if np.isscalar(length_scales):
            print("\nKernel uses a single shared length scale "
                  "(isotropic) — cannot rank individual features.")
            return

        print(f"\n── Feature Importance (via length scale) ──────────")
        importance = 1.0 / np.array(length_scales)
        ranked = sorted(zip(feature_names, importance),
                        key=lambda x: -x[1])
        for name, imp in ranked:
            print(f"  {name:20s}  relative importance: {imp:.4f}")
        print("─────────────────────────────────────────────────────")
    except Exception as e:
        print(f"Could not extract feature importance: {e}")


# ── Plot GPR Predictions ──────────────────────────────────────────────────────
def plot_gpr_predictions(gpr, X_test, y_test, target_name="R"):
    valid = np.logical_not(np.isnan(y_test))
    X_te  = X_test[valid]
    y_te  = y_test[valid]

    if len(y_te) == 0:
        return

    y_pred, y_std = gpr.predict(X_te, return_std=True)

    plt.figure(figsize=(6, 6))
    plt.errorbar(y_te, y_pred, yerr=1.96*y_std,
                 fmt="o", color="#2d2d6e", alpha=0.7,
                 ecolor="#aaaacc", capsize=3, label="Predictions ± 95% CI")

    lims = [min(y_te.min(), y_pred.min()) - 0.02,
            max(y_te.max(), y_pred.max()) + 0.02]
    plt.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")

    plt.xlabel(f"Actual {target_name}")
    plt.ylabel(f"Predicted {target_name}")
    plt.title(f"GPR: Predicted vs Actual {target_name} (Shape Descriptors)")
    plt.legend()
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    path = f"results/gpr_predictions_{target_name}.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Plot saved to {path}")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 1. load labeled data
    labeled_df = load_labeled()

    # 2. get waveforms with full shape data
    waveforms, df_waves = get_labeled_waveforms(labeled_df)

    # 3. build feature matrix using shape descriptors + physical params
    X, feature_names = build_gpr_features(waveforms, df_waves)

    # 4. get targets
    y_R = df_waves["R"].values.astype(np.float32)
    y_S = df_waves["S"].values.astype(np.float32)

    # 5. train/val/test split — generous split given small dataset
    splits = split_labeled(X, y_R, y_S, train=0.80, val=0.10, test=0.10)
    X_train, yR_train, yS_train = splits[0], splits[1], splits[2]
    X_val,   yR_val,   yS_val   = splits[3], splits[4], splits[5]
    X_test,  yR_test,  yS_test  = splits[6], splits[7], splits[8]

    # 6. scale features
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_SAVE)
    print(f"Scaler saved to {SCALER_SAVE}")

    # 7. train GPR for R
    gpr_R = train_gpr(X_train, yR_train, target_name="R")
    joblib.dump(gpr_R, GPR_SAVE_R)
    print(f"GPR (R) saved to {GPR_SAVE_R}")

    # 8. train GPR for S
    gpr_S = train_gpr(X_train, yS_train, target_name="S")
    joblib.dump(gpr_S, GPR_SAVE_S)
    print(f"GPR (S) saved to {GPR_SAVE_S}")

    # 9. evaluate
    mae_R, r2_R = evaluate_gpr(gpr_R, X_test, yR_test, target_name="R")
    mae_S, r2_S = evaluate_gpr(gpr_S, X_test, yS_test, target_name="S")

    # 10. feature importance
    print_feature_importance(gpr_R, feature_names)

    # 11. plots
    plot_gpr_predictions(gpr_R, X_test, yR_test, target_name="R")
    plot_gpr_predictions(gpr_S, X_test, yS_test, target_name="S")

    # 12. save metrics + feature names for optimizer to reuse
    with open("results/gpr_metrics.txt", "w") as f:
        f.write("GPR Evaluation Metrics (Shape Descriptor Features)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Features: {feature_names}\n\n")
        if mae_R is not None:
            f.write(f"Drag Reduction (R):\n  MAE: {mae_R:.4f}\n  R²: {r2_R:.4f}\n\n")
        if mae_S is not None:
            f.write(f"Net Power Saving (S):\n  MAE: {mae_S:.4f}\n  R²: {r2_S:.4f}\n")

    # save feature names so optimize.py knows the exact column order
    joblib.dump(feature_names, "models/feature_names.pkl")

    print("\nGPR training complete (shape descriptor features).")
    print("Next step: update optimize.py to use these features.")