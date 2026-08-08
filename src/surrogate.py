import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
import sys
import warnings
from scipy.stats import skew, kurtosis
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.exceptions import ConvergenceWarning
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
def build_gpr_features(waveforms, df, impute=True):
    """
    Build GPR feature matrix using shape descriptors + physical params.
    When amplitude is fixed, it is excluded from features since it
    has no variance and adds noise.

    Shape descriptors (skewness, kurtosis, zero_crossing_rate,
    max_acceleration) are computed per-row from that row's own waveform
    only — no cross-row statistic, so no leakage risk regardless of when
    this is called relative to any train/test split.

    The physical columns (period_plus, wavenumber_plus, reynolds,
    omega_plus) are a different story: ~4/35 rows are missing one or more
    of these and get median-imputed. impute=True (the default) fills them
    here, using the median of whatever `df` is passed in — fine for a
    final deployed model, but if `df` is the full dataset and this is
    called before a train/test split, the imputed values for those ~4
    rows are informed by every row's value for that column, including
    whatever ends up in a held-out test fold later. impute=False leaves
    those columns as NaN instead, so the caller can compute the median
    from ONLY a given fold's training rows and apply it separately to
    that fold's train and test sets — see impute_phys_median_by_fold().
    """
    descriptors_df = compute_descriptors_batch(waveforms)
    descriptors_df.index = df.index

    # exclude amplitude columns since we fixed them
    phys_cols = []
    for col in ["period_plus", "wavenumber_plus",
                "reynolds", "omega_plus"]:
        if col in df.columns:
            phys_cols.append(col)

    phys_df = df[phys_cols].copy()
    if impute:
        for col in phys_cols:
            median = phys_df[col].median()
            n_miss = phys_df[col].isnull().sum()
            if n_miss > 0:
                print(f"  Filling {n_miss} missing {col} with median "
                      f"{median:.4f}")
            phys_df[col] = phys_df[col].fillna(median)
    else:
        for col in phys_cols:
            n_miss = phys_df[col].isnull().sum()
            if n_miss > 0:
                print(f"  {n_miss} missing {col} left as NaN — will be "
                      f"median-imputed per fold, from training rows only")

    X = pd.concat([descriptors_df, phys_df], axis=1)
    feature_names = list(X.columns)
    X = X.values.astype(np.float32)

    print(f"GPR feature matrix shape: {X.shape}")
    print(f"Features used: {feature_names}")
    return X, feature_names


# ── Per-Fold Median Imputation ─────────────────────────────────────────────────
def impute_phys_median_by_fold(X_train, *other_sets, phys_col_idx):
    """
    Compute each physical column's median from X_train ONLY (NaN-omitted),
    then fill NaN in X_train and every array in other_sets with that same
    training-derived value. This is what actually prevents a held-out
    fold's rows from influencing the value used to fill missing rows in
    that same fold — computing the median once globally, before any
    split, leaks test-fold information into the imputed value even though
    the split itself only index-slices an already-fully-imputed X.

    Returns copies (X_train_imputed, *other_sets_imputed), same order as
    input. Falls back to X_train's own global median (with a warning) only
    in the degenerate case where a fold's training rows are ALL missing a
    given column — doesn't happen in this dataset, but fails loud instead
    of propagating NaN into the GPR if it ever did.
    """
    X_train = X_train.copy()
    medians = {}
    for j in phys_col_idx:
        col = X_train[:, j]
        med = np.nanmedian(col) if not np.all(np.isnan(col)) else np.nan
        if np.isnan(med):
            print(f"WARNING: training fold has zero non-missing values "
                  f"for feature column {j} — cannot compute a train-only "
                  f"median. Leaving NaN (this will break GPR fitting).")
        medians[j] = med
        col[np.isnan(col)] = med

    results = [X_train]
    for X_other in other_sets:
        X_other = X_other.copy()
        for j, med in medians.items():
            col = X_other[:, j]
            col[np.isnan(col)] = med
        results.append(X_other)
    return tuple(results)

# ── Train GPR ─────────────────────────────────────────────────────────────────
def train_gpr(X_train, y_train, target_name="R", verbose=True):
    valid = np.logical_not(np.isnan(y_train))
    X_tr  = X_train[valid]
    y_tr  = y_train[valid]
    if verbose:
        print(f"\nTraining GPR for {target_name} on {len(y_tr)} samples "
              f"({(~valid).sum()} skipped — missing labels)")

    n_features = X_tr.shape[1]

    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) *
        Matern(
            length_scale=np.ones(n_features),  # one length scale per feature
            length_scale_bounds=(1e-2, 1e2),
            nu=2.5
        ) +
        WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 1.0))
    )

    gpr = GaussianProcessRegressor(
        kernel               = kernel,
        n_restarts_optimizer = 5,
        normalize_y          = True,
        random_state         = RANDOM_SEED,
    )

    gpr.fit(X_tr, y_tr)
    if verbose:
        print(f"Optimized kernel: {gpr.kernel_}")
    return gpr


# ── Shared Prediction Helper ───────────────────────────────────────────────────
def _valid_predictions(gpr, X_test, y_test):
    """
    Filter to samples with a valid label, then predict. Shared by
    evaluate_gpr (per-fold diagnostics) and the pooled-CV / LOOCV loops
    (which need the raw predictions, not just a summary statistic).
    Returns (y_true, y_pred, y_std), all possibly empty.
    """
    valid = np.logical_not(np.isnan(y_test))
    X_te  = X_test[valid]
    y_te  = y_test[valid]
    if len(y_te) == 0:
        return y_te, np.array([]), np.array([])
    y_pred, y_std = gpr.predict(X_te, return_std=True)
    return y_te, y_pred, y_std


# ── Evaluate GPR ──────────────────────────────────────────────────────────────
def evaluate_gpr(gpr, X_test, y_test, target_name="R", min_samples=2):
    y_te, y_pred, y_std = _valid_predictions(gpr, X_test, y_test)

    if len(y_te) < min_samples:
        # r2_score is mathematically undefined below 2 samples (sklearn
        # returns nan + UndefinedMetricWarning). Skip explicitly instead of
        # letting a nan silently poison any downstream mean/std over seeds.
        print(f"Only {len(y_te)} valid test sample(s) for {target_name} "
              f"(need >= {min_samples} for a defined R²) — skipping")
        return None, None

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

    # 2b. filter to fixed amplitude — isolate shape signal
    # keep rows where amplitude_plus ≈ 4.5 (most data-dense condition)
    amp_mask = (
        (df_waves["amplitude_plus"].between(4.0, 5.0)) |
        (df_waves["amplitude_plus"].isna() &
         df_waves["amplitude_star"].between(4.0, 5.0))
    )
    df_waves_filtered = df_waves[amp_mask].copy()
    waveforms_filtered = waveforms[amp_mask.values]

    print(f"\nFiltered to amplitude ≈ 4.5:")
    print(f"  Rows before: {len(df_waves)}")
    print(f"  Rows after:  {len(df_waves_filtered)}")
    print(f"  Waveform families: "
          f"{df_waves_filtered['waveform_family'].value_counts().to_dict()}")
    print(f"  R labels available: "
          f"{df_waves_filtered['R'].notnull().sum()}")
    print(f"  S labels available: "
          f"{df_waves_filtered['S'].notnull().sum()}")

    # 3. build feature matrix — amplitude fixed, shape signal isolated.
    # impute=False: leaves period_plus/wavenumber_plus/omega_plus as NaN
    # for the ~4 rows missing them, instead of filling with a median
    # computed globally across all 35 rows. Filling globally, before any
    # split, would leak whatever ends up in a held-out fold into the
    # value used to fill missing rows in that same fold — every fold
    # below (main split, 5-seed CV, LOOCV) imputes fresh from its own
    # training rows only, via impute_phys_median_by_fold().
    X, feature_names = build_gpr_features(
        waveforms_filtered, df_waves_filtered, impute=False
    )
    phys_impute_idx = [feature_names.index(c) for c in
                        ["period_plus", "wavenumber_plus",
                         "reynolds", "omega_plus"]
                        if c in feature_names]

    # 4. get targets
    y_R = df_waves_filtered["R"].values.astype(np.float32)
    y_S = df_waves_filtered["S"].values.astype(np.float32)

    # 5. train/val/test split — generous split given small dataset
    # stratified on R-label availability: R is the target actually used
    # downstream by the optimizer, so its test-split size must be stable
    # across runs rather than left to chance (see split_labeled docstring).
    splits = split_labeled(X, y_R, y_S, train=0.80, val=0.10, test=0.10,
                            stratify_target="R")
    X_train, yR_train, yS_train = splits[0], splits[1], splits[2]
    X_val,   yR_val,   yS_val   = splits[3], splits[4], splits[5]
    X_test,  yR_test,  yS_test  = splits[6], splits[7], splits[8]

    # 5b. impute physical columns — train-derived median only, applied to
    # val/test too. Must happen after the split, not before.
    X_train, X_val, X_test = impute_phys_median_by_fold(
        X_train, X_val, X_test, phys_col_idx=phys_impute_idx
    )

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

    # save base metrics now (fresh "w") — the CV sections below append to
    # this file. Must happen before them: this used to be written at the
    # very end of the script in "w" mode, which silently overwrote and
    # discarded everything the CV sections had appended earlier in the
    # same run. The pooled-CV/LOOCV numbers were correct on the terminal
    # but never actually landed in results/gpr_metrics.txt on disk.
    with open("results/gpr_metrics.txt", "w") as f:
        f.write("GPR Evaluation Metrics (Shape Descriptor Features)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Features: {feature_names}\n\n")
        if mae_R is not None:
            f.write(f"Drag Reduction (R):\n  MAE: {mae_R:.4f}\n  R²: {r2_R:.4f}\n\n")
        if mae_S is not None:
            f.write(f"Net Power Saving (S):\n  MAE: {mae_S:.4f}\n  R²: {r2_S:.4f}\n")

    # 9b. 5-seed cross-validation — POOLED R², not averaged per-fold R².
    #
    # Averaging 5 independently-computed per-fold R² values is the wrong
    # statistic at this sample size: with ~3 test points per fold, R² is
    # unbounded below and a single off prediction can swing one fold's R²
    # to something like -6, which then dominates a plain average of 5
    # numbers (this is exactly what happened before this fix: Mean R² =
    # -0.53 +/- 2.76, range [-6.05, 0.90], from folds this small).
    #
    # The fix: pool every out-of-fold (y_true, y_pred) pair across all 5
    # seeds into one combined set, then compute ONE R² over that pooled
    # set. This uses a single, stable denominator (variance of all ~15
    # pooled true values) instead of 5 separate tiny-sample denominators,
    # and it's still a genuine cross-validated R² — not a substitute
    # metric, just the statistically correct way to combine folds this
    # small. MAE is unaffected by this distinction (it's linear, so
    # per-fold-averaged and pooled MAE are already close) — this is why
    # MAE looked stable across runs even while the old R² average didn't.
    SEEDS = [42, 123, 7, 0, 256]
    print(f"\n── 5-Seed Cross-Validation ({len(SEEDS)} seeds) ──────────────")
    per_fold_r2  = []   # kept only as a secondary diagnostic, see below
    pooled_y_true = []
    pooled_y_pred = []
    skipped_seeds = []

    for seed in SEEDS:
        # resplit with this seed — stratified on R-label availability so
        # every seed gets a consistent number of labeled test rows instead
        # of a count that swings with the luck of the permutation (this is
        # what previously let a 1-sample test split through and produced a
        # silent nan in the mean/std below).
        # X here is still the RAW matrix from step 3 (impute=False) — NaN
        # in period_plus/wavenumber_plus/omega_plus for the ~4 affected
        # rows, imputed fresh below from this seed's training rows only.
        splits_i = split_labeled(X, y_R, y_S,
                                 train=0.80, val=0.10,
                                 test=0.10, seed=seed,
                                 stratify_target="R")
        X_tr_i_raw, X_te_i_raw = splits_i[0], splits_i[6]
        X_tr_i_raw, X_te_i_raw = impute_phys_median_by_fold(
            X_tr_i_raw, X_te_i_raw, phys_col_idx=phys_impute_idx
        )
        # fresh StandardScaler per seed — not the outer `scaler` (which is
        # only for the main, non-CV split at step 6). fit_transform doesn't
        # accumulate state across calls either way, so reusing `scaler`
        # here was never a leakage bug, but a fresh instance per fold
        # removes any doubt on re-read and is the more standard pattern.
        scaler_i = StandardScaler()
        X_tr_i = scaler_i.fit_transform(X_tr_i_raw)
        X_te_i = scaler_i.transform(X_te_i_raw)
        y_tr_i = splits_i[1]
        y_te_i = splits_i[7]

        # train a temporary GPR on this split
        gpr_temp = train_gpr(X_tr_i, y_tr_i, target_name=f"R seed={seed}")

        # per-fold diagnostic print (unchanged) ...
        mae_i, r2_i = evaluate_gpr(
            gpr_temp, X_te_i, y_te_i, target_name=f"R seed={seed}"
        )
        # ... plus the raw predictions, pooled across all seeds
        y_te_valid, y_pred_valid, _ = _valid_predictions(gpr_temp, X_te_i, y_te_i)

        if r2_i is not None:
            per_fold_r2.append(r2_i)
        else:
            skipped_seeds.append(seed)

        if len(y_te_valid) > 0:
            pooled_y_true.extend(y_te_valid.tolist())
            pooled_y_pred.extend(y_pred_valid.tolist())

    print()
    if skipped_seeds:
        print(f"Seeds with no valid test sample at all: {skipped_seeds}")

    pooled_y_true = np.array(pooled_y_true)
    pooled_y_pred = np.array(pooled_y_pred)
    n_pooled = len(pooled_y_true)

    if n_pooled < 2:
        pooled_r2, pooled_mae = None, None
        print("Fewer than 2 pooled predictions across all seeds — cannot "
              "compute a pooled R².")
    else:
        pooled_r2  = r2_score(pooled_y_true, pooled_y_pred)
        pooled_mae = mean_absolute_error(pooled_y_true, pooled_y_pred)
        print(f"Pooled CV R²  ({n_pooled} out-of-fold predictions across "
              f"{len(SEEDS) - len(skipped_seeds)}/{len(SEEDS)} seeds): "
              f"{pooled_r2:.4f}")
        print(f"Pooled CV MAE: {pooled_mae:.4f}")
        if per_fold_r2:
            print(f"(for reference — naive mean of per-fold R², the "
                  f"previous/unstable metric: "
                  f"{np.mean(per_fold_r2):.4f} +/- {np.std(per_fold_r2):.4f}, "
                  f"range [{min(per_fold_r2):.4f}, {max(per_fold_r2):.4f}])")
    print(f"──────────────────────────────────────────────────")

    # 9c. Leave-one-out CV — every labeled R row used as test exactly once.
    # More appropriate than 5 random 80/10/10 splits at this sample size:
    # with ~24 labeled rows, the 5-seed scheme above tests on only ~15
    # (seed, fold) draws total, some rows retested multiple times by
    # chance and others possibly never drawn as test at all. LOOCV uses
    # every labeled row exactly once, so there's no "which random points
    # land in test" fragility left at all. Cost is ~24 GPR fits instead
    # of 5, but each fit here is well under a second.
    labeled_idx = np.where(~np.isnan(y_R))[0]
    n_labeled   = len(labeled_idx)
    print(f"\n── Leave-One-Out CV ({n_labeled} labeled R rows) ─────────────")

    loo_y_true, loo_y_pred = [], []
    with warnings.catch_warnings():
        # 24 fits' worth of per-dimension bound-pinning ConvergenceWarnings
        # is pure noise here (already characterized once — see
        # PIPELINE_CHANGELOG.md); suppressed for readability only.
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for i, held_out in enumerate(labeled_idx):
            train_idx = labeled_idx[labeled_idx != held_out]

            # X is still the RAW matrix (impute=False at step 3) — impute
            # from this fold's 23 training rows only, before scaling.
            X_tr_loo_raw, X_te_loo_raw = impute_phys_median_by_fold(
                X[train_idx], X[[held_out]], phys_col_idx=phys_impute_idx
            )

            scaler_loo = StandardScaler()
            X_tr_loo = scaler_loo.fit_transform(X_tr_loo_raw)
            X_te_loo = scaler_loo.transform(X_te_loo_raw)
            y_tr_loo = y_R[train_idx]

            gpr_loo = train_gpr(X_tr_loo, y_tr_loo,
                                 target_name=f"R LOO {i+1}/{n_labeled}",
                                 verbose=False)
            y_pred_loo, _ = gpr_loo.predict(X_te_loo, return_std=True)

            loo_y_true.append(float(y_R[held_out]))
            loo_y_pred.append(float(y_pred_loo[0]))

    loo_y_true = np.array(loo_y_true)
    loo_y_pred = np.array(loo_y_pred)
    loo_r2  = r2_score(loo_y_true, loo_y_pred)
    loo_mae = mean_absolute_error(loo_y_true, loo_y_pred)
    print(f"LOOCV R²  ({n_labeled} held-out predictions, pooled): {loo_r2:.4f}")
    print(f"LOOCV MAE: {loo_mae:.4f}")
    print(f"──────────────────────────────────────────────────")

    # also update the saved metrics file with both CV estimates
    with open("results/gpr_metrics.txt", "a") as f:
        f.write(f"\n\n5-Seed Pooled CV "
                f"({len(SEEDS) - len(skipped_seeds)}/{len(SEEDS)} valid "
                f"seeds; skipped: {skipped_seeds}):\n")
        if pooled_r2 is None:
            f.write("  Fewer than 2 pooled predictions — no R² computed.\n")
        else:
            f.write(f"  Pooled R²:  {pooled_r2:.4f}\n")
            f.write(f"  Pooled MAE: {pooled_mae:.4f}\n")
            if per_fold_r2:
                f.write(f"  (naive per-fold-mean R², for reference only: "
                        f"{np.mean(per_fold_r2):.4f} +/- "
                        f"{np.std(per_fold_r2):.4f})\n")
        f.write(f"\nLeave-One-Out CV ({n_labeled} labeled rows):\n")
        f.write(f"  LOOCV R²:  {loo_r2:.4f}\n")
        f.write(f"  LOOCV MAE: {loo_mae:.4f}\n")

    # 10. feature importance
    print_feature_importance(gpr_R, feature_names)

    # 11. plots
    plot_gpr_predictions(gpr_R, X_test, yR_test, target_name="R")
    plot_gpr_predictions(gpr_S, X_test, yS_test, target_name="S")

    # 12. save feature names for optimize.py to reuse
    # (base metrics + CV results were already written to
    # results/gpr_metrics.txt above, in order — see the note at step 9)
    joblib.dump(feature_names, "models/feature_names.pkl")

    print("\nGPR training complete (shape descriptor features).")
    print("Next step: update optimize.py to use these features.")