import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

# ── Configuration ────────────────────────────────────────────────────────────
LABELED_PATH   = "data/raw/dns_labeled.csv"
SYNTHETIC_PATH = "synthetic/synthetic_waveforms.csv"
N_POINTS       = 64   # number of velocity values per waveform
RANDOM_SEED    = 42

# ── Column definitions ───────────────────────────────────────────────────────
VELOCITY_COLS = [f"w_{i:02d}" for i in range(N_POINTS)]
PHYSICAL_COLS = ["amplitude_plus", "amplitude_star", "period_plus",
                 "wavenumber_plus", "reynolds", "omega_plus", "R_uncertainty"]
TARGET_COLS   = ["R", "S"]

# ── Load and Clean Labeled DNS Data ─────────────────────────────────────────
def load_labeled(path=LABELED_PATH):
    df = pd.read_csv(path)

    print(f"Loaded {len(df)} labeled rows")
    print(f"\nMissing values per column:")
    print(df.isnull().sum())

    # generate waveform shapes from family names
    df = add_waveform_columns(df)

    return df

def generate_waveform_from_family(family_name, n_points=N_POINTS):
    """
    Mathematically reconstruct a waveform shape from its family name.
    Covers the 5 waveform families from Cimarelli et al. with R/S data.
    """
    t = np.linspace(0, 1, n_points, endpoint=False)

    if family_name == "sine":
        # (a) standard sine wave
        w = np.sin(2 * np.pi * t)

    elif family_name == "square":
        # (b) square wave — +1 for first half, -1 for second half
        w = np.where(t < 0.5, 1.0, -1.0).astype(float)

    elif family_name == "revsawtooth":
        # (e) reverse sawtooth with step
        w = np.where(t < 0.5, 1.0 - 2*t, 3.0 - 2*t).astype(float)
        w = np.clip(w, -1.0, 1.0)

    elif family_name == "doublepeak":
        # (f) double peak — two humps positive, one valley negative
        w = np.sin(2 * np.pi * t) + 0.5 * np.sin(4 * np.pi * t)

    elif family_name == "asymmetric":
        # (j) asymmetric with flat section then sharp drop
        w = np.where(t < 0.4, np.sin(2 * np.pi * t / 0.8),
            np.where(t < 0.6, -1.0,
            np.sin(2 * np.pi * (t - 0.6) / 0.8 + np.pi))).astype(float)

    else:
        print(f"Warning: unknown waveform family '{family_name}', using sine")
        w = np.sin(2 * np.pi * t)

    # normalize so max absolute value = 1.0
    max_val = np.max(np.abs(w))
    if max_val > 0:
        w = w / max_val

    return w.astype(np.float32)


def add_waveform_columns(df):
    """
    For rows that have a waveform_family label but no velocity columns,
    generate the waveform shape mathematically and add w_00...w_63 columns.
    """
    # initialize velocity columns with NaN if they don't exist
    for col in VELOCITY_COLS:
        if col not in df.columns:
            df[col] = np.nan

    # fill in waveform shapes for rows that have a family name
    filled = 0
    for idx, row in df.iterrows():
        family = row.get("waveform_family", None)
        if pd.notnull(family) and pd.isnull(row.get("w_00", np.nan)):
            w = generate_waveform_from_family(str(family).strip().lower())
            for i, col in enumerate(VELOCITY_COLS):
                df.at[idx, col] = w[i]
            filled += 1

    print(f"Generated waveform shapes for {filled} labeled rows")
    return df

# ── Extract Waveform Shapes From Labeled Data ────────────────────────────────
def extract_waveforms_labeled(df):
    """
    Extract the 64-point wall velocity arrays from labeled DNS rows.
    Returns numpy array of shape (n_rows, 64).
    Only rows that have all 64 velocity columns filled are included.
    """
    # check which rows have waveform data
    has_waveform = df[VELOCITY_COLS].notnull().all(axis=1)
    df_waves = df[has_waveform].copy()

    print(f"\nLabeled rows with waveform shape data: {len(df_waves)} / {len(df)}")

    waveforms = df_waves[VELOCITY_COLS].values.astype(np.float32)
    return waveforms, df_waves


# ── Load Synthetic Waveforms ─────────────────────────────────────────────────
def load_synthetic(path=SYNTHETIC_PATH):
    df = pd.read_csv(path)
    waveforms = df[VELOCITY_COLS].values.astype(np.float32)
    print(f"Loaded {len(waveforms)} synthetic waveforms")
    return waveforms


# ── Build VAE Training Set ───────────────────────────────────────────────────
def build_vae_dataset(labeled_df):
    """
    Combine labeled DNS waveforms + synthetic waveforms for VAE training.
    VAE only needs waveform shapes — no labels required.
    Returns combined numpy array of shape (n_total, 64).
    """
    # labeled waveforms
    labeled_waves, _ = extract_waveforms_labeled(labeled_df)

    # synthetic waveforms
    synthetic_waves = load_synthetic()

    # combine
    combined = np.concatenate([labeled_waves, synthetic_waves], axis=0)
    print(f"\nVAE training set: {len(combined)} total waveforms")
    print(f"  — {len(labeled_waves)} from labeled DNS")
    print(f"  — {len(synthetic_waves)} synthetic")

    return combined


# ── Build GPR Training Set ───────────────────────────────────────────────────
def build_gpr_dataset(labeled_df):
    """
    Extract rows that have at least one target label (R or S).
    Returns features X and targets y as numpy arrays.
    Only uses physical parameter columns + targets — no raw waveform shape.
    """
    df = labeled_df.copy()

    # keep only rows with at least one label
    has_label = df[TARGET_COLS].notnull().any(axis=1)
    df_labeled = df[has_label].copy()

    print(f"\nGPR dataset: {len(df_labeled)} rows with at least one label")
    print(f"  — Rows with R: {df_labeled['R'].notnull().sum()}")
    print(f"  — Rows with S: {df_labeled['S'].notnull().sum()}")
    print(f"  — Rows with both R and S: {df_labeled[['R','S']].notnull().all(axis=1).sum()}")

    # features — use physical columns, fill missing with column median
    X = df_labeled[PHYSICAL_COLS].copy()
    for col in PHYSICAL_COLS:
        median = X[col].median()
        n_missing = X[col].isnull().sum()
        if n_missing > 0:
            print(f"  Filling {n_missing} missing values in {col} with median {median:.4f}")
        X[col] = X[col].fillna(median)

    X = X.values.astype(np.float32)
    y_R = df_labeled["R"].values.astype(np.float32)
    y_S = df_labeled["S"].values.astype(np.float32)

    return X, y_R, y_S, df_labeled


# ── Train / Validation / Test Split ─────────────────────────────────────────
def split_labeled(X, y_R, y_S, train=0.70, val=0.15, test=0.15, seed=RANDOM_SEED,
                   stratify_target=None, min_test_labeled=3):
    """
    Split labeled data into train / val / test sets.
    Applied to labeled rows only — synthetic data is never split this way.

    stratify_target: None, "R", or "S". When given, rows are split into a
    "has a valid label for this target" pool and a "doesn't" pool, each
    pool is divided by the same train/val/test ratios, and the results are
    recombined. This keeps the fraction of labeled rows landing in the test
    set consistent across seeds instead of leaving it to chance — with a
    plain index-based split, a random 10% slice of ALL rows can land on
    very few (or very many) of the rows that actually carry that target's
    label, purely by luck of the seed.
    """
    assert abs(train + val + test - 1.0) < 1e-6, "Splits must sum to 1.0"

    rng = np.random.default_rng(seed)
    n = len(X)

    def _split_pool(idx_pool):
        idx_pool = rng.permutation(idx_pool)
        n_pool   = len(idx_pool)
        n_tr     = int(round(n_pool * train))
        n_va     = int(round(n_pool * val))
        return (idx_pool[:n_tr],
                idx_pool[n_tr:n_tr + n_va],
                idx_pool[n_tr + n_va:])

    if stratify_target is not None:
        y_target = {"R": y_R, "S": y_S}[stratify_target]
        labeled_mask = ~np.isnan(y_target)

        tr_l, va_l, te_l = _split_pool(np.where(labeled_mask)[0])
        tr_u, va_u, te_u = _split_pool(np.where(~labeled_mask)[0])

        train_idx = rng.permutation(np.concatenate([tr_l, tr_u]))
        val_idx   = rng.permutation(np.concatenate([va_l, va_u]))
        test_idx  = rng.permutation(np.concatenate([te_l, te_u]))

        strat_note = f"  (stratified on '{stratify_target}' label availability)"
        if len(te_l) < min_test_labeled:
            print(f"WARNING: only {len(te_l)} labeled '{stratify_target}' "
                  f"sample(s) landed in the test split (recommended minimum: "
                  f"{min_test_labeled}). With so few labeled rows available "
                  f"({labeled_mask.sum()} total), R²/MAE for this seed will "
                  f"be noisy — this is a small-data limit, not something the "
                  f"split can fix.")
    else:
        indices   = rng.permutation(n)
        n_train   = int(n * train)
        n_val     = int(n * val)
        train_idx = indices[:n_train]
        val_idx   = indices[n_train:n_train + n_val]
        test_idx  = indices[n_train + n_val:]
        strat_note = ""

    print(f"\nSplit: {len(train_idx)} train / {len(val_idx)} val / "
          f"{len(test_idx)} test{strat_note}")

    return (X[train_idx], y_R[train_idx], y_S[train_idx],
            X[val_idx],   y_R[val_idx],   y_S[val_idx],
            X[test_idx],  y_R[test_idx],  y_S[test_idx])


# ── Entry Point — Sanity Check ───────────────────────────────────────────────
if __name__ == "__main__":
    # load
    labeled_df = load_labeled()

    # VAE dataset
    vae_data = build_vae_dataset(labeled_df)
    print(f"\nVAE input shape: {vae_data.shape}")
    print(f"Value range: [{vae_data.min():.4f}, {vae_data.max():.4f}]")

    # GPR dataset
    X, y_R, y_S, df_gpr = build_gpr_dataset(labeled_df)
    print(f"\nGPR feature matrix shape: {X.shape}")

    # split
    splits = split_labeled(X, y_R, y_S)
    X_train, yR_train, yS_train = splits[0], splits[1], splits[2]
    X_val,   yR_val,   yS_val   = splits[3], splits[4], splits[5]
    X_test,  yR_test,  yS_test  = splits[6], splits[7], splits[8]

    print("\nAll checks passed — dataset ready for VAE and GPR.")