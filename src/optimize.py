import numpy as np
import torch
import matplotlib.pyplot as plt
import joblib
import os
import sys
import pandas as pd
from scipy.stats import norm, skew, kurtosis

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import load_labeled, VELOCITY_COLS
from vae import VAE, N_POINTS, HIDDEN_DIM, LATENT_DIM, SAVE_PATH

# ── Configuration ────────────────────────────────────────────────────────────
N_ITERATIONS    = 500    # number of BO steps
N_RANDOM_INIT   = 50     # random exploration before BO kicks in
N_PROPOSALS     = 10     # number of novel waveforms to propose at end
LATENT_BOUNDS   = 3.0    # search within ±3 std of latent space
RANDOM_SEED     = 42

# fixed operating conditions — match GPR training data (A+=4.5 filter)
FIXED_AMPLITUDE  = 4.5
FIXED_PERIOD     = 125.0
FIXED_REYNOLDS   = 180
FIXED_WAVENUMBER = 0.0
FIXED_OMEGA      = 2 * np.pi / 125.0

RESULTS_DIR = "results/optimizer"
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Load Models ───────────────────────────────────────────────────────────────
def load_models():
    vae = VAE(
        n_points   = N_POINTS,
        hidden_dim = HIDDEN_DIM,
        latent_dim = LATENT_DIM,
    ).to(device)
    vae.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    vae.eval()
    print(f"VAE loaded from {SAVE_PATH}")

    gpr_R   = joblib.load("models/gpr_R.pkl")
    scaler  = joblib.load("models/scaler.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    print(f"GPR (R) loaded")
    print(f"Scaler loaded")
    print(f"Features expected: {feature_names}")
    return vae, gpr_R, scaler, feature_names


# ── Decode Latent to Waveform ─────────────────────────────────────────────────
def decode_latent(vae, z):
    """Decode a latent vector to a 64-point waveform."""
    z_tensor = torch.tensor(z, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        waveform = vae.decode(z_tensor).cpu().numpy()[0]
    return waveform


# ── Compute Shape Descriptors ─────────────────────────────────────────────────
def compute_shape_descriptors(waveform):
    """
    Compute the same 4 shape descriptors used in surrogate.py.
    Must match exactly — same features, same order.
    """
    n  = len(waveform)
    sk = skew(waveform)
    ku = kurtosis(waveform)

    signs             = np.sign(waveform)
    signs[signs == 0] = 1
    crossings         = np.sum(np.diff(signs) != 0)
    zc_rate           = crossings / n

    dt         = 1.0 / n
    accel      = np.diff(waveform) / dt
    max_accel  = np.max(np.abs(accel))

    return sk, ku, zc_rate, max_accel


# ── Build Feature Vector ──────────────────────────────────────────────────────
def build_feature(waveform, scaler, feature_names):
    """
    Build a GPR feature vector from a decoded waveform.
    Computes shape descriptors then appends fixed physical parameters.
    Feature order must exactly match what surrogate.py produced.
    """
    sk, ku, zc_rate, max_accel = compute_shape_descriptors(waveform)

    # build feature dict matching surrogate.py column order
    feature_dict = {
        "skewness":           sk,
        "kurtosis":           ku,
        "zero_crossing_rate": zc_rate,
        "max_acceleration":   max_accel,
        "period_plus":        FIXED_PERIOD,
        "wavenumber_plus":    FIXED_WAVENUMBER,
        "reynolds":           FIXED_REYNOLDS,
        "omega_plus":         FIXED_OMEGA,
    }

    # build vector in exact feature_names order
    x = np.array([feature_dict[f] for f in feature_names],
                 dtype=np.float32).reshape(1, -1)
    x_scaled = scaler.transform(x)
    return x_scaled


# ── GPR Prediction ────────────────────────────────────────────────────────────
def predict_R(waveform, gpr_R, scaler, feature_names):
    """Predict drag reduction R for a decoded waveform."""
    x         = build_feature(waveform, scaler, feature_names)
    mu, sigma = gpr_R.predict(x, return_std=True)
    return float(mu[0]), float(sigma[0])


# ── Expected Improvement ──────────────────────────────────────────────────────
def expected_improvement(mu, sigma, best_so_far, xi=0.01):
    improvement          = mu - best_so_far - xi
    Z                    = improvement / (sigma + 1e-9)
    ei                   = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-10]    = 0.0
    return ei


# ── Random Latent Samples ─────────────────────────────────────────────────────
def random_latent_samples(n, latent_dim=LATENT_DIM, bounds=LATENT_BOUNDS):
    return np.random.uniform(
        -bounds, bounds, size=(n, latent_dim)
    ).astype(np.float32)


# ── Bayesian Optimization Loop ────────────────────────────────────────────────
def run_bayesian_optimization(vae, gpr_R, scaler, feature_names):
    """
    Main BO loop. Searches VAE latent space for waveform shapes
    that maximize predicted drag reduction at fixed A+=4.5.
    Each candidate is decoded to a waveform before GPR query —
    the optimizer evaluates actual shapes, not just coordinates.
    """
    np.random.seed(RANDOM_SEED)

    evaluated_z        = []
    evaluated_waveforms = []
    evaluated_R        = []
    evaluated_std      = []

    print(f"\nPhase 1: Random exploration ({N_RANDOM_INIT} points)...")
    print(f"Fixed operating conditions: A+={FIXED_AMPLITUDE}, "
          f"T+={FIXED_PERIOD}, Re={FIXED_REYNOLDS}")

    # phase 1 — random exploration
    random_z = random_latent_samples(N_RANDOM_INIT)
    for z in random_z:
        waveform  = decode_latent(vae, z)
        mu, sigma = predict_R(waveform, gpr_R, scaler, feature_names)
        evaluated_z.append(z)
        evaluated_waveforms.append(waveform)
        evaluated_R.append(mu)
        evaluated_std.append(sigma)

    best_so_far = max(evaluated_R)
    best_idx    = np.argmax(evaluated_R)
    print(f"Best R after random exploration: {best_so_far:.4f}")
    print(f"Best waveform shape descriptors:")
    sk, ku, zc, ac = compute_shape_descriptors(
        evaluated_waveforms[best_idx]
    )
    print(f"  skewness={sk:.3f}, kurtosis={ku:.3f}, "
          f"zero_crossing_rate={zc:.3f}, max_accel={ac:.3f}")

    print(f"\nPhase 2: Bayesian optimization ({N_ITERATIONS} iterations)...")

    # phase 2 — BO loop
    for iteration in range(N_ITERATIONS):

        # generate candidate latent vectors
        candidates = random_latent_samples(1000)

        # decode all candidates to waveforms and compute features
        candidate_features = []
        for z in candidates:
            w = decode_latent(vae, z)
            x = build_feature(w, scaler, feature_names)
            candidate_features.append(x[0])
        candidate_features = np.array(candidate_features)

        # predict R for all candidates
        mu_all, sigma_all = gpr_R.predict(
            candidate_features, return_std=True
        )

        # compute EI and pick best candidate
        ei       = expected_improvement(mu_all, sigma_all, best_so_far)
        best_idx = np.argmax(ei)
        next_z   = candidates[best_idx]

        # evaluate chosen candidate
        next_waveform     = decode_latent(vae, next_z)
        mu, sigma         = predict_R(
            next_waveform, gpr_R, scaler, feature_names
        )

        evaluated_z.append(next_z)
        evaluated_waveforms.append(next_waveform)
        evaluated_R.append(mu)
        evaluated_std.append(sigma)

        if mu > best_so_far:
            best_so_far = mu

        if (iteration + 1) % 100 == 0:
            print(f"  Iteration {iteration+1:4d} | "
                  f"Best R: {best_so_far:.4f} | "
                  f"Current R: {mu:.4f} ± {sigma:.4f}")

    print(f"\nOptimization complete. Best predicted R: {best_so_far:.4f}")
    return (np.array(evaluated_z),
            evaluated_waveforms,
            np.array(evaluated_R),
            np.array(evaluated_std))


# ── Get Sinusoidal Baseline ───────────────────────────────────────────────────
def get_sinusoidal_baseline(labeled_df):
    """
    Best sinusoidal R at matched operating conditions (A+=4.5, T+=125).
    This is what the optimizer needs to beat using shape alone.
    """
    sine_rows = labeled_df[
        (labeled_df["waveform_family"] == "sine") &
        (labeled_df["amplitude_plus"].between(4.0, 5.0)) &
        (labeled_df["period_plus"].between(100, 150))
    ]["R"].dropna()

    if len(sine_rows) > 0:
        best_sine_R = sine_rows.max()
        print(f"Best sinusoidal R at A+=4.5, T+=125: {best_sine_R:.4f} "
              f"(from {len(sine_rows)} matching rows)")
    else:
        best_sine_R = labeled_df[
            labeled_df["waveform_family"] == "sine"
        ]["R"].dropna().max()
        print(f"Warning: no exact match. Using global sine max: "
              f"{best_sine_R:.4f}")

    return best_sine_R


# ── Extract Diverse Proposals ─────────────────────────────────────────────────
def extract_proposals(evaluated_z, evaluated_waveforms,
                      evaluated_R, evaluated_std,
                      n_proposals=N_PROPOSALS):
    """
    Extract top N proposals by predicted R with diversity constraint.
    Proposals must be at least 0.5 units apart in latent space.
    Also computes shape descriptors for each proposal.
    """
    sorted_idx = np.argsort(evaluated_R)[::-1]
    selected   = []
    selected_z = []

    for idx in sorted_idx:
        z        = evaluated_z[idx]
        r        = evaluated_R[idx]
        std      = evaluated_std[idx]
        waveform = evaluated_waveforms[idx]

        # diversity check
        too_close = any(
            np.linalg.norm(z - z_sel) < 0.5
            for z_sel in selected_z
        )

        if not too_close:
            sk, ku, zc, ac = compute_shape_descriptors(waveform)
            selected.append({
                "latent_coords":      z,
                "waveform":           waveform,
                "predicted_R":        r,
                "uncertainty":        std,
                "skewness":           sk,
                "kurtosis":           ku,
                "zero_crossing_rate": zc,
                "max_acceleration":   ac,
            })
            selected_z.append(z)

        if len(selected) >= n_proposals:
            break

    print(f"\nSelected {len(selected)} diverse proposals:")
    for i, p in enumerate(selected):
        print(f"  Proposal {i+1:2d}: R={p['predicted_R']:.4f} "
              f"± {p['uncertainty']:.4f} | "
              f"skew={p['skewness']:.3f} | "
              f"kurt={p['kurtosis']:.3f}")

    return selected


# ── Plot Optimization History ─────────────────────────────────────────────────
def plot_optimization_history(evaluated_R, best_sine_R):
    running_best = np.maximum.accumulate(evaluated_R)

    plt.figure(figsize=(10, 4))
    plt.plot(evaluated_R,  alpha=0.3, color="#2d2d6e",
             linewidth=0.8, label="Evaluated R")
    plt.plot(running_best, color="#2d2d6e",
             linewidth=2.0, label="Best R so far")
    plt.axhline(best_sine_R, color="#e06c75", linewidth=1.5,
                linestyle="--",
                label=f"Sinusoidal baseline R={best_sine_R:.4f}")
    plt.axvline(N_RANDOM_INIT, color="gray", linewidth=0.8,
                linestyle=":", label="BO starts")

    plt.xlabel("Evaluation number")
    plt.ylabel("Predicted R")
    plt.title("Bayesian Optimization — Shape Space Search (A+=4.5 fixed)")
    plt.legend()
    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(f"{RESULTS_DIR}/optimization_history.png", dpi=150)
    plt.show()
    print("Optimization history saved")


# ── Plot Proposed Waveforms ───────────────────────────────────────────────────
def plot_proposals(proposals, best_sine_R):
    n    = len(proposals)
    cols = 5
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 3, rows * 2.5))
    axes = axes.flatten() if n > 1 else [axes]

    t_ref    = np.linspace(0, 1, N_POINTS, endpoint=False)
    sine_ref = np.sin(2 * np.pi * t_ref)

    for i, prop in enumerate(proposals):
        ax = axes[i]
        ax.plot(prop["waveform"], color="#2d2d6e",
                linewidth=1.5, label="Proposed")
        ax.plot(sine_ref, color="#e06c75", linewidth=1.0,
                linestyle="--", alpha=0.5, label="Sine")
        ax.set_title(
            f"Proposal {i+1}\n"
            f"R={prop['predicted_R']:.4f} ± {prop['uncertainty']:.4f}\n"
            f"skew={prop['skewness']:.2f} kurt={prop['kurtosis']:.2f}",
            fontsize=7
        )
        ax.set_ylim(-1.3, 1.3)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.tick_params(labelsize=6)
        if i == 0:
            ax.legend(fontsize=6)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        f"Novel Waveform Proposals (A+=4.5 fixed)\n"
        f"Sinusoidal baseline R={best_sine_R:.4f}",
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/proposed_waveforms.png", dpi=150)
    plt.show()
    print("Proposed waveforms plot saved")


# ── Save Proposals to CSV ─────────────────────────────────────────────────────
def save_proposals(proposals, best_sine_R):
    rows = []
    for i, prop in enumerate(proposals):
        row = {
            "proposal_id":        i + 1,
            "predicted_R":        prop["predicted_R"],
            "uncertainty":        prop["uncertainty"],
            "beats_baseline":     prop["predicted_R"] > best_sine_R,
            "skewness":           prop["skewness"],
            "kurtosis":           prop["kurtosis"],
            "zero_crossing_rate": prop["zero_crossing_rate"],
            "max_acceleration":   prop["max_acceleration"],
            "fixed_amplitude":    FIXED_AMPLITUDE,
            "fixed_period":       FIXED_PERIOD,
            "fixed_reynolds":     FIXED_REYNOLDS,
        }
        for j, val in enumerate(prop["waveform"]):
            row[f"w_{j:02d}"] = val
        rows.append(row)

    df   = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = f"{RESULTS_DIR}/proposed_waveforms.csv"
    df.to_csv(path, index=False)

    print(f"\nProposed waveforms saved to {path}")
    print(f"\n── Final Summary ───────────────────────────────────")
    print(f"Operating conditions:          A+={FIXED_AMPLITUDE}, "
          f"T+={FIXED_PERIOD}, Re={FIXED_REYNOLDS}")
    print(f"Sinusoidal baseline R:         {best_sine_R:.4f}")
    print(f"Best proposed R:               "
          f"{max(p['predicted_R'] for p in proposals):.4f}")
    beats = sum(1 for p in proposals
                if p["predicted_R"] > best_sine_R)
    print(f"Proposals beating baseline:    {beats} / {len(proposals)}")
    print(f"Best proposal skewness:        "
          f"{proposals[0]['skewness']:.4f}")
    print(f"Best proposal kurtosis:        "
          f"{proposals[0]['kurtosis']:.4f}")
    print(f"────────────────────────────────────────────────────")
    return df


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # load models
    vae, gpr_R, scaler, feature_names = load_models()

    # get sinusoidal baseline at matched conditions
    labeled_df  = load_labeled()
    best_sine_R = get_sinusoidal_baseline(labeled_df)

    # run optimizer
    evaluated_z, evaluated_waveforms, evaluated_R, evaluated_std = \
        run_bayesian_optimization(vae, gpr_R, scaler, feature_names)

    # extract diverse proposals
    proposals = extract_proposals(
        evaluated_z, evaluated_waveforms,
        evaluated_R, evaluated_std
    )

    # plots
    plot_optimization_history(evaluated_R, best_sine_R)
    plot_proposals(proposals, best_sine_R)

    # save
    df_proposals = save_proposals(proposals, best_sine_R)

    # push results
    print("\nOptimization complete.")
    print("Run: git add results/optimizer/ && git commit -m "
          "'optimizer results' && git push")