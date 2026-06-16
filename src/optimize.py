import numpy as np
import torch
import matplotlib.pyplot as plt
import joblib
import os
import sys
import pandas as pd
from scipy.stats import norm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import load_labeled, VELOCITY_COLS
from vae import VAE, N_POINTS, HIDDEN_DIM, LATENT_DIM, SAVE_PATH

# ── Configuration ────────────────────────────────────────────────────────────
N_ITERATIONS    = 500    # number of BO steps
N_RANDOM_INIT   = 50     # random exploration before BO kicks in
N_PROPOSALS     = 10     # number of novel waveforms to propose at end
LATENT_BOUNDS   = 3.0    # search within ±3 std of latent space
RANDOM_SEED     = 42
RESULTS_DIR     = "results/optimizer"
device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# physical parameters to use alongside latent coords
# these are fixed to a representative operating condition
FIXED_AMPLITUDE = 4.5    # A+ — Center of Cimarelli range
FIXED_PERIOD    = 125.0  # T+ — matches Cimarelli optimal region
FIXED_REYNOLDS  = 180    # Re — matches Cimarelli


# ── Load Models ───────────────────────────────────────────────────────────────
def load_models():
    """Load VAE, GPR, and scaler."""
    # VAE
    vae = VAE(
        n_points   = N_POINTS,
        hidden_dim = HIDDEN_DIM,
        latent_dim = LATENT_DIM,
    ).to(device)
    vae.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    vae.eval()
    print(f"VAE loaded from {SAVE_PATH}")

    # GPR for R
    gpr_R = joblib.load("models/gpr_R.pkl")
    print(f"GPR (R) loaded")

    # scaler
    scaler = joblib.load("models/scaler.pkl")
    print(f"Scaler loaded")

    return vae, gpr_R, scaler


# ── Build Feature Vector ──────────────────────────────────────────────────────
def build_feature(z, scaler):
    """
    Build a single feature vector for the GPR from latent coordinates.
    Appends fixed physical parameters to the latent vector.
    """
    phys = np.array([
        FIXED_AMPLITUDE,  # amplitude_plus
        6.2518,           # amplitude_star median from training
        FIXED_PERIOD,     # period_plus
        0.0,              # wavenumber_plus
        FIXED_REYNOLDS,   # reynolds
        2 * np.pi / FIXED_PERIOD,  # omega_plus
    ], dtype=np.float32)

    x = phys.reshape(1, -1)
    x_scaled = scaler.transform(x)
    return x_scaled


# ── GPR Prediction ────────────────────────────────────────────────────────────
def predict_R(z, gpr_R, scaler):
    """
    Predict drag reduction R for a latent coordinate z.
    Returns mean prediction and uncertainty.
    """
    x = build_feature(z, scaler)
    mu, sigma = gpr_R.predict(x, return_std=True)
    return float(mu[0]), float(sigma[0])


# ── Expected Improvement Acquisition Function ─────────────────────────────────
def expected_improvement(mu, sigma, best_so_far, xi=0.01):
    """
    Expected Improvement acquisition function.
    Balances exploration (high sigma) and exploitation (high mu).
    xi controls exploration — higher xi = more exploration.
    Returns EI score — higher is better to evaluate next.
    """
    improvement = mu - best_so_far - xi
    Z           = improvement / (sigma + 1e-9)
    ei          = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-10] = 0.0
    return ei


# ── Random Latent Samples ─────────────────────────────────────────────────────
def random_latent_samples(n, latent_dim=LATENT_DIM, bounds=LATENT_BOUNDS):
    """Sample random points in latent space within bounds."""
    return np.random.uniform(
        -bounds, bounds, size=(n, latent_dim)
    ).astype(np.float32)


# ── Bayesian Optimization Loop ────────────────────────────────────────────────
def run_bayesian_optimization(vae, gpr_R, scaler):
    """
    Main BO loop. Searches latent space for coordinates that
    maximize predicted drag reduction R.
    """
    np.random.seed(RANDOM_SEED)

    # track all evaluated points
    evaluated_z  = []   # latent coordinates tried
    evaluated_R  = []   # predicted R at each point
    evaluated_std = []  # uncertainty at each point

    print(f"\nPhase 1: Random exploration ({N_RANDOM_INIT} points)...")

    # phase 1 — random exploration
    random_z = random_latent_samples(N_RANDOM_INIT)
    for i, z in enumerate(random_z):
        mu, sigma = predict_R(z, gpr_R, scaler)
        evaluated_z.append(z)
        evaluated_R.append(mu)
        evaluated_std.append(sigma)

    best_so_far = max(evaluated_R)
    print(f"Best R after random exploration: {best_so_far:.4f}")

    print(f"\nPhase 2: Bayesian optimization ({N_ITERATIONS} iterations)...")

    # phase 2 — BO loop
    for iteration in range(N_ITERATIONS):

        # generate candidate points to evaluate
        candidates = random_latent_samples(1000)

        # predict R and uncertainty for all candidates
        candidate_features = np.array([
            build_feature(z, scaler)[0] for z in candidates
        ])
        mu_all, sigma_all = gpr_R.predict(
            candidate_features, return_std=True
        )

        # compute expected improvement for each candidate
        ei = expected_improvement(
            mu_all, sigma_all, best_so_far
        )

        # pick candidate with highest EI
        best_idx = np.argmax(ei)
        next_z   = candidates[best_idx]

        # evaluate it
        mu, sigma = predict_R(next_z, gpr_R, scaler)
        evaluated_z.append(next_z)
        evaluated_R.append(mu)
        evaluated_std.append(sigma)

        # update best
        if mu > best_so_far:
            best_so_far = mu

        # print progress every 100 iterations
        if (iteration + 1) % 100 == 0:
            print(f"  Iteration {iteration+1:4d} | "
                  f"Best R so far: {best_so_far:.4f} | "
                  f"Current R: {mu:.4f} ± {sigma:.4f}")

    print(f"\nOptimization complete. Best predicted R: {best_so_far:.4f}")
    return (np.array(evaluated_z),
            np.array(evaluated_R),
            np.array(evaluated_std))


# ── Get Sinusoidal Baseline ───────────────────────────────────────────────────
def get_sinusoidal_baseline(labeled_df):
    """
    Find best sinusoidal R at the same operating conditions
    used by the optimizer — not the global maximum.
    """
    sine_rows = labeled_df[
        (labeled_df["waveform_family"] == "sine") &
        (labeled_df["amplitude_plus"].between(5, 9)) &
        (labeled_df["period_plus"].between(100, 150))
    ]["R"].dropna()

    if len(sine_rows) > 0:
        best_sine_R = sine_rows.max()
        print(f"Best sinusoidal R at matched conditions "
              f"(A+=5-9, T+=100-150): {best_sine_R:.4f}")
    else:
        # fall back to global max with a warning
        best_sine_R = labeled_df[
            labeled_df["waveform_family"] == "sine"
        ]["R"].dropna().max()
        print(f"Warning: no sine rows at matched conditions. "
              f"Using global max: {best_sine_R:.4f}")

    return best_sine_R

# ── Extract Top Proposals ─────────────────────────────────────────────────────
def extract_proposals(evaluated_z, evaluated_R, evaluated_std,
                      n_proposals=N_PROPOSALS):
    """
    Extract the top N latent coordinates by predicted R.
    Apply diversity constraint — proposals must be at least
    0.5 units apart in latent space to avoid clustered results.
    """
    # sort by predicted R descending
    sorted_idx = np.argsort(evaluated_R)[::-1]

    selected   = []
    selected_z = []

    for idx in sorted_idx:
        z   = evaluated_z[idx]
        r   = evaluated_R[idx]
        std = evaluated_std[idx]

        # diversity check — skip if too close to already selected point
        too_close = False
        for z_sel in selected_z:
            dist = np.linalg.norm(z - z_sel)
            if dist < 0.5:
                too_close = True
                break

        if not too_close:
            selected.append({
                "latent_coords": z,
                "predicted_R":   r,
                "uncertainty":   std,
            })
            selected_z.append(z)

        if len(selected) >= n_proposals:
            break

    print(f"\nSelected {len(selected)} diverse proposals")
    return selected


# ── Decode Proposals to Waveforms ─────────────────────────────────────────────
def decode_proposals(vae, proposals):
    """
    Decode latent coordinates back to waveform shapes.
    Adds the decoded waveform to each proposal dict.
    """
    for i, prop in enumerate(proposals):
        z      = torch.tensor(
            prop["latent_coords"], dtype=torch.float32
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            waveform = vae.decode(z).cpu().numpy()[0]

        prop["waveform"] = waveform
        print(f"Proposal {i+1:2d}: predicted R = {prop['predicted_R']:.4f} "
              f"± {prop['uncertainty']:.4f}")

    return proposals


# ── Plot Optimization History ─────────────────────────────────────────────────
def plot_optimization_history(evaluated_R, best_sine_R):
    """
    Plot predicted R over optimization iterations.
    Shows how the optimizer improves over time.
    """
    # running best
    running_best = np.maximum.accumulate(evaluated_R)

    plt.figure(figsize=(10, 4))
    plt.plot(evaluated_R,   alpha=0.3, color="#2d2d6e",
             linewidth=0.8, label="Evaluated R")
    plt.plot(running_best,  color="#2d2d6e",
             linewidth=2.0, label="Best R so far")
    plt.axhline(best_sine_R, color="#e06c75", linewidth=1.5,
                linestyle="--", label=f"Best sinusoidal R = {best_sine_R:.4f}")
    plt.axvline(50, color="gray", linewidth=0.8, linestyle=":",
                label="BO starts (after random init)")

    plt.xlabel("Evaluation number")
    plt.ylabel("Predicted R")
    plt.title("Bayesian Optimization — Drag Reduction Search")
    plt.legend()
    plt.tight_layout()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(f"{RESULTS_DIR}/optimization_history.png", dpi=150)
    plt.show()
    print(f"Optimization history saved")


# ── Plot Proposed Waveforms ───────────────────────────────────────────────────
def plot_proposals(proposals, best_sine_R):
    """
    Plot all proposed novel waveforms with their predicted R values.
    Also shows a reference sine wave for comparison.
    """
    n = len(proposals)
    cols = 5
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 3, rows * 2.5))
    axes = axes.flatten() if n > 1 else [axes]

    # reference sine wave
    t_ref   = np.linspace(0, 1, N_POINTS, endpoint=False)
    sine_ref = np.sin(2 * np.pi * t_ref)

    for i, prop in enumerate(proposals):
        ax = axes[i]
        ax.plot(prop["waveform"], color="#2d2d6e",
                linewidth=1.5, label="Proposed")
        ax.plot(sine_ref, color="#e06c75", linewidth=1.0,
                linestyle="--", alpha=0.5, label="Sine ref")
        ax.set_title(
            f"Proposal {i+1}\n"
            f"R = {prop['predicted_R']:.4f} ± {prop['uncertainty']:.4f}",
            fontsize=8
        )
        ax.set_ylim(-1.3, 1.3)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=6)

    # hide unused subplots
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        f"Novel Waveform Proposals\n"
        f"Best sinusoidal baseline R = {best_sine_R:.4f}",
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/proposed_waveforms.png", dpi=150)
    plt.show()
    print(f"Proposed waveforms plot saved")


# ── Save Proposals to CSV ─────────────────────────────────────────────────────
def save_proposals(proposals, best_sine_R):
    """
    Save proposed waveforms to CSV for future DNS validation.
    Each row is one proposed waveform with its predicted R and
    waveform shape as 64 velocity values.
    """
    rows = []
    for i, prop in enumerate(proposals):
        row = {
            "proposal_id":   i + 1,
            "predicted_R":   prop["predicted_R"],
            "uncertainty":   prop["uncertainty"],
            "beats_baseline": prop["predicted_R"] > best_sine_R,
        }
        # add waveform values
        for j, val in enumerate(prop["waveform"]):
            row[f"w_{j:02d}"] = val

        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = f"{RESULTS_DIR}/proposed_waveforms.csv"
    df.to_csv(path, index=False)

    print(f"\nProposed waveforms saved to {path}")
    print(f"\n── Summary ─────────────────────────────────────")
    print(f"Best sinusoidal R in dataset:  {best_sine_R:.4f}")
    print(f"Best proposed R:               "
          f"{max(p['predicted_R'] for p in proposals):.4f}")
    beats = sum(1 for p in proposals
                if p["predicted_R"] > best_sine_R)
    print(f"Proposals beating baseline:    {beats} / {len(proposals)}")
    print(f"────────────────────────────────────────────────")
    return df


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # load everything
    vae, gpr_R, scaler = load_models()

    # get sinusoidal baseline
    labeled_df  = load_labeled()
    best_sine_R = get_sinusoidal_baseline(labeled_df)

    # run optimizer
    evaluated_z, evaluated_R, evaluated_std = run_bayesian_optimization(
        vae, gpr_R, scaler
    )

    # extract diverse top proposals
    proposals = extract_proposals(evaluated_z, evaluated_R, evaluated_std)

    # decode to waveform shapes
    proposals = decode_proposals(vae, proposals)

    # plots
    plot_optimization_history(evaluated_R, best_sine_R)
    plot_proposals(proposals, best_sine_R)

    # save proposals
    df_proposals = save_proposals(proposals, best_sine_R)

    print("\nOptimization complete.")
    print("Next step: validate proposed waveforms with DNS simulation.")