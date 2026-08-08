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
N_ITERATIONS   = 1000   # doubled from 500
N_RANDOM_INIT  = 100    # doubled from 50
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

# Confidence bar for a proposal to be reportable. Derived empirically: an
# audit of the global BO search found predictive std is sharply bimodal —
# ~1093/1100 evaluated candidates sit at the far-field ceiling (~0.1034,
# i.e. the GPR has no nearby training data), while only a handful of points
# near the known training waveform shapes score well below that. 0.06 sits
# in the gap between those two populations.
CONFIDENCE_STD_THRESHOLD = 0.06
N_ANCHOR_ROUNDS   = 5    # shrinking-radius local search rounds per anchor
N_ANCHOR_PER_ROUND = 200 # candidates sampled per round
ANCHOR_INIT_SIGMA  = 0.5 # initial latent-space search radius around anchor

# Minimum diversity distance between selected proposals, measured in shape-
# DESCRIPTOR space (skewness, kurtosis, zero_crossing_rate, max_acceleration,
# each standardized by their std across the confident candidate pool) — NOT
# raw latent-vector distance. Two different latent codes can decode to
# near-identical waveforms (the decoder is a highly nonlinear, effectively
# many-to-one map), so latent distance doesn't guarantee the proposals are
# actually different shapes; descriptor distance does.
MIN_DESCRIPTOR_DIST = 1.0


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
    # NOTE: this is a single GPR fit from surrogate.py's main (non-cross-
    # validated) 80/10/10 split — not an ensemble, and not one of the
    # cross-validation folds computed alongside it (5-seed pooled CV,
    # LOOCV). Every proposal below inherits whatever fold-to-fold
    # instability those CV numbers quantify for this specific fit; a
    # different train/test split of the same 24 labeled rows could train
    # a meaningfully different GPR_R and shift every proposal in this run.
    # This is an accepted architectural tradeoff (the optimizer needs one
    # deployed model, not an ensemble) — flagged here so it isn't silently
    # forgotten when reading proposal_R values as if they were exact.
    print("NOTE: gpr_R.pkl is a single fit on one 80/10/10 split, not "
          "cross-validated — see surrogate.py's pooled-CV/LOOCV R² for "
          "how much this specific fit could vary under a different split.")
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


# ── Anchor Latents From Known Training Families ───────────────────────────────
def get_family_anchor_latents(vae, labeled_df):
    """
    Encode the mean VAE latent position of each known R-labeled waveform
    family at the fixed A+=4.5 condition (sine, square, revsawtooth,
    doublepeak, asymmetric). These are the only regions of latent space
    the GPR has training data near — global random search only stumbled
    into one of them in 1100 evaluated points, so local search is seeded
    here directly instead of hoping random exploration finds them by luck.
    """
    amp_mask = (
        (labeled_df["amplitude_plus"].between(4.0, 5.0)) |
        (labeled_df["amplitude_plus"].isna() &
         labeled_df["amplitude_star"].between(4.0, 5.0))
    )
    has_wave = labeled_df[VELOCITY_COLS].notnull().all(axis=1)
    df_sub = labeled_df[amp_mask & has_wave & labeled_df["R"].notnull()]

    anchors = {}
    print("\nFamily anchor latents (from known R-labeled training shapes):")
    for family, group in df_sub.groupby("waveform_family"):
        waveforms = group[VELOCITY_COLS].values.astype(np.float32)
        w_tensor  = torch.tensor(waveforms, dtype=torch.float32).to(device)
        with torch.no_grad():
            mu = vae.encode(w_tensor).cpu().numpy()
        z0 = mu.mean(axis=0).astype(np.float32)
        anchors[family] = z0
        print(f"  '{family}': {len(group)} rows -> latent mean "
              f"(||z||={np.linalg.norm(z0):.3f})")
    return anchors


# ── Local Search Around an Anchor ─────────────────────────────────────────────
def local_search_around_anchor(vae, gpr_R, scaler, feature_names, z0, seed,
                                n_rounds=N_ANCHOR_ROUNDS,
                                n_per_round=N_ANCHOR_PER_ROUND,
                                init_sigma=ANCHOR_INIT_SIGMA):
    """
    Shrinking-radius local search seeded at a known-data anchor. Every round
    samples candidates from N(z0, sigma) — centered on the ORIGINAL anchor,
    never recentered on the running best — and only sigma shrinks each
    round. This is a genuine trust-region refinement: it zooms in around
    z0 to locate its local peak precisely, but cannot walk away from it.

    (An earlier version of this function recentered on the best candidate
    found each round, which turned it into an unconstrained greedy
    hill-climb: every anchor — sine, square, revsawtooth, doublepeak,
    asymmetric — drifted to the same single global optimum, defeating the
    entire point of anchoring per-family. Fixed centering is what actually
    keeps each anchor's search local to its own family's neighborhood.)

    `seed` is required (not defaulted/omitted) — an earlier version used
    `np.random.default_rng()` unseeded, which draws from OS entropy and
    ignores the script's RANDOM_SEED/np.random.seed(42) entirely (Generator
    objects are deliberately independent of the legacy global seed), so
    each anchor's result silently changed between runs of identical code.

    Returns (evaluated list of (z, mu, sigma), best_z, best_mu, best_sigma).
    """
    rng = np.random.default_rng(seed)
    evaluated = []
    best_z, best_mu, best_sigma = z0, None, None
    sigma = init_sigma

    for _ in range(n_rounds):
        candidates = (z0 +
                      rng.normal(0, sigma, size=(n_per_round, len(z0)))
                      ).astype(np.float32)
        feats = np.array([
            build_feature(decode_latent(vae, z), scaler, feature_names)[0]
            for z in candidates
        ])
        mu_all, sigma_all = gpr_R.predict(feats, return_std=True)
        for z, mu, s in zip(candidates, mu_all, sigma_all):
            evaluated.append((z, float(mu), float(s)))

        round_best_idx = int(np.argmax(mu_all))
        if best_mu is None or mu_all[round_best_idx] > best_mu:
            best_z     = candidates[round_best_idx]
            best_mu    = float(mu_all[round_best_idx])
            best_sigma = float(sigma_all[round_best_idx])

        sigma *= 0.5  # zoom in around z0, not around the running best

    return evaluated, best_z, best_mu, best_sigma


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
        ei = expected_improvement(mu_all, sigma_all, best_so_far, xi=0.001) 
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
                      n_proposals=N_PROPOSALS,
                      max_std=CONFIDENCE_STD_THRESHOLD,
                      min_desc_dist=MIN_DESCRIPTOR_DIST):
    """
    Extract top N proposals by predicted R with a diversity constraint,
    restricted to candidates the GPR is actually confident about
    (std < max_std). Does NOT backfill with low-confidence candidates to
    force the count up to n_proposals — a proposal only gets reported if
    the model has real basis for the prediction.

    Diversity is enforced in normalized shape-descriptor space, not raw
    latent distance — see MIN_DESCRIPTOR_DIST for why.
    """
    evaluated_std = np.asarray(evaluated_std)
    evaluated_R   = np.asarray(evaluated_R)

    confident_idx = np.where(evaluated_std < max_std)[0]
    print(f"\n{len(confident_idx)} / {len(evaluated_std)} evaluated "
          f"candidates clear the confidence bar (std < {max_std})")

    # precompute shape descriptors for every confident candidate, then
    # standardize each dimension by its std across that pool so no single
    # descriptor (e.g. max_acceleration, which spans ~0-100) dominates the
    # distance calculation
    descriptors = {
        idx: compute_shape_descriptors(evaluated_waveforms[idx])
        for idx in confident_idx
    }
    if len(descriptors) > 0:
        desc_matrix = np.array(list(descriptors.values()))
        desc_std    = desc_matrix.std(axis=0)
        desc_std[desc_std < 1e-8] = 1.0  # constant dims: don't divide by 0
    normalized = {idx: np.array(d) / desc_std for idx, d in descriptors.items()}

    sorted_idx = confident_idx[np.argsort(evaluated_R[confident_idx])[::-1]]
    selected   = []
    selected_desc = []

    for idx in sorted_idx:
        z        = evaluated_z[idx]
        r        = evaluated_R[idx]
        std      = evaluated_std[idx]
        waveform = evaluated_waveforms[idx]
        desc_n   = normalized[idx]

        # diversity check — in standardized shape-descriptor space
        too_close = any(
            np.linalg.norm(desc_n - d_sel) < min_desc_dist
            for d_sel in selected_desc
        )

        if not too_close:
            sk, ku, zc, ac = descriptors[idx]
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
            selected_desc.append(desc_n)

        if len(selected) >= n_proposals:
            break

    if len(selected) < n_proposals:
        print(f"NOTE: only {len(selected)} confident, sufficiently diverse "
              f"proposal(s) found (requested up to {n_proposals}). Not "
              f"backfilling with low-confidence candidates to pad the count.")

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
    n = len(proposals)
    if n == 0:
        print("No confident proposals to plot — skipping proposed_waveforms.png")
        return
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
    if len(proposals) == 0:
        print("\nNo proposals cleared the confidence bar — nothing to save.")
        print("This means no region of latent space searched (global BO + "
              "anchor-seeded local search) landed close enough to the "
              "labeled training data for the GPR to make a confident "
              "prediction. Consider more DNS-labeled shapes, or a lower "
              "CONFIDENCE_STD_THRESHOLD if a looser bar is acceptable.")
        return pd.DataFrame()

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

    # run optimizer — global search (random init + greedy EI)
    bo_z, bo_waveforms, bo_R, bo_std = \
        run_bayesian_optimization(vae, gpr_R, scaler, feature_names)

    # Phase 3: anchor-seeded local search near each known training family.
    # The global search above only reliably finds ONE confident region by
    # luck (see PIPELINE_CHANGELOG.md) because the training data is 5
    # discrete waveform families in a continuous latent space, not a
    # continuum BO can smoothly climb toward. Seed local refinement
    # directly at each known family's encoded position instead.
    print(f"\nPhase 3: Anchor-seeded local search near known training "
          f"families...")
    anchors = get_family_anchor_latents(vae, labeled_df)
    anchor_z, anchor_R, anchor_std = [], [], []
    # anchors.items() order is deterministic (pandas groupby sorts keys by
    # default), so RANDOM_SEED + i gives each family a fixed, distinct,
    # reproducible seed rather than an unseeded/OS-entropy RNG
    for i, (family, z0) in enumerate(anchors.items()):
        evaluated, best_z, best_mu, best_sigma = local_search_around_anchor(
            vae, gpr_R, scaler, feature_names, z0, seed=RANDOM_SEED + i
        )
        for z, mu, s in evaluated:
            anchor_z.append(z)
            anchor_R.append(mu)
            anchor_std.append(s)
        print(f"  '{family}' anchor best: R={best_mu:.4f} ± {best_sigma:.4f}")

    anchor_waveforms = [decode_latent(vae, z) for z in anchor_z]

    # merge global + anchor-seeded evaluations into one candidate pool —
    # for proposal extraction ONLY. Kept separate from bo_R/bo_z/etc. above
    # because the anchor-search points are not part of the Bayesian
    # optimization's running-best trace: they come from 5 independent
    # fixed-center local searches (see local_search_around_anchor), not
    # from iterative EI-guided improvement, so plotting them on the same
    # "evaluation number" axis as the BO history would misrepresent 5000
    # local-refinement samples as 5000 more BO iterations.
    all_z         = np.concatenate([bo_z, np.array(anchor_z)], axis=0)
    all_waveforms = bo_waveforms + anchor_waveforms
    all_R         = np.concatenate([bo_R, np.array(anchor_R)])
    all_std       = np.concatenate([bo_std, np.array(anchor_std)])

    # extract diverse proposals — confidence-gated, does not pad with
    # low-confidence candidates to force a fixed count (see extract_proposals)
    proposals = extract_proposals(
        all_z, all_waveforms,
        all_R, all_std
    )

    # plots — history plot shows ONLY the true BO trace (bo_R), not the
    # anchor-seeded local search, so "Evaluation number" on the x-axis
    # means what it says
    plot_optimization_history(bo_R, best_sine_R)
    plot_proposals(proposals, best_sine_R)

    # save
    df_proposals = save_proposals(proposals, best_sine_R)

    # push results
    print("\nOptimization complete.")
    print("Run: git add results/optimizer/ && git commit -m "
          "'optimizer results' && git push")