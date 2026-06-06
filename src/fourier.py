import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ── Configuration ────────────────────────────────────────────────────────────
N_WAVEFORMS   = 10000   # number of synthetic waveforms to generate
N_POINTS      = 64      # time steps per waveform (one full oscillation cycle)
N_HARMONICS   = 5       # how many sine waves to superimpose
MAX_ACCEL     = 5.0     # acceleration filter threshold (normalized units)
RANDOM_SEED   = 42      # fixed seed for reproducibility
OUTPUT_PATH   = "synthetic/synthetic_waveforms.csv"

# ── Core Generator ───────────────────────────────────────────────────────────
def generate_waveform(rng, n_points=N_POINTS, n_harmonics=N_HARMONICS):
    """Generate a one non-sinusoidal waveform via Fourier superposition. Returns a normalized array of n_points wall velocity values"""
    t = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    waveform = np.zeros(n_points)

    for k in range(1, n_harmonics + 1):
        amplitude = rng.uniform(0, 1.0 / k)
        phase = rng.uniform(0, 2*np.pi)
        waveform += amplitude * np.sin(k * t + phase)

    # normalize so max absolute value = 1.0
    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        waveform /= max_val
    return waveform

def compute_acceleration(waveform, n_points=N_POINTS):
    """
    Compute peak acceleration of a waveform.
    Acceleration = rate of change of velocity (finite difference approximation).
    """
    dt = (2 * np.pi) / n_points
    acceleration = np.diff(waveform) / dt
    return np.max(np.abs(acceleration))


# ── Main Generation Loop ─────────────────────────────────────────────────────
def generate_dataset(
    n_waveforms=N_WAVEFORMS,
    max_accel=MAX_ACCEL,
    seed=RANDOM_SEED,
):
    rng = np.random.default_rng(seed)

    waveforms   = []
    rejected    = 0

    print(f"Generating {n_waveforms} synthetic waveforms...")

    while len(waveforms) < n_waveforms:
        w = generate_waveform(rng)

        # filter out physically unrealizable waveforms
        if compute_acceleration(w) > max_accel:
            rejected += 1
            continue

        waveforms.append(w)

    print(f"Done. Accepted: {len(waveforms)} | Rejected (accel filter): {rejected}")
    return np.array(waveforms)


# ── Save to CSV ──────────────────────────────────────────────────────────────
def save_dataset(waveforms, output_path=OUTPUT_PATH):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # column names: w_00, w_01, ... w_63
    col_names = [f"w_{i:02d}" for i in range(waveforms.shape[1])]
    df = pd.DataFrame(waveforms, columns=col_names)

    # metadata columns — all null for synthetic data
    df.insert(0, "source",          "synthetic")
    df.insert(1, "waveform_family", "fourier")
    df["amplitude_plus"] = None   # A+ — unknown for synthetic
    df["period_plus"]    = None   # T+ — unknown for synthetic
    df["reynolds"]       = None   # Re — unknown for synthetic
    df["DR_percent"]     = None   # drag reduction — unknown for synthetic
    df["net_power_pct"]  = None   # net power saving — unknown for synthetic

    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}  ({df.shape[0]} rows × {df.shape[1]} cols)")
    return df


# ── Quick Visual Check ───────────────────────────────────────────────────────
def plot_samples(waveforms, n_samples=9):
    """Plot a grid of random waveforms to visually verify variety."""
    fig, axes = plt.subplots(3, 3, figsize=(10, 7))
    indices = np.random.choice(len(waveforms), n_samples, replace=False)

    for ax, idx in zip(axes.flat, indices):
        ax.plot(waveforms[idx], color="#2d2d6e", linewidth=1.5)
        ax.axhline(0, color="#aaaaaa", linewidth=0.5, linestyle="--")
        ax.set_title(f"Waveform #{idx}", fontsize=9)
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlabel("Time step", fontsize=8)
        ax.set_ylabel("w (normalized)", fontsize=8)
        ax.tick_params(labelsize=7)

    plt.suptitle("Sample Synthetic Waveforms — Fourier Superposition", fontsize=11)
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/sample_waveforms.png", dpi=150)
    plt.show()
    print("Plot saved to results/sample_waveforms.png")


# ── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    waveforms = generate_dataset()
    df        = save_dataset(waveforms)
    plot_samples(waveforms)

    # quick sanity checks
    print("\n── Sanity Checks ──")
    print(f"Shape:               {waveforms.shape}")
    print(f"Max absolute value:  {np.max(np.abs(waveforms)):.4f}  (should be 1.0)")
    print(f"Min absolute value:  {np.min(np.abs(waveforms)):.4f}")
    print(f"Mean across dataset: {np.mean(waveforms):.4f}  (should be near 0)")
    print(f"CSV columns:         {list(df.columns[:5])} ... + metadata")