import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import os
import sys

# add src to path to import dataset
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import load_labeled, build_vae_dataset

# ── Configuration ────────────────────────────────────────────────────────────
N_POINTS = 64 # waveform length
LATENT_DIM = 8 # latent space size - will ablate later
HIDDEN_DIM = 256 # neurons in hidden layers
BATCH_SIZE = 128 # waveforms per training batch
N_EPOCHS = 500 # training epochs
LR = 1e-3 # learning rate
BETA_START = 0.0 # KL divergence weight at start
BETA_END = 0.001 # KL divergence weight at end
WARMUP = 100 # epochs before KL kicks in
VAL_SPLIT = 0.15 # fraction of VAE data used for validation
RANDOM_SEED = 42
SAVE_PATH = "models/vae.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Encoder ──────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, n_points, hidden_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_points, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        )
        self.mu_layer      = nn.Linear(hidden_dim // 2, latent_dim)
        self.log_var_layer = nn.Linear(hidden_dim // 2, latent_dim)

        # initialize log_var to output small values — prevents early collapse
        nn.init.constant_(self.log_var_layer.bias, -2.0)
        nn.init.xavier_uniform_(self.log_var_layer.weight)

    def forward(self, x):
        h       = self.net(x)
        mu      = self.mu_layer(h)
        log_var = self.log_var_layer(h)
        # clamp log_var to prevent numerical instability
        log_var = torch.clamp(log_var, -10, 2)
        return mu, log_var


# ── Decoder ──────────────────────────────────────────────────────────────────
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_points):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_points),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


# ── VAE ───────────────────────────────────────────────────────────────────────
class VAE(nn.Module):
    def __init__(self, n_points=N_POINTS, hidden_dim=HIDDEN_DIM,
                 latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder    = Encoder(n_points, hidden_dim, latent_dim)
        self.decoder    = Decoder(latent_dim, hidden_dim, n_points)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z           = self.reparameterize(mu, log_var)
        x_recon     = self.decoder(z)
        return x_recon, mu, log_var

    def decode(self, z):
        return self.decoder(z)

    def encode(self, x):
        mu, _ = self.encoder(x)
        return mu


# ── Loss ──────────────────────────────────────────────────────────────────────
def vae_loss(x, x_recon, mu, log_var, beta=0.0):
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction="mean")
    kl_loss    = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


# ── Data Preparation ──────────────────────────────────────────────────────────
def prepare_data():
    labeled_df = load_labeled()
    waveforms  = build_vae_dataset(labeled_df)

    X = torch.tensor(waveforms, dtype=torch.float32)
    print(f"\nFull VAE dataset: {X.shape}")

    n_total = len(X)
    n_val   = int(n_total * VAL_SPLIT)
    n_train = n_total - n_val

    dataset            = TensorDataset(X)
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train: {n_train} | Val: {n_val}")
    return train_loader, val_loader


# ── Training ──────────────────────────────────────────────────────────────────
def train(model, train_loader, val_loader):
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=1e-5
    )

    train_losses  = []
    val_losses    = []
    recon_losses  = []
    kl_losses     = []
    best_val      = float("inf")

    print(f"\nTraining VAE for {N_EPOCHS} epochs...")
    print(f"{'Epoch':>6} {'Beta':>8} {'Train':>10} {'Val':>10} "
          f"{'Recon':>10} {'KL':>10}")
    print("-" * 60)

    for epoch in range(1, N_EPOCHS + 1):

        # KL annealing — zero for first WARMUP epochs, then linear ramp
        if epoch <= WARMUP:
            current_beta = 0.0
        else:
            progress     = (epoch - WARMUP) / (N_EPOCHS - WARMUP)
            current_beta = BETA_START + (BETA_END - BETA_START) * progress

        # training pass
        model.train()
        total_train = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            x_recon, mu, log_var = model(batch)
            loss, recon, kl      = vae_loss(batch, x_recon, mu, log_var,
                                            beta=current_beta)
            loss.backward()
            # gradient clipping — prevents exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train += loss.item()

        avg_train = total_train / len(train_loader)

        # validation pass
        model.eval()
        total_val   = 0.0
        total_recon = 0.0
        total_kl    = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                x_recon, mu, log_var = model(batch)
                loss, recon, kl      = vae_loss(batch, x_recon, mu, log_var,
                                                beta=current_beta)
                total_val   += loss.item()
                total_recon += recon.item()
                total_kl    += kl.item()

        avg_val   = total_val   / len(val_loader)
        avg_recon = total_recon / len(val_loader)
        avg_kl    = total_kl    / len(val_loader)

        train_losses.append(avg_train)
        val_losses.append(avg_val)
        recon_losses.append(avg_recon)
        kl_losses.append(avg_kl)

        if epoch % 25 == 0 or epoch == 1:
            print(f"{epoch:>6} {current_beta:>8.5f} {avg_train:>10.6f} "
                  f"{avg_val:>10.6f} {avg_recon:>10.6f} {avg_kl:>10.6f}")

        if avg_val < best_val:
            best_val = avg_val
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)

        scheduler.step()

    print(f"\nBest validation loss: {best_val:.6f}")
    print(f"Model saved to {SAVE_PATH}")
    return train_losses, val_losses, recon_losses, kl_losses


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_loss(train_losses, val_losses, recon_losses, kl_losses):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_losses, label="Train Loss", color="#2d2d6e")
    axes[0].plot(val_losses,   label="Val Loss",   color="#e06c75")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total Loss")
    axes[0].set_title("Total Loss")
    axes[0].legend()

    axes[1].plot(recon_losses, label="Recon Loss", color="#2d2d6e")
    axes[1].plot(kl_losses,    label="KL Loss",    color="#e06c75")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Reconstruction vs KL Loss")
    axes[1].legend()

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/vae_loss_curve.png", dpi=150)
    plt.show()
    print("Loss curve saved to results/vae_loss_curve.png")


def plot_reconstructions(model, val_loader, n_samples=6):
    model.eval()
    batch = next(iter(val_loader))[0][:n_samples].to(device)

    with torch.no_grad():
        recon, _, _ = model(batch)

    batch = batch.cpu().numpy()
    recon = recon.cpu().numpy()

    fig, axes = plt.subplots(2, n_samples, figsize=(14, 5))
    for i in range(n_samples):
        axes[0, i].plot(batch[i], color="#2d2d6e", linewidth=1.5)
        axes[0, i].set_title(f"Original {i+1}", fontsize=8)
        axes[0, i].set_ylim(-1.2, 1.2)
        axes[0, i].axis("off")

        axes[1, i].plot(recon[i], color="#e06c75", linewidth=1.5)
        axes[1, i].set_title(f"Reconstructed {i+1}", fontsize=8)
        axes[1, i].set_ylim(-1.2, 1.2)
        axes[1, i].axis("off")

    plt.suptitle("VAE Reconstruction Quality", fontsize=11)
    plt.tight_layout()
    plt.savefig("results/vae_reconstructions.png", dpi=150)
    plt.show()
    print("Reconstruction plot saved to results/vae_reconstructions.png")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)

    model = VAE(
        n_points   = N_POINTS,
        hidden_dim = HIDDEN_DIM,
        latent_dim = LATENT_DIM,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"VAE parameters: {total_params:,}")

    train_loader, val_loader = prepare_data()

    train_losses, val_losses, recon_losses, kl_losses = train(
        model, train_loader, val_loader
    )

    model.load_state_dict(torch.load(SAVE_PATH))
    plot_loss(train_losses, val_losses, recon_losses, kl_losses)
    plot_reconstructions(model, val_loader)

    print("\nVAE training complete.")
