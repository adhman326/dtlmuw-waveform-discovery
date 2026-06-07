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
HIDDEN_DIM = 128 # neurons in hidden layers
BATCH_SIZE = 64 # waveforms per training batch
N_EPOCHS = 200 # training epochs
LR = 1e-3 # learning rate
BETA = 1.0 # KL divergence weight (1.0 = standard VAE)
VAL_SPLIT = 0.15 # fraction of VAE data used for validation
RANDOM_SEED = 42
SAVE_PATH = "models/vae.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Encoder ──────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    """
    Takes a 64-point waveform and compresses it to latent space.
    Outputs mu and log_var — the mean and variance of the latent distribution.
    """
    def __init__(self, n_points, hidden_dim, latent_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_points, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # two seperate output heads - one for mean, one for variance
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.log_var_layer = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.net(x)
        mu = self.mu_layer(h)
        log_var = self.log_var_layer(h)
        return mu, log_var
    
# ── Decoder ──────────────────────────────────────────────────────────────────
class Decoder(nn.Module):
    """
    Takes latent coordinates and reconstructs a 64-point waveform.
    Tanh output keeps values between -1 and 1, matching normalized waveforms.
    """
    def __init__(self, latent_dim, hidden_dim, n_points):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_points),
            nn.Tanh() #output between -1 and 1
        )

    def forward(self, z):
        return self.net(z)
    
# ── VAE ───────────────────────────────────────────────────────────────────────
class VAE(nn.Module):
    """
    Full VAE, encodes a waveform to latent space, samples a point, then decodes back to a waveform.
    """
    def __init__(self, n_points=N_POINTS, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = Encoder(n_points, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, n_points)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, log_var):
        """
        The reparameterization trick — allows gradients to flow through
        the random sampling step during training.
        Instead of sampling z directly, we sample noise and scale it.
        z = mu + std * noise
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std) # random noise
        return mu + std * eps
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var
    
    def decode(self, z):
        """
        Decode a latent vector directly - used during generation.
        """
        return self.decoder(z)
    
    def encode(self, x):
        """
        Encode a waveform to its latent coordinates - used for GPR."""
        mu, log_var = self.encoder(x)
        return mu # use mean as point estimate
    
# ── Loss Function ─────────────────────────────────────────────────────────────
def vae_loss(x, x_recon, mu, log_var, beta=BETA):
    """
    VAE loss = reconstruction loss + KL divergence
    Reconstruction loss: how different is the output from the input?
    KL divergence: how far is the latent distribution from a standard normal?
    This is what forces the latent space to be smooth.
    Beta controls the tradeoff — higher beta = smoother latent space
    but worse reconstruction.
    """
    #mean square error between input and reconstruction
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction = "mean")

    # KL divergence - analytical formula for Gaussian distributions
    kl_loss = -0.5 * torch.mean(1+ log_var - mu.pow(2) - log_var.exp())

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss

# ── Data Preparation ───────────────────────────────────────────────────────────
def prepare_data():
    """
    Load waveforms and create PyTorch DataLoaders.
    """
    labeled_df = load_labeled()
    waveforms = build_vae_dataset(labeled_df)

    # convert to PyTorch tensor
    X = torch.tensor(waveforms, dtype=torch.float32)
    print(f"\nFull VAE dataset: {X.shape}")

    # split into train and validation sets
    n_total = len(X)
    n_val = int(n_total * VAL_SPLIT)
    n_train = n_total - n_val

    dataset = TensorDataset(X)
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train: {n_train} | Val: {n_val}")
    return train_loader, val_loader

# ── Main Training Loop ─────────────────────────────────────────────────────────
def train(model, train_loader, val_loader, n_epochs=N_EPOCHS, lr=LR):
    optimizer = optim.Adam(model.parameters(), lr-lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    train_losses = []
    val_losses = []
    best_val = float("inf")

    print(f"\nTraining VAE for {n_epochs} epochs...")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} "
          f"{'Recon':>8} {'KL':>8}")
    print("-" * 50)

    for epoch in range(1, n_epochs + 1):

        # training
        model.train()
        total_train = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            x_recon, mu, log_var = model(batch)
            loss, recon, kl = vae_loss(batch, x_recon, mu, log_var)
            loss.backward()
            optimizer.step()
            total_train += loss.item()

        avg_train = total_train / len(train_loader)

        # validation
        model.eval()
        total_val = 0.0
        total_recon = 0.0
        total_kl = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                x_recon, mu, log_var = model(batch)
                loss, recon, kl = vae_loss(batch, x_recon, mu, log_var)
                total_val += loss.item()
                total_recon += recon.item()
                total_kl += kl.item()

        avg_val = total_val / len(val_loader)
        avg_recon = total_recon / len(val_loader)
        avg_kl = total_kl / len(val_loader)

        train_losses.append(avg_train)
        val_losses.append(avg_val)

        # print every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>6} {avg_train:>12.6f} {avg_val:>10.6f} "
                  f"{avg_recon:>8.6f} {avg_kl:>8.6f}")
            
        # save best model
        if avg_val < best_val:
            best_val = avg_val
            os.makerdirs("models", exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)

        scheduler.step(avg_val)

    print(f"\nBest validation loss: {best_val:.6f}")
    print(f"Model saved to {SAVE_PATH}")
    return train_losses, val_losses

# ── Plot Training Curves ──────────────────────────────────────────────────────
def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss", color="#2d2d6e")
    plt.plot(val_losses,   label="Val Loss",   color="#e06c75")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VAE Training — Loss Curves")
    plt.legend()
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/vae_loss_curve.png", dpi=150)
    plt.show()
    print("Loss curve saved to results/vae_loss_curve.png")

# ── Plot Reconstructions ──────────────────────────────────────────────────────
def plot_reconstructions(model, val_loader, n_samples=6):
    """
    Visual sanity check — plot original waveforms vs reconstructions.
    If these look similar the VAE is learning properly.
    """
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

    axes[0, 0].set_ylabel("Original", fontsize=9)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=9)
    plt.suptitle("VAE Reconstruction Quality", fontsize=11)
    plt.tight_layout()
    plt.savefig("results/vae_reconstructions.png", dpi=150)
    plt.show()
    print("Reconstruction plot saved to results/vae_reconstructions.png")

# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)

    # build model
    model = VAE(
        n_points = N_POINTS,
        hidden_dim = HIDDEN_DIM,
        latent_dim = LATENT_DIM,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"VAE parameters: {total_params:,}")

    # load data
    train_loader, val_loader = prepare_data()

    # train
    train_losses, val_losses = train(model, train_loader, val_loader)

    # load best model and evaluate
    model.load_state_dict(torch.load(SAVE_PATH))

    # plots
    plot_loss(train_losses, val_losses)
    plot_reconstructions(model, val_loader)

    print("\nVAE training complete.")
