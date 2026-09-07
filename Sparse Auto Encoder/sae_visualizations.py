"""
Training + visualization script for the SparseAutoencoder in sparse_autoencoder.py.

Produces four matplotlib figures, saved as PNGs:
  1. training_curves.png   - reconstruction loss, L1 loss, and total loss vs step
  2. sparsity.png          - mean L0 (active latents per sample) vs step, plus a
                             histogram of latent activation frequency at the end
  3. feature_recovery.png  - cosine similarity heatmap between learned dictionary
                             atoms and the ground-truth synthetic features, showing
                             how many true features the SAE actually recovered
  4. reconstruction_quality.png - scatter of true vs reconstructed activation
                             values on held-out data

Run: python3 sae_visualizations.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from sparse_autoencoder import SparseAutoencoder, make_synthetic_batch

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})

COLOR_RECON = "#2a78d6"
COLOR_L1 = "#eb6834"
COLOR_TOTAL = "#52514e"
COLOR_L0 = "#1baf7a"


def train_with_logging(d_model=64, n_latents=512, n_true_features=100, l1_coeff=3e-3,
                        steps=3000, batch_size=256, lr=1e-3, device="cpu", seed=0):
    torch.manual_seed(seed)
    sae = SparseAutoencoder(d_model, n_latents, l1_coeff).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    # fixed ground-truth features so recovery can be measured at the end
    true_dirs = F.normalize(torch.randn(n_true_features, d_model, device=device), dim=-1)

    history = {"step": [], "total": [], "recon": [], "l1": [], "l0": []}

    for step in range(1, steps + 1):
        coeffs = torch.rand(batch_size, n_true_features, device=device)
        mask = (torch.rand(batch_size, n_true_features, device=device) > 0.95).float()
        x = (coeffs * mask) @ true_dirs

        optimizer.zero_grad()
        loss, metrics = sae.loss(x)
        loss.backward()
        optimizer.step()
        sae._normalize_decoder()

        if step % 25 == 0:
            history["step"].append(step)
            history["total"].append(loss.item())
            history["recon"].append(metrics["recon_loss"])
            history["l1"].append(metrics["l1_loss"])
            history["l0"].append(metrics["l0"])

        if step % 500 == 0:
            sae.resample_dead_latents(x)

    return sae, true_dirs, history


def plot_training_curves(history, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(history["step"], history["recon"], color=COLOR_RECON, lw=1.8, label="reconstruction")
    ax1.plot(history["step"], history["l1"], color=COLOR_L1, lw=1.8, label="L1 (sparsity)")
    ax1.plot(history["step"], history["total"], color=COLOR_TOTAL, lw=1.2, ls="--", label="total")
    ax1.set_xlabel("training step")
    ax1.set_ylabel("loss")
    ax1.set_title("Loss components over training")
    ax1.legend(frameon=False)

    ax2.plot(history["step"], history["recon"], color=COLOR_RECON, lw=1.8)
    ax2.set_yscale("log")
    ax2.set_xlabel("training step")
    ax2.set_ylabel("reconstruction loss (log scale)")
    ax2.set_title("Reconstruction loss (log scale)")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sparsity(history, sae, x_eval, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(history["step"], history["l0"], color=COLOR_L0, lw=1.8)
    ax1.set_xlabel("training step")
    ax1.set_ylabel("mean active latents per sample (L0)")
    ax1.set_title("Sparsity over training")

    with torch.no_grad():
        h = sae.encode(x_eval)
        firing_freq = (h > 0).float().mean(dim=0).numpy()

    ax2.hist(np.log10(firing_freq + 1e-8), bins=40, color=COLOR_L0, alpha=0.8)
    ax2.set_xlabel("log10(activation frequency)")
    ax2.set_ylabel("number of latents")
    ax2.set_title("Latent firing-frequency distribution\n(left spike = dead/rare latents)")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_recovery(sae, true_dirs, path):
    with torch.no_grad():
        dict_atoms = F.normalize(sae.W_dec, dim=1)          # [n_latents, d_model]
        sims = dict_atoms @ true_dirs.t()                    # [n_latents, n_true_features]
        best_match = sims.max(dim=0).values.numpy()          # best learned atom per true feature
        n_true_features = true_dirs.shape[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    order = np.argsort(-best_match)
    ax1.bar(range(n_true_features), best_match[order], color=COLOR_RECON, width=1.0)
    ax1.axhline(0.9, color=COLOR_L1, lw=1, ls="--", label="0.9 similarity threshold")
    ax1.set_xlabel("ground-truth feature (sorted)")
    ax1.set_ylabel("cosine sim. to best-matching dictionary atom")
    ax1.set_title("Feature recovery")
    ax1.set_ylim(0, 1.02)
    ax1.legend(frameon=False)

    n_recovered = int((best_match > 0.9).sum())
    subset = sims[:60, :60].numpy()
    im = ax2.imshow(subset, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax2.set_xlabel("true feature index (first 60)")
    ax2.set_ylabel("dictionary atom index (first 60)")
    ax2.set_title(f"Similarity matrix\n{n_recovered}/{n_true_features} true features recovered (sim>0.9)")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="cosine similarity")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_reconstruction_quality(sae, x_eval, path):
    with torch.no_grad():
        x_hat, h = sae(x_eval)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    true_vals = x_eval.flatten().numpy()
    pred_vals = x_hat.flatten().numpy()
    idx = np.random.choice(len(true_vals), size=min(4000, len(true_vals)), replace=False)

    ax.scatter(true_vals[idx], pred_vals[idx], s=4, alpha=0.25, color=COLOR_RECON)
    lims = [min(true_vals.min(), pred_vals.min()), max(true_vals.max(), pred_vals.max())]
    ax.plot(lims, lims, color=COLOR_TOTAL, lw=1, ls="--", label="perfect reconstruction")
    ax.set_xlabel("true activation value")
    ax.set_ylabel("reconstructed activation value")
    r2 = 1 - np.sum((true_vals - pred_vals) ** 2) / np.sum((true_vals - true_vals.mean()) ** 2)
    ax.set_title(f"Reconstruction quality (R² = {r2:.3f})")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    sae, true_dirs, history = train_with_logging(steps=3000)

    x_eval, _ = make_synthetic_batch(d_model=64, n_true_features=100, batch_size=2000)

    plot_training_curves(history, "training_curves.png")
    plot_sparsity(history, sae, x_eval, "sparsity.png")
    plot_feature_recovery(sae, true_dirs, "feature_recovery.png")
    plot_reconstruction_quality(sae, x_eval, "reconstruction_quality.png")

    print("Saved: training_curves.png, sparsity.png, feature_recovery.png, reconstruction_quality.png")
