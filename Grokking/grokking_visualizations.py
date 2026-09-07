"""
Visualizations for the grokking run saved by grokking_experiment.py (grokking_run.pt).

Produces:
  1. grokking_curve.png       - train/test accuracy vs step (log x-axis), the
                                 classic grokking plot: train saturates almost
                                 immediately, test lags for ~10-15k steps then
                                 rises sharply.
  2. loss_curve.png           - train/test loss vs step (log-log), showing test
                                 loss falling long after train loss has hit zero.
  3. weight_spectrum_evolution.png
                               - singular value spectrum of W1 (input->hidden)
                                 at several checkpoints spanning the memorization
                                 -> grokking transition, overlaid on one plot.
  4. effective_rank_vs_test_acc.png
                               - effective rank (participation ratio of the
                                 singular value spectrum) of W1 over training,
                                 overlaid with test accuracy, showing the weight
                                 matrix simplifying as the model generalizes.

Run: python3 grokking_visualizations.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})

COLOR_TRAIN = "#2a78d6"
COLOR_TEST = "#eb6834"
COLOR_RANK = "#6250d6"


def load_run(path="grokking_run.pt"):
    data = torch.load(path, weights_only=False)
    return data["history"], data["checkpoints"]


def plot_grokking_curve(history, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history["step"], history["train_acc"], color=COLOR_TRAIN, lw=2, label="train accuracy")
    ax.plot(history["step"], history["test_acc"], color=COLOR_TEST, lw=2, label="test accuracy")
    ax.set_xscale("log")
    ax.set_xlabel("training step (log scale)")
    ax.set_ylabel("accuracy")
    ax.set_title("Grokking: train saturates immediately, test catches up much later")
    ax.legend(frameon=False)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_loss_curve(history, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history["step"], history["train_loss"], color=COLOR_TRAIN, lw=2, label="train loss")
    ax.plot(history["step"], history["test_loss"], color=COLOR_TEST, lw=2, label="test loss")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training step (log scale)")
    ax.set_ylabel("loss (log scale)")
    ax.set_title("Test loss keeps falling long after train loss hits ~0")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def singular_values(W: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        s = torch.linalg.svdvals(W)
    return s.numpy()


def effective_rank(s: np.ndarray) -> float:
    """Participation-ratio effective rank: (sum s_i^2)^2 / sum s_i^4.
    Equals the true rank for a flat spectrum, drops toward 1 as the
    spectrum concentrates onto a few dominant singular values."""
    s2 = s ** 2
    return float((s2.sum() ** 2) / (s2 ** 2).sum())


def plot_weight_spectrum_evolution(checkpoints, path, matrix_key="W1", n_snapshots=6):
    steps = sorted(checkpoints.keys())
    snap_steps = [steps[i] for i in np.linspace(0, len(steps) - 1, n_snapshots).astype(int)]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.cm.viridis
    for i, step in enumerate(snap_steps):
        s = singular_values(checkpoints[step][matrix_key])
        s_norm = s / s[0]  # normalize so curves are comparable in shape
        ax.plot(np.arange(1, len(s) + 1), s_norm, color=cmap(i / (n_snapshots - 1)),
                lw=1.8, label=f"step {step}")

    ax.set_xlabel("singular value index")
    ax.set_ylabel("singular value (normalized to largest)")
    ax.set_yscale("log")
    ax.set_title(f"{matrix_key} singular value spectrum barely shifts across training\n(memorization and grokking use a similarly broad spectrum)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_effective_rank_vs_test_acc(checkpoints, history, path, matrix_key="W1"):
    ckpt_steps = sorted(checkpoints.keys())
    eff_ranks = [effective_rank(singular_values(checkpoints[s][matrix_key])) for s in ckpt_steps]

    hist_steps = np.array(history["step"])
    hist_test_acc = np.array(history["test_acc"])
    # nearest matching test accuracy for each checkpoint step, for a shared x-axis
    matched_acc = [hist_test_acc[np.argmin(np.abs(hist_steps - s))] for s in ckpt_steps]

    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax2 = ax1.twinx()

    ax1.plot(ckpt_steps, eff_ranks, color=COLOR_RANK, lw=2, label=f"effective rank of {matrix_key}")
    ax1.set_xlabel("training step")
    ax1.set_ylabel("effective rank", color=COLOR_RANK)
    ax1.tick_params(axis="y", labelcolor=COLOR_RANK)

    ax2.plot(ckpt_steps, matched_acc, color=COLOR_TEST, lw=2, ls="--", label="test accuracy")
    ax2.set_ylabel("test accuracy", color=COLOR_TEST)
    ax2.tick_params(axis="y", labelcolor=COLOR_TEST)
    ax2.set_ylim(-0.02, 1.02)

    ax1.set_title(f"{matrix_key} effective rank dips then recovers through the transition\n(no clean rank collapse here — see notes)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    history, checkpoints = load_run("grokking_run.pt")

    plot_grokking_curve(history, "grokking_curve.png")
    plot_loss_curve(history, "loss_curve.png")
    plot_weight_spectrum_evolution(checkpoints, "weight_spectrum_evolution.png", matrix_key="W1")
    plot_effective_rank_vs_test_acc(checkpoints, history, "effective_rank_vs_test_acc.png", matrix_key="W1")

    print("Saved: grokking_curve.png, loss_curve.png, weight_spectrum_evolution.png, effective_rank_vs_test_acc.png")
