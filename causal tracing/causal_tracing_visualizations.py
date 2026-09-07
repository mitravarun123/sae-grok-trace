"""
Visualizations for causal_tracing.py.

Produces:
  1. training_curve.png     - loss/accuracy while the toy transformer memorizes
                               the subject/relation -> object lookup table
  2. causal_trace_heatmap.png
                               - the main result: (layer x position) grid of
                                 restoration scores, averaged over many facts.
                                 This is the classic ROME-style causal tracing
                                 figure.
  3. per_layer_importance.png
                               - for each layer, the best restoration score
                                 achieved at any position (which layers matter)
  4. per_position_importance.png
                               - for each position (subject/relation/SEP), the
                                 best restoration score at any layer (which
                                 token positions carry the causal information)

Run: python3 causal_tracing_visualizations.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from causal_tracing import build_task, ToyTransformer, causal_trace_averaged, SUBJECT, RELATION, SEP

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})

COLOR_LOSS = "#2a78d6"
COLOR_ACC = "#1baf7a"
COLOR_BAR = "#6250d6"

POSITION_LABELS = ["subject", "relation", "SEP"]


def train_model_with_logging(task, steps=800, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    model = ToyTransformer(task["vocab_size"], task["n_objects"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    tokens, labels = task["tokens"], task["labels"]
    history = {"step": [], "loss": [], "acc": []}
    for step in range(1, steps + 1):
        x = model.embed(tokens)
        logits = model(x)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 5 == 0 or step == 1:
            acc = (logits.argmax(-1) == labels).float().mean().item()
            history["step"].append(step)
            history["loss"].append(loss.item())
            history["acc"].append(acc)

    return model, history


def plot_training_curve(history, path):
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax2 = ax1.twinx()

    ax1.plot(history["step"], history["loss"], color=COLOR_LOSS, lw=2, label="loss")
    ax1.set_yscale("log")
    ax1.set_xlabel("training step")
    ax1.set_ylabel("loss (log scale)", color=COLOR_LOSS)
    ax1.tick_params(axis="y", labelcolor=COLOR_LOSS)
    ax1.grid(alpha=0.25)

    ax2.plot(history["step"], history["acc"], color=COLOR_ACC, lw=2, label="accuracy")
    ax2.set_ylabel("accuracy", color=COLOR_ACC)
    ax2.tick_params(axis="y", labelcolor=COLOR_ACC)
    ax2.set_ylim(-0.02, 1.05)

    ax1.set_title("Toy transformer memorizing the subject/relation -> object task")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_causal_trace_heatmap(scores, path, clean_prob, corrupt_prob, n_examples):
    n_layers, n_positions = scores.shape
    fig, ax = plt.subplots(figsize=(5.5, 5))

    im = ax.imshow(scores, cmap="magma", vmin=0, vmax=1, aspect="auto", origin="lower")
    ax.set_xticks(range(n_positions))
    ax.set_xticklabels(POSITION_LABELS)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"layer {l}" for l in range(n_layers)])
    ax.set_xlabel("token position (patched)")
    ax.set_ylabel("layer (patched)")
    ax.set_title(f"Causal tracing: probability restored by patching\n"
                 f"(avg over {n_examples} facts; clean={clean_prob:.2f}, corrupt={corrupt_prob:.2f})")

    for l in range(n_layers):
        for p in range(n_positions):
            val = scores[l, p]
            color = "white" if val < 0.6 else "black"
            ax.text(p, l, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="restoration score")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_layer_importance(scores, path):
    n_layers = scores.shape[0]
    best_per_layer = scores.max(axis=1)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(range(n_layers), best_per_layer, color=COLOR_BAR)
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f"layer {l}" for l in range(n_layers)])
    ax.set_ylabel("best restoration score (any position)")
    ax.set_title("Which layers carry the causally relevant information?")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_position_importance(scores, path):
    n_positions = scores.shape[1]
    best_per_position = scores.max(axis=0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(n_positions), best_per_position, color=["#eb6834", "#2a78d6", "#1baf7a"])
    ax.set_xticks(range(n_positions))
    ax.set_xticklabels(POSITION_LABELS)
    ax.set_ylabel("best restoration score (any layer)")
    ax.set_title("Which token positions carry the\ncausally relevant information?")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    task = build_task()
    model, history = train_model_with_logging(task)
    plot_training_curve(history, "ct_training_curve.png")

    result = causal_trace_averaged(model, task, n_examples=25)
    scores = result["scores"]

    plot_causal_trace_heatmap(scores, "causal_trace_heatmap.png",
                               result["clean_prob_mean"], result["corrupt_prob_mean"], result["n_examples"])
    plot_per_layer_importance(scores, "per_layer_importance.png")
    plot_per_position_importance(scores, "per_position_importance.png")

    print("Saved: ct_training_curve.png, causal_trace_heatmap.png, "
          "per_layer_importance.png, per_position_importance.png")
