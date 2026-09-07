"""
Grokking reproduction: modular addition (a + b) mod p, classified with a small
2-layer MLP trained on one-hot inputs (Gromov-style setup — this architecture
has known closed-form Fourier solutions, which is why its weight spectrum is
worth tracking).

Model:  x = concat(onehot(a), onehot(b))  in R^(2p)
        h = ReLU(x @ W1 + b1)             in R^hidden
        y = h @ W2 + b2                   in R^p   (logits over the p classes)

Grokking setup (Power et al. 2022): train on a small fraction of all (a, b)
pairs, use AdamW with nontrivial weight decay, and train far past the point
where training accuracy hits 100%. Weight decay is what drives the network
from a memorizing solution to a generalizing one long after the training
loss has flatlined.

At regular checkpoints we snapshot W1 and W2 so a separate script can analyze
how their singular-value spectra evolve across the memorization -> grokking
transition (that's the "weight spectrum" half of the project).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GrokkingMLP(nn.Module):
    def __init__(self, p: int, hidden_dim: int = 256):
        super().__init__()
        self.p = p
        self.W1 = nn.Linear(2 * p, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, p)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        onehot_a = F.one_hot(a, self.p).float()
        onehot_b = F.one_hot(b, self.p).float()
        x = torch.cat([onehot_a, onehot_b], dim=-1)
        h = F.relu(self.W1(x))
        return self.W2(h)


def make_dataset(p: int, train_frac: float, seed: int = 0, device="cpu"):
    """All p^2 pairs (a, b), label = (a + b) mod p, split into train/test."""
    g = torch.Generator().manual_seed(seed)
    a_vals, b_vals = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
    a_vals, b_vals = a_vals.flatten(), b_vals.flatten()
    labels = (a_vals + b_vals) % p

    n = p * p
    perm = torch.randperm(n, generator=g)
    n_train = int(train_frac * n)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    def to(idx):
        return a_vals[idx].to(device), b_vals[idx].to(device), labels[idx].to(device)

    return to(train_idx), to(test_idx)


@torch.no_grad()
def accuracy(model, a, b, labels):
    logits = model(a, b)
    return (logits.argmax(dim=-1) == labels).float().mean().item()


def train(p=59, train_frac=0.4, hidden_dim=128, weight_decay=1.0, lr=1e-3,
          steps=15000, log_every=25, checkpoint_every=250, device="cpu", seed=0):
    torch.manual_seed(seed)
    (a_tr, b_tr, y_tr), (a_te, b_te, y_te) = make_dataset(p, train_frac, seed, device)

    model = GrokkingMLP(p, hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))

    history = {"step": [], "train_acc": [], "test_acc": [], "train_loss": [], "test_loss": []}
    checkpoints = {}  # step -> {"W1": tensor, "W2": tensor}

    for step in range(1, steps + 1):
        model.train()
        logits = model(a_tr, b_tr)
        loss = F.cross_entropy(logits, y_tr)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                test_logits = model(a_te, b_te)
                test_loss = F.cross_entropy(test_logits, y_te)
            history["step"].append(step)
            history["train_loss"].append(loss.item())
            history["test_loss"].append(test_loss.item())
            history["train_acc"].append(accuracy(model, a_tr, b_tr, y_tr))
            history["test_acc"].append(accuracy(model, a_te, b_te, y_te))

        if step % checkpoint_every == 0 or step == 1:
            checkpoints[step] = {
                "W1": model.W1.weight.detach().clone(),
                "W2": model.W2.weight.detach().clone(),
            }

        if step % 1000 == 0:
            print(f"step {step:6d} | train_acc {history['train_acc'][-1]:.3f} | "
                  f"test_acc {history['test_acc'][-1]:.3f} | "
                  f"train_loss {loss.item():.4f} | test_loss {history['test_loss'][-1]:.4f}")

    return model, history, checkpoints


if __name__ == "__main__":
    model, history, checkpoints = train()
