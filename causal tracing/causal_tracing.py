"""
Activation patching / causal tracing for circuit discovery.

Task (a simplified version of the ROME "factual recall" setup): the model sees
a 3-token sequence [subject, relation, SEP] and must predict the correct
"object" from a fixed random lookup table object[subject, relation]. This
gives us a task with a *known* ground-truth input->output mapping, so we can
meaningfully ask "which layer, at which token position, is causally
responsible for the correct prediction?"

Causal tracing procedure (Meng et al., 2022 "ROME", simplified):
  1. Clean run:      normal input -> correct prediction. Cache the residual
                      stream (post-block hidden state) at every (layer, position).
  2. Corrupted run:  add Gaussian noise to the subject token's embedding ->
                      prediction degrades/flips.
  3. Patched run:    for every (layer, position), run the corrupted input but
                      splice in the *clean* residual stream at just that one
                      (layer, position), then let the rest of the forward pass
                      run normally on corrupted inputs. Measure how much of the
                      correct-answer probability gets restored.
  4. The resulting (layer x position) grid of "restoration scores" is the
     causal-tracing heatmap -- it shows where in the network the information
     needed for the correct answer is localized.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Toy task: object = lookup_table[subject, relation]
# ---------------------------------------------------------------------------

SUBJECT, RELATION, SEP = 0, 1, 2  # token positions
N_POSITIONS = 3


def build_task(n_subjects=20, n_relations=5, n_objects=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    lookup_table = torch.randint(0, n_objects, (n_subjects, n_relations), generator=g)

    subj_ids = torch.arange(n_subjects)                       # token ids 0..S-1
    rel_ids = torch.arange(n_relations) + n_subjects           # token ids S..S+R-1
    sep_id = n_subjects + n_relations                          # single SEP token id
    vocab_size = n_subjects + n_relations + 1

    all_pairs = [(s, r) for s in range(n_subjects) for r in range(n_relations)]
    tokens = torch.tensor([[subj_ids[s], rel_ids[r], sep_id] for s, r in all_pairs])
    labels = torch.tensor([lookup_table[s, r] for s, r in all_pairs])

    return {
        "tokens": tokens, "labels": labels, "lookup_table": lookup_table,
        "vocab_size": vocab_size, "n_objects": n_objects,
        "n_subjects": n_subjects, "n_relations": n_relations, "sep_id": sep_id,
    }


# ---------------------------------------------------------------------------
# Minimal transformer, built explicitly (not nn.TransformerEncoder) so we can
# read out / overwrite the residual stream after every block.
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_mlp):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_mlp), nn.GELU(), nn.Linear(d_mlp, d_model))

    def forward(self, x):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class ToyTransformer(nn.Module):
    def __init__(self, vocab_size, n_objects, d_model=64, n_heads=4, n_layers=4, d_mlp=256,
                 n_positions=N_POSITIONS):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(n_positions, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_mlp) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_objects)
        self.n_layers = n_layers
        self.n_positions = n_positions

    def embed(self, tokens, noise_std=0.0, noise_position=None, noise=None):
        """Token+position embedding, optionally with noise injected at one position
        (this is how we 'corrupt' the subject token, as in ROME)."""
        pos_ids = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        x = self.tok_emb(tokens) + self.pos_emb(pos_ids)
        if noise_std > 0.0 and noise_position is not None:
            if noise is None:
                noise = torch.randn_like(x[:, noise_position, :]) * noise_std
            x = x.clone()
            x[:, noise_position, :] = x[:, noise_position, :] + noise
        return x

    def forward(self, x, patch=None, return_residuals=False):
        """
        x: embedded input [batch, n_positions, d_model] (from self.embed)
        patch: optional dict {layer_idx: {position_idx: replacement_vector}}.
               After block `layer_idx` runs, the residual stream at
               `position_idx` is overwritten with `replacement_vector`.
        """
        residuals = []
        for l, block in enumerate(self.blocks):
            x = block(x)
            if patch is not None and l in patch:
                x = x.clone()
                for pos, vec in patch[l].items():
                    x[:, pos, :] = vec
            if return_residuals:
                residuals.append(x.clone())
        x = self.ln_f(x)
        logits = self.head(x[:, -1, :])  # prediction read off the SEP position
        if return_residuals:
            return logits, residuals
        return logits


# ---------------------------------------------------------------------------
# Training (just memorize the lookup table -- the point is circuit discovery,
# not generalization, so we train and evaluate on the same full set of facts)
# ---------------------------------------------------------------------------

def train_model(task, steps=800, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    model = ToyTransformer(task["vocab_size"], task["n_objects"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    tokens, labels = task["tokens"], task["labels"]
    for step in range(1, steps + 1):
        x = model.embed(tokens)
        logits = model(x)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == 1:
            acc = (logits.argmax(-1) == labels).float().mean().item()
            print(f"step {step:4d} | loss {loss.item():.4f} | accuracy {acc:.3f}")

    return model


# ---------------------------------------------------------------------------
# Causal tracing
# ---------------------------------------------------------------------------

@torch.no_grad()
def causal_trace_single(model, tokens_row, label, noise_std):
    """Run clean / corrupted / all single-site patches for ONE example.
    Returns a [n_layers, n_positions] array of restoration scores in [0, 1]
    (0 = no better than corrupted, 1 = fully recovers clean performance)."""
    tokens_row = tokens_row.unsqueeze(0)  # [1, n_positions]

    # clean run
    x_clean = model.embed(tokens_row)
    clean_logits, clean_residuals = model(x_clean, return_residuals=True)
    clean_prob = F.softmax(clean_logits, dim=-1)[0, label].item()

    # corrupted run (noise on the SUBJECT token embedding)
    x_corrupt = model.embed(tokens_row, noise_std=noise_std, noise_position=SUBJECT)
    corrupt_logits = model(x_corrupt)
    corrupt_prob = F.softmax(corrupt_logits, dim=-1)[0, label].item()

    scores = torch.zeros(model.n_layers, model.n_positions)
    denom = max(clean_prob - corrupt_prob, 1e-6)
    for l in range(model.n_layers):
        for p in range(model.n_positions):
            patch = {l: {p: clean_residuals[l][0, p, :]}}
            patched_logits = model(x_corrupt, patch=patch)
            patched_prob = F.softmax(patched_logits, dim=-1)[0, label].item()
            scores[l, p] = (patched_prob - corrupt_prob) / denom

    return scores.numpy(), clean_prob, corrupt_prob


@torch.no_grad()
def causal_trace_averaged(model, task, n_examples=25, noise_std=None, seed=0):
    torch.manual_seed(seed)
    if noise_std is None:
        noise_std = 3.0 * model.tok_emb.weight.std().item()

    idx = torch.randperm(task["tokens"].shape[0])[:n_examples]
    all_scores, clean_probs, corrupt_probs = [], [], []
    for i in idx.tolist():
        scores, cp, xp = causal_trace_single(model, task["tokens"][i], task["labels"][i].item(), noise_std)
        all_scores.append(scores)
        clean_probs.append(cp)
        corrupt_probs.append(xp)

    import numpy as np
    return {
        "scores": np.stack(all_scores).mean(axis=0),   # [n_layers, n_positions]
        "clean_prob_mean": float(np.mean(clean_probs)),
        "corrupt_prob_mean": float(np.mean(corrupt_probs)),
        "noise_std": noise_std,
        "n_examples": n_examples,
    }


if __name__ == "__main__":
    task = build_task()
    model = train_model(task)
    result = causal_trace_averaged(model, task)
    print(f"\nmean clean prob:    {result['clean_prob_mean']:.3f}")
    print(f"mean corrupted prob: {result['corrupt_prob_mean']:.3f}  (should be near chance)")
    print("restoration scores [layer x position]:")
    print(result["scores"])
