"""All diagnostic measurements D1-D6."""

import torch
import torch.nn.functional as F
import numpy as np

from .model import Transformer


def compute_train_loss(model: Transformer, dataset_tensor: torch.Tensor,
                       device: torch.device, batch_size: int = 2048) -> dict:
    """D1: Full training set loss with per-position decomposition.

    Processes in batches to avoid OOM on large datasets.
    """
    model.eval()
    D = dataset_tensor.shape[0]
    total_loss = 0.0
    pos_losses = [0.0, 0.0, 0.0, 0.0]
    eos_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for start in range(0, D, batch_size):
            batch = dataset_tensor[start:start + batch_size].to(device)
            n = batch.shape[0]
            loss, logits = model(batch, batch)
            total_loss += loss.item() * n

            # Per-position losses
            for i in range(4):
                pos = Transformer.LOSS_START + i
                l = F.cross_entropy(logits[:, pos], batch[:, pos + 1])
                pos_losses[i] += l.item() * n
            l = F.cross_entropy(logits[:, 14], batch[:, 15])
            eos_loss += l.item() * n
            total_examples += n

    return {
        "train_loss": total_loss / total_examples,
        "train_loss_pos1": pos_losses[0] / total_examples,
        "train_loss_pos2": pos_losses[1] / total_examples,
        "train_loss_pos3": pos_losses[2] / total_examples,
        "train_loss_pos4": pos_losses[3] / total_examples,
        "train_loss_eos": eos_loss / total_examples,
    }


def compute_z_shuffle_gap(model: Transformer, dataset_tensor: torch.Tensor,
                           device: torch.device, batch_size: int = 1024,
                           rng: np.random.RandomState | None = None) -> dict:
    """D3: z-shuffle gap.

    Compare loss with original z vs randomly permuted z.
    """
    model.eval()
    D = dataset_tensor.shape[0]
    if rng is None:
        rng = np.random.RandomState(0)

    # Select subset
    n = min(batch_size, D)
    indices = rng.choice(D, size=n, replace=False) if D > n else np.arange(D)
    batch = dataset_tensor[indices].clone()

    with torch.no_grad():
        # Clean loss
        clean = batch.to(device)
        loss_clean, _ = model(clean, clean)

        # Shuffled: permute z tokens (positions 8, 9) across batch
        shuffled = batch.clone()
        perm = rng.permutation(n)
        shuffled[:, 8] = batch[perm, 8]
        shuffled[:, 9] = batch[perm, 9]
        shuffled = shuffled.to(device)
        loss_shuffled, _ = model(shuffled, shuffled)

    return {
        "z_shuffle_loss_clean": loss_clean.item(),
        "z_shuffle_loss_shuffled": loss_shuffled.item(),
        "z_shuffle_gap": loss_shuffled.item() - loss_clean.item(),
    }


def compute_group_accuracy(model: Transformer, dataset, device: torch.device,
                            n_groups: int = 200) -> dict:
    """D4: Per-group accuracy.

    For sampled B-groups, check if model predicts all K targets correctly.
    """
    model.eval()
    n_eval = min(n_groups, dataset.n_b)
    rng = np.random.RandomState(0)
    group_indices = rng.choice(dataset.n_b, size=n_eval, replace=False) \
        if dataset.n_b > n_eval else np.arange(dataset.n_b)

    correct_per_group = []

    with torch.no_grad():
        for gi in group_indices:
            examples = dataset.get_group_examples(gi).to(device)  # (K, 16)
            _, logits = model(examples)

            # Check accuracy on A positions (11-14)
            preds = logits[:, Transformer.LOSS_START:Transformer.LOSS_START + 4].argmax(dim=-1)
            targets = examples[:, Transformer.LOSS_START + 1:Transformer.LOSS_START + 5]
            correct = (preds == targets).all(dim=-1)  # (K,) bool
            correct_per_group.append(correct.float().mean().item())

    acc_array = np.array(correct_per_group)
    return {
        "group_accuracy_frac_80": float((acc_array >= 0.8).mean()),
        "group_accuracy_mean": float(acc_array.mean()),
    }


def compute_stable_ranks(model: Transformer) -> dict:
    """D6: Stable rank of all weight matrices."""
    results = {}

    def _stable_rank(name: str, W: torch.Tensor):
        W2d = W.detach().float().cpu()
        if W2d.ndim > 2:
            W2d = W2d.reshape(W2d.shape[0], -1)
        fro_sq = (W2d ** 2).sum().item()
        # Spectral norm = largest singular value
        s = torch.linalg.svdvals(W2d)
        spec_sq = (s[0] ** 2).item()
        if spec_sq > 0:
            results[f"stable_rank_{name}"] = fro_sq / spec_sq
        else:
            results[f"stable_rank_{name}"] = 0.0

    # Embeddings
    _stable_rank("embed", model.tok_embed.weight)
    _stable_rank("unembed", model.unembed.weight)

    # Per-layer weights
    for li, block in enumerate(model.blocks):
        _stable_rank(f"attn_L{li}_Q", block.attn.W_Q.weight)
        _stable_rank(f"attn_L{li}_K", block.attn.W_K.weight)
        _stable_rank(f"attn_L{li}_V", block.attn.W_V.weight)
        _stable_rank(f"attn_L{li}_O", block.attn.W_O.weight)
        _stable_rank(f"mlp_L{li}_in", block.mlp_in.weight)
        _stable_rank(f"mlp_L{li}_out", block.mlp_out.weight)

    return results
