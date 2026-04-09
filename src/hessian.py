"""Hessian eigenvalue computation via power iteration."""

import torch
import torch.nn.functional as F
import numpy as np

from .model import Transformer


def _flatten_params(model: Transformer) -> list[torch.nn.Parameter]:
    """Return list of all trainable parameters."""
    return [p for p in model.parameters() if p.requires_grad]


def _hvp(model: Transformer, data: torch.Tensor, device: torch.device,
         vector: list[torch.Tensor]) -> list[torch.Tensor]:
    """Compute Hessian-vector product using double backward.

    H @ v = d/dθ (g^T v) where g = ∇L(θ).
    """
    model.zero_grad()
    batch = data.to(device)
    loss, _ = model(batch, batch)

    params = _flatten_params(model)
    grads = torch.autograd.grad(loss, params, create_graph=True)

    # g^T v
    gv = sum((g * v).sum() for g, v in zip(grads, vector))

    hvp = torch.autograd.grad(gv, params)
    return [h.detach() for h in hvp]


def _random_vector_like(params: list[torch.nn.Parameter],
                        rng: np.random.RandomState) -> list[torch.Tensor]:
    """Generate a random unit vector in parameter space."""
    vecs = []
    for p in params:
        v = torch.from_numpy(
            rng.randn(*p.shape).astype(np.float32)
        ).to(p.device)
        vecs.append(v)
    # Normalize
    norm = sum((v ** 2).sum().item() for v in vecs) ** 0.5
    return [v / norm for v in vecs]


def _vector_norm(vecs: list[torch.Tensor]) -> float:
    return sum((v ** 2).sum().item() for v in vecs) ** 0.5


def _vector_dot(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    return sum((va * vb).sum().item() for va, vb in zip(a, b))


def _normalize(vecs: list[torch.Tensor]) -> list[torch.Tensor]:
    norm = _vector_norm(vecs)
    if norm == 0:
        return vecs
    return [v / norm for v in vecs]


def power_iteration(model: Transformer, data: torch.Tensor, device: torch.device,
                    n_iter: int = 100, seed: int = 0) -> dict:
    """Compute largest Hessian eigenvalue via power iteration.

    Returns lambda_max, the eigenvector (as a list of tensors), iteration count,
    and residual.
    """
    model.eval()
    rng = np.random.RandomState(seed)
    params = _flatten_params(model)
    v = _random_vector_like(params, rng)

    eigenvalue = 0.0
    for i in range(n_iter):
        hv = _hvp(model, data, device, v)
        eigenvalue = _vector_dot(v, hv)
        v = _normalize(hv)

    # Compute residual: ||Hv - λv||
    hv = _hvp(model, data, device, v)
    residual_vecs = [hvi - eigenvalue * vi for hvi, vi in zip(hv, v)]
    residual = _vector_norm(residual_vecs)

    return {
        "lambda_max": eigenvalue,
        "n_iter_max": n_iter,
        "residual_max": residual,
        "eigenvector_max": v,
    }


def inverse_power_iteration(model: Transformer, data: torch.Tensor,
                             device: torch.device, n_iter: int = 100,
                             shift: float = 0.0, seed: int = 0) -> dict:
    """Compute smallest Hessian eigenvalue via shift-and-invert power iteration.

    Approximation: instead of true inverse, we use gradient descent on
    (H - shift*I)^{-1} v. This is a simplified version — for better results
    use Lanczos.

    Here we use the approach of tracking the eigenvalue that converges to
    the smallest magnitude. We apply power iteration and track the minimum
    Rayleigh quotient seen.
    """
    model.eval()
    rng = np.random.RandomState(seed + 1)
    params = _flatten_params(model)

    # Use Lanczos-style approach: run power iteration on -H to find most
    # negative eigenvalue, or track minimum Rayleigh quotient via
    # random restarts
    v = _random_vector_like(params, rng)

    # Power iteration on -H to find most negative eigenvalue of H
    eigenvalue = 0.0
    for i in range(n_iter):
        hv = _hvp(model, data, device, v)
        neg_hv = [-h for h in hv]
        eigenvalue_neg = _vector_dot(v, neg_hv)
        v = _normalize(neg_hv)

    # Rayleigh quotient with original H
    hv = _hvp(model, data, device, v)
    eigenvalue = _vector_dot(v, hv)

    residual_vecs = [hvi - eigenvalue * vi for hvi, vi in zip(hv, v)]
    residual = _vector_norm(residual_vecs)

    return {
        "lambda_min": eigenvalue,
        "n_iter_min": n_iter,
        "residual_min": residual,
        "eigenvector_min": v,
    }


def compute_hessian_eigenvalues(model: Transformer, dataset_tensor: torch.Tensor,
                                 device: torch.device, n_iter: int = 100,
                                 batch_size: int = 512, seed: int = 0) -> dict:
    """Compute lambda_max and lambda_min of the Hessian.

    Uses a subsample of the dataset for efficiency.
    """
    D = dataset_tensor.shape[0]
    rng = np.random.RandomState(seed)
    n = min(batch_size, D)
    indices = rng.choice(D, size=n, replace=False) if D > n else np.arange(D)
    data = dataset_tensor[indices]

    result_max = power_iteration(model, data, device, n_iter, seed)
    result_min = inverse_power_iteration(model, data, device, n_iter, seed=seed)

    return {
        "lambda_max": result_max["lambda_max"],
        "lambda_min": result_min["lambda_min"],
        "n_iter_max": result_max["n_iter_max"],
        "n_iter_min": result_min["n_iter_min"],
        "residual_max": result_max["residual_max"],
        "residual_min": result_min["residual_min"],
    }
