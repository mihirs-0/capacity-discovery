"""Tests for diagnostic computations."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import pytest
import torch
import torch.nn as nn
import numpy as np

from src.model import Transformer
from src.task import SurjectiveMap
from src.diagnostics import (
    compute_train_loss,
    compute_z_shuffle_gap,
    compute_group_accuracy,
    compute_stable_ranks,
)


@pytest.fixture
def small_dataset():
    return SurjectiveMap(K=5, n_b=10, seed=42)


@pytest.fixture
def small_model():
    torch.manual_seed(0)
    return Transformer(n_layers=2, n_heads=2, d_model=32, d_mlp=64)


def test_train_loss_shape(small_model, small_dataset):
    """compute_train_loss returns expected keys."""
    device = torch.device("cpu")
    result = compute_train_loss(small_model, small_dataset.get_full(), device)
    expected_keys = ["train_loss", "train_loss_pos1", "train_loss_pos2",
                     "train_loss_pos3", "train_loss_pos4", "train_loss_eos"]
    for k in expected_keys:
        assert k in result, f"Missing key: {k}"
        assert isinstance(result[k], float)
        assert result[k] >= 0


def test_z_shuffle_gap_shape(small_model, small_dataset):
    """compute_z_shuffle_gap returns expected keys."""
    device = torch.device("cpu")
    result = compute_z_shuffle_gap(small_model, small_dataset.get_full(), device)
    assert "z_shuffle_gap" in result
    assert "z_shuffle_loss_clean" in result
    assert "z_shuffle_loss_shuffled" in result


def test_z_shuffle_gap_untrained(small_model, small_dataset):
    """For an untrained model, z-shuffle gap should be near zero."""
    device = torch.device("cpu")
    result = compute_z_shuffle_gap(small_model, small_dataset.get_full(), device)
    # Untrained model shouldn't rely on z, so gap should be small
    assert abs(result["z_shuffle_gap"]) < 2.0, \
        f"Unexpectedly large gap for untrained model: {result['z_shuffle_gap']}"


def test_group_accuracy_shape(small_model, small_dataset):
    """compute_group_accuracy returns expected keys."""
    device = torch.device("cpu")
    result = compute_group_accuracy(small_model, small_dataset, device, n_groups=5)
    assert "group_accuracy_frac_80" in result
    assert "group_accuracy_mean" in result
    assert 0 <= result["group_accuracy_frac_80"] <= 1
    assert 0 <= result["group_accuracy_mean"] <= 1


def test_stable_rank_known_matrix():
    """Stable rank of a rank-1 matrix should be 1."""
    a = torch.randn(10, 1)
    b = torch.randn(1, 5)
    W = a @ b  # rank 1

    fro_sq = (W ** 2).sum().item()
    s = torch.linalg.svdvals(W)
    spec_sq = (s[0] ** 2).item()
    stable_rank = fro_sq / spec_sq

    assert abs(stable_rank - 1.0) < 1e-4, f"Stable rank of rank-1 matrix: {stable_rank}"


def test_stable_rank_identity():
    """Stable rank of identity matrix should equal its dimension."""
    n = 10
    W = torch.eye(n)
    fro_sq = (W ** 2).sum().item()  # = n
    s = torch.linalg.svdvals(W)
    spec_sq = (s[0] ** 2).item()  # = 1
    stable_rank = fro_sq / spec_sq

    assert abs(stable_rank - n) < 1e-4, f"Stable rank of I_{n}: {stable_rank}"


def test_stable_ranks_output(small_model):
    """compute_stable_ranks returns metrics for all weight matrices."""
    result = compute_stable_ranks(small_model)
    assert "stable_rank_embed" in result
    assert "stable_rank_unembed" in result
    assert "stable_rank_attn_L0_Q" in result
    assert "stable_rank_mlp_L0_in" in result
    # All should be positive
    for k, v in result.items():
        assert v > 0, f"{k} is not positive: {v}"
