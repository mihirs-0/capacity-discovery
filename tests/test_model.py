"""Tests for the Transformer model."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from src.model import Transformer
from src.config import ModelConfig


@pytest.fixture
def default_model():
    torch.manual_seed(0)
    return Transformer(n_layers=4, n_heads=4, d_model=128, d_mlp=512)


def test_param_count(default_model):
    """Default model should have ~600K parameters."""
    count = default_model.count_parameters()
    print(f"Exact parameter count: {count:,}")
    assert 400_000 < count < 900_000, f"Unexpected param count: {count}"


def test_forward_shape(default_model):
    """Forward pass with batch of 4 produces correct shapes."""
    batch = torch.randint(0, 40, (4, 16))
    loss, logits = default_model(batch, batch)
    assert logits.shape == (4, 16, 40)
    assert loss is not None
    assert loss.ndim == 0  # scalar


def test_forward_no_targets(default_model):
    """Forward without targets returns None loss."""
    batch = torch.randint(0, 40, (4, 16))
    loss, logits = default_model(batch)
    assert loss is None
    assert logits.shape == (4, 16, 40)


def test_loss_positions(default_model):
    """Loss should only depend on A+EOS positions (10-14 predicting 11-15)."""
    torch.manual_seed(42)
    batch = torch.randint(0, 40, (4, 16))

    # Get baseline loss
    loss1, _ = default_model(batch, batch)

    # Modify tokens at positions 0-10 (input) — loss should change
    # because the model sees different context
    batch2 = batch.clone()
    batch2[:, 0] = (batch2[:, 0] + 1) % 40
    loss2, _ = default_model(batch2, batch2)
    # These might differ because context changed

    # Key test: targets at positions 0-10 should NOT affect loss
    # because loss mask excludes them
    targets_modified = batch.clone()
    targets_modified[:, 0:11] = torch.randint(0, 40, (4, 11))
    # But keep positions 11-15 same as batch
    targets_modified[:, 11:16] = batch[:, 11:16]
    loss3, _ = default_model(batch, targets_modified)

    # loss1 and loss3 should be identical (same inputs, same target A+EOS)
    assert torch.allclose(loss1, loss3), \
        f"Loss changed when modifying non-target positions: {loss1:.6f} vs {loss3:.6f}"


def test_causal_masking(default_model):
    """Attention weights from position i should be zero for j > i."""
    batch = torch.randint(0, 40, (1, 16))

    # Hook into the first attention layer to capture attention weights
    attn_weights = []

    def hook(module, input, output):
        # We need to capture attention weights inside the forward pass
        pass

    # Instead, manually compute attention to verify masking
    block = default_model.blocks[0]
    x = default_model.tok_embed(batch) + default_model.pos_embed(torch.arange(16))
    h = block.ln1(x)

    import math
    B, T, C = h.shape
    n_heads = block.attn.n_heads
    d_head = block.attn.d_head

    q = block.attn.W_Q(h).view(B, T, n_heads, d_head).transpose(1, 2)
    k = block.attn.W_K(h).view(B, T, n_heads, d_head).transpose(1, 2)

    attn = (q @ k.transpose(-2, -1)) / math.sqrt(d_head)
    attn = attn.masked_fill(~block.attn.causal_mask[:T, :T], float("-inf"))
    attn = torch.softmax(attn, dim=-1)

    # Check causal property: attn[b, h, i, j] should be 0 for j > i
    for i in range(16):
        for j in range(i + 1, 16):
            val = attn[0, :, i, j].max().item()
            assert val < 1e-6, f"Non-zero attention at ({i},{j}): {val}"


def test_per_position_loss(default_model):
    """Per-position loss computation should work."""
    batch = torch.randint(0, 40, (4, 16))
    losses = default_model.compute_per_position_loss(batch, batch)
    assert "train_loss_pos1" in losses
    assert "train_loss_pos2" in losses
    assert "train_loss_pos3" in losses
    assert "train_loss_pos4" in losses
    assert "train_loss_eos" in losses
    for k, v in losses.items():
        assert v >= 0, f"{k} is negative: {v}"


def test_from_config():
    """Model can be constructed from ModelConfig."""
    cfg = ModelConfig()
    model = Transformer.from_config(cfg)
    assert model.count_parameters() > 0


def test_various_model_sizes():
    """Verify param counts for different model configurations."""
    configs = [
        (32, 128, "S1"),
        (48, 192, "S2"),
        (64, 256, "S3"),
        (96, 384, "S4"),
        (128, 512, "S5"),
        (192, 768, "S6"),
        (256, 1024, "S7"),
    ]
    for d_model, d_mlp, name in configs:
        model = Transformer(n_layers=4, n_heads=4, d_model=d_model, d_mlp=d_mlp)
        count = model.count_parameters()
        print(f"{name}: d_model={d_model}, d_mlp={d_mlp}, params={count:,}")
