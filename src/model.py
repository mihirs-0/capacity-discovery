"""Decoder-only Transformer from scratch in PyTorch."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        )

    def forward(self, x: torch.Tensor,
                return_attention: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape

        q = self.W_Q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_K(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_V(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = attn.masked_fill(~self.causal_mask[:T, :T], float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        if return_attention:
            return self.W_O(out), attn  # attn: (B, n_heads, T, T)
        return self.W_O(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_mlp: int, max_seq_len: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp_in = nn.Linear(d_model, d_mlp)
        self.mlp_out = nn.Linear(d_mlp, d_model)

    def forward(self, x: torch.Tensor,
                return_attention: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if return_attention:
            attn_out, attn_weights = self.attn(self.ln1(x), return_attention=True)
            x = x + attn_out
        else:
            x = x + self.attn(self.ln1(x))
            attn_weights = None
        h = self.ln2(x)
        x = x + self.mlp_out(F.gelu(self.mlp_in(h)))
        if return_attention:
            return x, attn_weights
        return x


class Transformer(nn.Module):
    """Decoder-only transformer for the surjective map task.

    Input: (batch, seq_len) token IDs
    Output: (loss, logits) where loss is computed only on A+EOS positions.

    Sequence layout (16 tokens):
        [BOS] b1 b2 b3 b4 b5 b6 [SEP] z1 z2 [SEP] a1 a2 a3 a4 [EOS]
        pos:  0   1  2  3  4  5  6    7    8  9   10  11 12 13 14  15

    Loss positions: logits at positions 10-14 predict tokens at positions 11-15
    (i.e., a1 a2 a3 a4 EOS).
    """

    # Positions where we compute loss (predicting the *next* token)
    LOSS_START = 10  # logit at pos 10 predicts a1 at pos 11
    LOSS_END = 15    # logit at pos 14 predicts EOS at pos 15

    def __init__(self, n_layers: int, n_heads: int, d_model: int, d_mlp: int,
                 vocab_size: int = 40, max_seq_len: int = 16):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_mlp, max_seq_len)
            for _ in range(n_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)
        # Linear layers use default Kaiming uniform init (PyTorch default)

    def forward(self, input_ids: torch.Tensor,
                targets: torch.Tensor | None = None) -> tuple[torch.Tensor | None, torch.Tensor]:
        """
        Args:
            input_ids: (batch, seq_len) token IDs
            targets: (batch, seq_len) token IDs (same as input_ids for teacher forcing).
                     If None, returns (None, logits).

        Returns:
            (loss, logits) where logits is (batch, seq_len, vocab_size).
            loss is scalar mean over A+EOS positions only.
        """
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)

        x = self.tok_embed(input_ids) + self.pos_embed(pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        logits = self.unembed(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Loss on positions 10..14 predicting targets at 11..15
            loss_logits = logits[:, self.LOSS_START:self.LOSS_END]  # (B, 5, V)
            loss_targets = targets[:, self.LOSS_START + 1:self.LOSS_END + 1]  # (B, 5)
            loss = F.cross_entropy(
                loss_logits.reshape(-1, self.vocab_size),
                loss_targets.reshape(-1),
            )

        return loss, logits

    def forward_with_attention(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass that also returns attention weights from every layer.

        Returns:
            (logits, attention_weights) where attention_weights is a list of
            tensors, one per layer, each of shape (batch, n_heads, seq_len, seq_len).

        Only use during evaluation — not compatible with torch.compile.
        """
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)
        x = self.tok_embed(input_ids) + self.pos_embed(pos)

        all_attn = []
        for block in self.blocks:
            x, attn_weights = block(x, return_attention=True)
            all_attn.append(attn_weights)

        x = self.ln_final(x)
        logits = self.unembed(x)
        return logits, all_attn

    def compute_per_position_loss(self, input_ids: torch.Tensor,
                                  targets: torch.Tensor) -> dict[str, float]:
        """Compute loss broken down by A position (1-4) and EOS."""
        B, T = input_ids.shape
        with torch.no_grad():
            _, logits = self.forward(input_ids)

        losses = {}
        # Position 1 of A: logit at pos 10 → target at pos 11
        for i in range(4):
            pos = self.LOSS_START + i
            l = F.cross_entropy(logits[:, pos], targets[:, pos + 1])
            losses[f"train_loss_pos{i + 1}"] = l.item()

        # EOS position: logit at pos 14 → target at pos 15
        l = F.cross_entropy(logits[:, 14], targets[:, 15])
        losses["train_loss_eos"] = l.item()

        return losses

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_config(cls, cfg) -> "Transformer":
        """Construct from a ModelConfig dataclass."""
        return cls(
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            d_model=cfg.d_model,
            d_mlp=cfg.d_mlp,
            vocab_size=cfg.vocab_size,
            max_seq_len=cfg.max_seq_len,
        )
