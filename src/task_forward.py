"""Forward task: A -> B.

Reuses the underlying (A, B) associations from the backward SurjectiveMap
(same `seed` produces the same groups), but flips the direction:

    input  : [BOS] a1 a2 a3 a4 [SEP] b1 b2 b3 b4 b5 b6 [EOS] [PAD] [PAD] [PAD]
    pos    :  0    1  2  3  4   5    6  7  8  9 10 11  12    13    14    15

    Loss   : positions 5..11 predict positions 6..12 (B tokens + EOS)
    Last input position (pre-decode): 5  (the [SEP] after A)

Each A maps to exactly one B. K different A strings per group still share the
same B, and `get_group_members` returns those K tokenized sequences so the
representation-convergence analysis can query them.

Notes on uniqueness:
  The backward dataset only guarantees A-string uniqueness *within* a group.
  Across groups, A strings can collide (len=4, vocab=36 ~ 1.7M strings, but
  at D=10K we expect a handful of cross-group collisions).  Any A that maps
  to more than one distinct B breaks A->B as a function, so we drop those
  examples and record how many were dropped.
"""

from __future__ import annotations

import numpy as np
import torch

from .task import SurjectiveMap
from .tokenizer import CharTokenizer


# ---- Sequence layout for the forward task ---------------------------------
SEQ_LEN = 16  # keep model.max_seq_len=16 compatible
BOS_POS = 0
A_POS = [1, 2, 3, 4]
A_SEP_POS = 5
B_POS = [6, 7, 8, 9, 10, 11]
EOS_POS = 12
PAD_POS = [13, 14, 15]

LOSS_START = 5        # logit at pos 5 predicts b1 at pos 6
LOSS_END = 12         # logit at pos 11 predicts EOS at pos 12 (inclusive)
N_LOSS_POS = LOSS_END - LOSS_START + 1  # 7 positions

LAST_INPUT_POS = A_SEP_POS  # the residual stream position used to emit b1


class ForwardSurjectiveMap:
    """Forward-direction dataset: A (+ no z) -> B.

    Parameters match SurjectiveMap so the same (A, B) groups are reproduced
    from the same seed.  K and n_b still refer to the underlying grouping:
    K A-strings map to the same B.
    """

    def __init__(self, K: int, n_b: int, len_b: int = 6, len_a: int = 4,
                 len_z: int = 2, vocab_size: int = 36, seed: int = 42):
        self.K = K
        self.n_b = n_b
        self.len_b = len_b
        self.len_a = len_a
        self.len_z = len_z           # unused but kept for signature parity
        self.vocab_size = vocab_size
        self.seed = seed

        self.tokenizer = CharTokenizer()

        # Build the backward dataset purely to reuse its (B, [(z, A), ...]) groups.
        backward = SurjectiveMap(
            K=K, n_b=n_b, len_b=len_b, len_a=len_a, len_z=len_z,
            vocab_size=vocab_size, seed=seed,
        )

        # Walk groups, collect (a_str, b_str, group_idx).  Detect any A string
        # that maps to more than one distinct B and drop those.
        a_to_b: dict[str, str] = {}
        collisions: set[str] = set()
        for gi, (b_str, pairs) in enumerate(backward.groups):
            for _z, a in pairs:
                prior = a_to_b.get(a)
                if prior is None:
                    a_to_b[a] = b_str
                elif prior != b_str:
                    collisions.add(a)
        self.n_collisions = len(collisions)

        # Build per-group member lists (only valid A strings) and the data tensor.
        self.groups: list[tuple[str, list[str]]] = []  # (b_str, [a_str, ...])
        group_ids: list[int] = []
        all_tokens: list[list[int]] = []
        self._group_example_indices: list[list[int]] = []

        for gi, (b_str, pairs) in enumerate(backward.groups):
            members_a: list[str] = []
            member_indices: list[int] = []
            for _z, a in pairs:
                if a in collisions:
                    continue
                tokens = self._encode_forward(a, b_str)
                member_indices.append(len(all_tokens))
                all_tokens.append(tokens)
                group_ids.append(gi)
                members_a.append(a)
            self.groups.append((b_str, members_a))
            self._group_example_indices.append(member_indices)

        self.data = torch.tensor(all_tokens, dtype=torch.long)  # (N, 16)
        self.group_ids = torch.tensor(group_ids, dtype=torch.long)
        self.D = self.data.shape[0]
        # Target D if no collisions had occurred
        self.D_nominal = n_b * K

        assert self.data.shape == (self.D, SEQ_LEN)
        self._verify()

    # ---- encoding --------------------------------------------------------

    def _encode_forward(self, a_str: str, b_str: str) -> list[int]:
        """[BOS] A(4) [SEP] B(6) [EOS] [PAD] [PAD] [PAD]  -> 16 tokens."""
        assert len(a_str) == self.len_a
        assert len(b_str) == self.len_b
        tok = self.tokenizer
        seq = [tok.BOS]
        seq.extend(tok.encode(a_str))         # pos 1..4
        seq.append(tok.SEP)                   # pos 5
        seq.extend(tok.encode(b_str))         # pos 6..11
        seq.append(tok.EOS)                   # pos 12
        seq.extend([tok.PAD, tok.PAD, tok.PAD])  # pos 13..15
        assert len(seq) == SEQ_LEN
        return seq

    def _verify(self):
        # Every A string present must map to exactly one B (already enforced
        # by the collision drop, but double-check).
        seen: dict[str, str] = {}
        for b_str, a_list in self.groups:
            for a in a_list:
                prev = seen.get(a)
                assert prev is None or prev == b_str, \
                    f"A string {a!r} maps to multiple Bs — collision drop failed"
                seen[a] = b_str

    # ---- sampling --------------------------------------------------------

    def get_batch(self, batch_size: int, rng: np.random.RandomState) -> torch.Tensor:
        indices = rng.randint(0, self.D, size=batch_size)
        return self.data[indices]

    def get_full(self) -> torch.Tensor:
        return self.data

    def get_group_members(self, group_idx: int) -> torch.Tensor:
        """All K (or fewer, after collision drop) tokenized A inputs that
        share the same B.  Shape (k, 16)."""
        idxs = self._group_example_indices[group_idx]
        if not idxs:
            return self.data[0:0]
        return self.data[torch.tensor(idxs, dtype=torch.long)]

    def group_size(self, group_idx: int) -> int:
        return len(self._group_example_indices[group_idx])

    # ---- static constants used by the trainer / analysis ----------------

    LOSS_START = LOSS_START
    LOSS_END = LOSS_END          # inclusive
    N_LOSS_POS = N_LOSS_POS
    LAST_INPUT_POS = LAST_INPUT_POS
    SEQ_LEN = SEQ_LEN
