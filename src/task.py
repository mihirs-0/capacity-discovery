"""Data generation: SurjectiveMap class.

Generates the (B, z) -> A lookup table dataset with all constraints verified.
"""

import numpy as np
import torch
from .tokenizer import CharTokenizer


CHARS = list("abcdefghijklmnopqrstuvwxyz0123456789")


class SurjectiveMap:
    """Dataset of (B, z) -> A triples.

    Each of n_b base strings B maps to K distinct target strings A,
    selected by a selector string z.  Total dataset size D = n_b * K.
    """

    def __init__(self, K: int, n_b: int, len_b: int = 6, len_a: int = 4,
                 len_z: int = 2, vocab_size: int = 36, seed: int = 42):
        self.K = K
        self.n_b = n_b
        self.D = n_b * K
        self.len_b = len_b
        self.len_a = len_a
        self.len_z = len_z
        self.vocab_size = vocab_size
        self.seed = seed

        self.tokenizer = CharTokenizer()
        self.rng = np.random.RandomState(seed)

        # Generate and verify dataset
        self.groups = []  # list of (B_str, [(z_str, A_str), ...])
        self._generate()
        self._verify()

        # Pre-tokenize entire dataset as a tensor
        self._build_tensor()

    def _random_string(self, length: int) -> str:
        return "".join(self.rng.choice(CHARS[:self.vocab_size], size=length))

    def _generate(self):
        used_b = set()

        for _ in range(self.n_b):
            # Generate unique B
            for _attempt in range(1000):
                b = self._random_string(self.len_b)
                if b not in used_b:
                    used_b.add(b)
                    break
            else:
                raise RuntimeError("Could not generate unique B string after 1000 attempts")

            # Generate K distinct z strings for this group
            z_strings = self._sample_distinct_strings(self.len_z, self.K)

            # Generate K distinct A strings with distinct first characters
            a_strings = self._sample_distinct_a_strings(self.K)

            pairs = list(zip(z_strings, a_strings))
            self.groups.append((b, pairs))

    def _sample_distinct_strings(self, length: int, count: int) -> list[str]:
        """Sample `count` distinct random strings of given length."""
        seen = set()
        result = []
        for _attempt in range(count * 100):
            s = self._random_string(length)
            if s not in seen:
                seen.add(s)
                result.append(s)
                if len(result) == count:
                    return result
        raise RuntimeError(f"Could not sample {count} distinct strings of length {length}")

    def _sample_distinct_a_strings(self, count: int) -> list[str]:
        """Sample `count` distinct A strings with distinct first characters."""
        if count > self.vocab_size:
            raise ValueError(f"K={count} > vocab_size={self.vocab_size}, "
                             "cannot guarantee distinct first characters")
        for _attempt in range(100):
            strings = []
            first_chars = set()
            full_strings = set()
            ok = True
            for _ in range(count):
                for _inner in range(1000):
                    s = self._random_string(self.len_a)
                    if s[0] not in first_chars and s not in full_strings:
                        first_chars.add(s[0])
                        full_strings.add(s)
                        strings.append(s)
                        break
                else:
                    ok = False
                    break
            if ok:
                return strings
        raise RuntimeError(f"Could not sample {count} A strings with distinct first chars")

    def _verify(self):
        """Verify all dataset constraints."""
        all_b = [g[0] for g in self.groups]
        assert len(set(all_b)) == self.n_b, "B strings not unique"

        for b, pairs in self.groups:
            z_strs = [p[0] for p in pairs]
            a_strs = [p[1] for p in pairs]
            assert len(set(z_strs)) == self.K, f"Duplicate z in group {b}"
            assert len(set(a_strs)) == self.K, f"Duplicate A in group {b}"

            first_chars = [a[0] for a in a_strs]
            assert len(set(first_chars)) == self.K, \
                f"A first chars not distinct in group {b}"

    def _build_tensor(self):
        """Pre-tokenize dataset into a single tensor of shape (D, 16)."""
        all_tokens = []
        for b, pairs in self.groups:
            for z, a in pairs:
                tokens = self.tokenizer.encode_example(b, z, a)
                all_tokens.append(tokens)
        self.data = torch.tensor(all_tokens, dtype=torch.long)  # (D, 16)
        assert self.data.shape == (self.D, 16)

        # Also build group index: group_ids[i] = group index for example i
        self.group_ids = torch.zeros(self.D, dtype=torch.long)
        for gi in range(self.n_b):
            start = gi * self.K
            self.group_ids[start:start + self.K] = gi

    def get_batch(self, batch_size: int, rng: np.random.RandomState) -> torch.Tensor:
        """Sample a batch with replacement. Returns (batch_size, 16)."""
        indices = rng.randint(0, self.D, size=batch_size)
        return self.data[indices]

    def get_full(self) -> torch.Tensor:
        """Return the full dataset as a tensor. Shape (D, 16)."""
        return self.data

    def get_group_examples(self, group_idx: int) -> torch.Tensor:
        """Return all K examples for a given group. Shape (K, 16)."""
        start = group_idx * self.K
        return self.data[start:start + self.K]
