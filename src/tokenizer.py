"""Character-level tokenizer. Fixed vocabulary, no learning."""


class CharTokenizer:
    PAD = 0
    BOS = 1
    SEP = 2
    EOS = 3
    VOCAB_SIZE = 40
    ALPHA_OFFSET = 4  # first alphanumeric token id

    def __init__(self):
        chars = list("abcdefghijklmnopqrstuvwxyz0123456789")
        self.char_to_id = {c: i + self.ALPHA_OFFSET for i, c in enumerate(chars)}
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.id_to_char[self.PAD] = "[PAD]"
        self.id_to_char[self.BOS] = "[BOS]"
        self.id_to_char[self.SEP] = "[SEP]"
        self.id_to_char[self.EOS] = "[EOS]"

    def encode_char(self, c: str) -> int:
        return self.char_to_id[c]

    def decode_id(self, i: int) -> str:
        return self.id_to_char[i]

    def encode(self, s: str) -> list[int]:
        return [self.char_to_id[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            tok = self.id_to_char.get(i)
            if tok is None:
                out.append("?")
            elif tok.startswith("["):
                out.append(tok)
            else:
                out.append(tok)
        return "".join(out)

    def encode_example(self, b: str, z: str, a: str) -> list[int]:
        """Encode a (B, z, A) triple into a full 16-token sequence."""
        tokens = [self.BOS]
        tokens.extend(self.encode(b))
        tokens.append(self.SEP)
        tokens.extend(self.encode(z))
        tokens.append(self.SEP)
        tokens.extend(self.encode(a))
        tokens.append(self.EOS)
        assert len(tokens) == 16, f"Expected 16 tokens, got {len(tokens)}"
        return tokens
