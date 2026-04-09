"""Tests for the character-level tokenizer."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.tokenizer import CharTokenizer


@pytest.fixture
def tok():
    return CharTokenizer()


def test_vocab_size(tok):
    assert tok.VOCAB_SIZE == 40


def test_special_tokens(tok):
    assert tok.PAD == 0
    assert tok.BOS == 1
    assert tok.SEP == 2
    assert tok.EOS == 3


def test_encode_decode_roundtrip(tok):
    """All 36 alphanumeric characters roundtrip correctly."""
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    for c in chars:
        encoded = tok.encode_char(c)
        decoded = tok.decode_id(encoded)
        assert decoded == c, f"Roundtrip failed for '{c}': encoded={encoded}, decoded={decoded}"


def test_encode_string(tok):
    ids = tok.encode("abc")
    assert ids == [4, 5, 6]


def test_decode_string(tok):
    s = tok.decode([4, 5, 6])
    assert s == "abc"


def test_encode_example_length(tok):
    """Full input sequence should be exactly 16 tokens."""
    tokens = tok.encode_example("abcdef", "xy", "wxyz")
    assert len(tokens) == 16


def test_encode_example_structure(tok):
    """Verify token layout: [BOS] B*6 [SEP] z*2 [SEP] A*4 [EOS]."""
    tokens = tok.encode_example("abcdef", "01", "wxyz")
    assert tokens[0] == tok.BOS
    assert tokens[7] == tok.SEP
    assert tokens[10] == tok.SEP
    assert tokens[15] == tok.EOS
    # B tokens at 1-6
    assert tokens[1:7] == tok.encode("abcdef")
    # z tokens at 8-9
    assert tokens[8:10] == tok.encode("01")
    # A tokens at 11-14
    assert tokens[11:15] == tok.encode("wxyz")


def test_special_token_decoding(tok):
    assert tok.decode_id(0) == "[PAD]"
    assert tok.decode_id(1) == "[BOS]"
    assert tok.decode_id(2) == "[SEP]"
    assert tok.decode_id(3) == "[EOS]"


def test_all_alphanumeric_have_unique_ids(tok):
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    ids = [tok.encode_char(c) for c in chars]
    assert len(set(ids)) == 36
    # All IDs should be 4-39
    assert min(ids) == 4
    assert max(ids) == 39
