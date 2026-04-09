"""Tests for data generation (SurjectiveMap)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.task import SurjectiveMap


def test_dataset_size():
    """K=5, n_b=10 => D=50."""
    ds = SurjectiveMap(K=5, n_b=10, seed=42)
    assert ds.D == 50
    assert ds.data.shape == (50, 16)


def test_b_strings_unique():
    """All B strings must be unique."""
    ds = SurjectiveMap(K=5, n_b=10, seed=42)
    b_strings = [g[0] for g in ds.groups]
    assert len(set(b_strings)) == 10


def test_a_first_chars_distinct():
    """Within each group, all A strings have distinct first characters."""
    ds = SurjectiveMap(K=5, n_b=10, seed=42)
    for b, pairs in ds.groups:
        a_strings = [p[1] for p in pairs]
        first_chars = [a[0] for a in a_strings]
        assert len(set(first_chars)) == 5, \
            f"Group {b}: first chars not distinct: {first_chars}"


def test_z_strings_distinct():
    """Within each group, all z strings are distinct."""
    ds = SurjectiveMap(K=5, n_b=10, seed=42)
    for b, pairs in ds.groups:
        z_strings = [p[0] for p in pairs]
        assert len(set(z_strings)) == 5, \
            f"Group {b}: z strings not distinct"


def test_a_strings_distinct():
    """Within each group, all A strings are distinct."""
    ds = SurjectiveMap(K=5, n_b=10, seed=42)
    for b, pairs in ds.groups:
        a_strings = [p[1] for p in pairs]
        assert len(set(a_strings)) == 5, \
            f"Group {b}: A strings not distinct"


def test_deterministic_seed():
    """Same seed produces identical dataset."""
    ds1 = SurjectiveMap(K=5, n_b=10, seed=42)
    ds2 = SurjectiveMap(K=5, n_b=10, seed=42)
    assert (ds1.data == ds2.data).all()


def test_different_seeds():
    """Different seeds produce different datasets."""
    ds1 = SurjectiveMap(K=5, n_b=10, seed=42)
    ds2 = SurjectiveMap(K=5, n_b=10, seed=99)
    assert not (ds1.data == ds2.data).all()


def test_large_K():
    """K=36 should still satisfy first-char distinctness."""
    ds = SurjectiveMap(K=36, n_b=3, seed=42)
    for b, pairs in ds.groups:
        a_strings = [p[1] for p in pairs]
        first_chars = [a[0] for a in a_strings]
        assert len(set(first_chars)) == 36


def test_get_batch():
    """get_batch returns correct shape."""
    import numpy as np
    ds = SurjectiveMap(K=5, n_b=10, seed=42)
    rng = np.random.RandomState(0)
    batch = ds.get_batch(8, rng)
    assert batch.shape == (8, 16)


def test_get_full():
    """get_full returns entire dataset."""
    ds = SurjectiveMap(K=5, n_b=10, seed=42)
    full = ds.get_full()
    assert full.shape == (50, 16)
    assert (full == ds.data).all()


def test_get_group_examples():
    """get_group_examples returns K examples for a group."""
    ds = SurjectiveMap(K=5, n_b=10, seed=42)
    group = ds.get_group_examples(0)
    assert group.shape == (5, 16)


def test_group_ids():
    """group_ids correctly maps examples to groups."""
    ds = SurjectiveMap(K=5, n_b=10, seed=42)
    assert ds.group_ids.shape == (50,)
    # First 5 examples should be group 0
    assert (ds.group_ids[:5] == 0).all()
    # Last 5 should be group 9
    assert (ds.group_ids[45:50] == 9).all()
