"""Checks on src/embeddings.py.

The determinism test is the important one: if a sequence does not embed
identically twice, every cached vector is suspect and so is every result
downstream. These tests load a real model, so they are slow; the smallest
one (438k parameters) is used and marked so it can be deselected.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.embeddings import Embedder, EmbeddingCache  # noqa: E402

pytestmark = pytest.mark.slow

SEQ_SHORT = "ATGGCTAGCTAGCTAGCATCGATCGATCGTAGCTAGCTAGCTAACGTAGCTAG" * 4     # ~212 bp
SEQ_LONG = "ATGCGTACGTTAGCACCGATGCATCGATCGGATCCTAGCATCGGCATTACGGA" * 60   # ~3.2 kb


@pytest.fixture(scope="module")
def embedder():
    return Embedder("hyenadna-tiny", device="cpu")


def test_same_sequence_embeds_identically_twice(embedder):
    a = embedder.embed(SEQ_SHORT)
    b = embedder.embed(SEQ_SHORT)
    assert np.array_equal(a, b)


def test_long_sequence_is_also_deterministic(embedder):
    assert np.array_equal(embedder.embed(SEQ_LONG), embedder.embed(SEQ_LONG))


def test_embedding_is_fixed_length_regardless_of_input_length(embedder):
    short, long_ = embedder.embed(SEQ_SHORT), embedder.embed(SEQ_LONG)
    assert short.shape == long_.shape == (128,)
    assert short.dtype == np.float32


def test_poolings_differ(embedder):
    assert not np.allclose(embedder.embed(SEQ_SHORT, pooling="mean"),
                           embedder.embed(SEQ_SHORT, pooling="last"))


def test_long_strategies_differ_but_short_ones_do_not(embedder):
    # Over the window, the strategies must disagree: they read different bases.
    vecs = [embedder.embed(SEQ_LONG, strategy=s) for s in ("chunk_mean", "truncate", "center")]
    assert not np.allclose(vecs[0], vecs[1])
    assert not np.allclose(vecs[1], vecs[2])
    # Under the window, there is nothing to choose, so all three must agree.
    same = [embedder.embed(SEQ_SHORT, strategy=s) for s in ("chunk_mean", "truncate", "center")]
    assert np.array_equal(same[0], same[1]) and np.array_equal(same[1], same[2])


def test_truncate_ignores_bases_past_the_window(embedder):
    head = SEQ_LONG[:1024]
    assert np.allclose(embedder.embed(SEQ_LONG, strategy="truncate"),
                       embedder.embed(head, strategy="truncate"), atol=1e-5)


def test_cache_round_trips(tmp_path, embedder):
    path = tmp_path / "c.npz"
    cache = EmbeddingCache(path)
    cache.add("BW25113_0001", embedder.embed(SEQ_SHORT))
    cache.save()
    assert np.array_equal(EmbeddingCache(path)["BW25113_0001"], cache["BW25113_0001"])
