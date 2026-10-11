"""The cross-validation folds must never move.

Every model in this project is compared against every other one, so they all
have to be evaluated on the same splits. If the fold file changes, earlier
results silently stop being comparable with later ones. These tests fail loudly
if that happens.
"""

import csv
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.splits import N_SPLITS, SEED, FOLDS_PATH, build_folds, load_folds  # noqa: E402

PROCESSED = ROOT / "data" / "processed"

if not FOLDS_PATH.exists():
    pytest.skip("run scripts/baseline_features.py first", allow_module_level=True)


@pytest.fixture(scope="module")
def rows():
    with FOLDS_PATH.open() as f:
        return list(csv.DictReader(f))


def test_every_gene_has_exactly_one_fold(rows):
    tags = [r["locus_tag"] for r in rows]
    assert len(tags) == len(set(tags))
    assert all(0 <= int(r["fold"]) < N_SPLITS for r in rows)


def test_folds_cover_the_labelled_protein_coding_set(rows):
    genes = {r["locus_tag"]: r for r in csv.DictReader((PROCESSED / "genes.csv").open())}
    labelled = {r["locus_tag"] for r in csv.DictReader((PROCESSED / "essentiality.csv").open())
                if r["source"] == "Goodall2018_ST1" and r["call"] in ("Essential", "Non-essential")
                and genes.get(r["locus_tag"], {}).get("biotype") == "protein_coding"}
    assert {r["locus_tag"] for r in rows} == labelled


def test_folds_are_stratified_and_balanced(rows):
    sizes = Counter(r["fold"] for r in rows)
    assert max(sizes.values()) - min(sizes.values()) <= 1
    overall = sum(int(r["essential"]) for r in rows) / len(rows)
    for fold in range(N_SPLITS):
        inside = [r for r in rows if int(r["fold"]) == fold]
        rate = sum(int(r["essential"]) for r in inside) / len(inside)
        assert abs(rate - overall) < 0.02, (fold, rate, overall)


def test_fold_assignment_is_reproducible(rows, tmp_path):
    """Rebuilding from the same ids, labels and seed must give the same folds."""
    ids = [r["locus_tag"] for r in rows]
    labels = [int(r["essential"]) for r in rows]
    rebuilt = build_folds(ids, labels, tmp_path / "folds.csv")
    assert rebuilt == load_folds(FOLDS_PATH)
