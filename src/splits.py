"""The canonical cross-validation split, created once and reused by everything.

Comparing a language model evaluated on one set of folds against a baseline
evaluated on another is not a comparison. So the fold assignment is written to
disk the first time it is needed and every later script loads that same file.
A test asserts the folds have not moved.
"""

import csv
from pathlib import Path

from sklearn.model_selection import StratifiedKFold

SEED = 0
N_SPLITS = 5
FOLDS_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "cv_folds.csv"


def build_folds(ids, labels, path=FOLDS_PATH):
    """Assign each id a fold, stratified on the label. Written once, never rewritten."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_of = {}
    for fold, (_, test_idx) in enumerate(skf.split(ids, labels)):
        for i in test_idx:
            fold_of[ids[i]] = fold
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["locus_tag", "fold", "essential"])
        for i, gene_id in enumerate(ids):
            w.writerow([gene_id, fold_of[gene_id], int(labels[i])])
    return fold_of


def load_folds(path=FOLDS_PATH):
    with path.open() as f:
        return {r["locus_tag"]: int(r["fold"]) for r in csv.DictReader(f)}


def get_folds(ids, labels, path=FOLDS_PATH):
    """Load the existing split, or create it. Refuses to silently change it."""
    if not path.exists():
        return build_folds(ids, labels, path)
    existing = load_folds(path)
    missing = set(ids) - set(existing)
    if missing:
        raise SystemExit(
            f"{len(missing)} genes are not in {path.name} (e.g. {sorted(missing)[:5]}). "
            "The dataset changed after the folds were fixed. Delete the file to rebuild "
            "it, and re-run every result that used the old folds."
        )
    return existing
