"""The classical-features baseline: the number every language model must beat.

Extracts the features in src/features.py and trains logistic regression and
gradient boosting under the canonical stratified 5-fold split from
src/splits.py. The same folds are reused by every later model, so the
comparison is real.

Reports MCC, per-class precision and recall, AUROC and average precision, each
against the majority-class baseline. Accuracy is printed only next to the
majority-class figure, to make clear how little it says: with a 1:10 imbalance,
predicting "non-essential" for everything already scores about 91%.

Also runs two ablations, because they decide how the headline result reads:
  length only          how much is explained by gene length alone
  no length features   whether the rest of the signal survives without it

    .venv\\Scripts\\python scripts\\baseline_features.py [--quick]
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from Bio import SeqIO
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features import feature_names, features_for, relative_adaptiveness  # noqa: E402
from src.splits import FOLDS_PATH, N_SPLITS, SEED, get_folds  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RIBOSOMAL = ("rps", "rpl", "rpm")   # reference set for CAI: highly expressed genes


def load_dataset(quick):
    genes = {r["locus_tag"]: r for r in csv.DictReader((PROCESSED / "genes.csv").open())}
    seqs = {r.id: str(r.seq).upper() for r in SeqIO.parse(PROCESSED / "genes.fasta", "fasta")}
    labels = {}
    for r in csv.DictReader((PROCESSED / "essentiality.csv").open()):
        if r["source"] == "Goodall2018_ST1" and r["call"] in ("Essential", "Non-essential"):
            labels[r["locus_tag"]] = r["call"] == "Essential"

    usable = sorted(t for t, g in genes.items()
                    if g["biotype"] == "protein_coding" and t in labels and t in seqs)
    if quick:
        # Keep both classes represented in the smoke run.
        ess = [t for t in usable if labels[t]][:40]
        non = [t for t in usable if not labels[t]][:160]
        usable = sorted(ess + non)
    return genes, seqs, labels, usable


def build_matrix(genes, seqs, usable):
    reference = [seqs[t] for t in usable if genes[t]["gene_name"].startswith(RIBOSOMAL)]
    print(f"CAI reference set: {len(reference)} ribosomal protein genes")
    weights = relative_adaptiveness(reference)
    t0 = time.perf_counter()
    X = np.array([features_for(seqs[t], weights) for t in usable], dtype=np.float64)
    print(f"Features: {X.shape[0]:,} genes x {X.shape[1]} features "
          f"in {time.perf_counter() - t0:.1f}s")
    return X


def models():
    # class_weight="balanced" matters here: with a 1:10 split the unweighted
    # models mostly predict the majority class and MCC collapses.
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED),
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.06, early_stopping=True,
            validation_fraction=0.15, class_weight="balanced", random_state=SEED,
        ),
    }


def cross_validate(X, y, fold_ids, name):
    """Out-of-fold predictions, so every gene is predicted by a model that never saw it."""
    pred = np.zeros(len(y), dtype=int)
    score = np.zeros(len(y), dtype=float)
    for fold in range(N_SPLITS):
        test = fold_ids == fold
        model = models()[name]
        model.fit(X[~test], y[~test])
        pred[test] = model.predict(X[test])
        score[test] = model.predict_proba(X[test])[:, 1]
    return pred, score


def evaluate(y, pred, score):
    p, r, f1, support = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    return {
        "mcc": round(matthews_corrcoef(y, pred), 4),
        "auroc": round(roc_auc_score(y, score), 4),
        "average_precision": round(average_precision_score(y, score), 4),
        "accuracy": round(float((pred == y).mean()), 4),
        "essential_precision": round(p[1], 4),
        "essential_recall": round(r[1], 4),
        "essential_f1": round(f1[1], 4),
        "non_essential_precision": round(p[0], 4),
        "non_essential_recall": round(r[0], 4),
        "n_essential": int(support[1]),
        "n_non_essential": int(support[0]),
    }


def bootstrap_mcc(y, pred, n=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() in (0, len(idx)):
            continue
        vals.append(matthews_corrcoef(y[idx], pred[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="200-gene smoke run")
    args = parser.parse_args()

    genes, seqs, labels, usable = load_dataset(args.quick)
    y = np.array([labels[t] for t in usable], dtype=int)
    print(f"Dataset: {len(usable):,} protein-coding genes, "
          f"{y.sum()} essential, {len(y) - y.sum()} non-essential")

    # A --quick run must never write the canonical split: it would fix the folds
    # from a 200-gene subset and every later model would inherit them.
    folds_path = (PROCESSED / "cv_folds_quick.csv") if args.quick else FOLDS_PATH
    folds = get_folds(usable, y, folds_path)
    fold_ids = np.array([folds[t] for t in usable])
    majority = float(max(y.mean(), 1 - y.mean()))
    print(f"Majority-class baseline: accuracy {majority:.4f}, MCC 0.0 by construction")

    names = feature_names()
    X = build_matrix(genes, seqs, usable)

    length_cols = [i for i, n in enumerate(names) if n in ("length", "log_length", "n_codons")]
    feature_sets = {
        "all": list(range(len(names))),
        "length_only": length_cols,
        "no_length": [i for i in range(len(names)) if i not in length_cols],
    }

    results, importances, oof = {}, {}, {}
    for set_name, cols in feature_sets.items():
        for model_name in models():
            t0 = time.perf_counter()
            pred, score = cross_validate(X[:, cols], y, fold_ids, model_name)
            metrics = evaluate(y, pred, score)
            metrics["mcc_ci95"] = bootstrap_mcc(y, pred)
            metrics["seconds"] = round(time.perf_counter() - t0, 1)
            metrics["n_features"] = len(cols)
            results[f"{set_name}__{model_name}"] = metrics
            oof[f"{set_name}__{model_name}"] = pred
            print(f"  {set_name:12} {model_name:20} MCC {metrics['mcc']:.4f} "
                  f"CI {metrics['mcc_ci95']}  AUROC {metrics['auroc']:.4f}  "
                  f"ess recall {metrics['essential_recall']:.3f}  ({metrics['seconds']}s)")

    # Feature importance from the full-feature logistic regression, fit once on
    # everything; coefficients are on standardised features so they compare.
    lr = models()["logistic_regression"].fit(X, y)
    coefs = lr.named_steps["logisticregression"].coef_[0]
    importances["logistic_regression_abs_coef"] = sorted(
        ({"feature": n, "coef": round(float(c), 4)} for n, c in zip(names, coefs)),
        key=lambda d: -abs(d["coef"]))[:25]

    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": {
            "n_genes": len(usable), "n_essential": int(y.sum()),
            "n_non_essential": int(len(y) - y.sum()),
            "majority_class_accuracy": round(majority, 4),
            "n_folds": N_SPLITS, "seed": SEED, "quick": args.quick,
        },
        "results": results,
        "importances": importances,
    }
    (RESULTS / "baseline.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (RESULTS / "baseline.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature_set", "model", "n_features", "mcc", "mcc_lo", "mcc_hi", "auroc",
                    "average_precision", "accuracy", "ess_precision", "ess_recall", "ess_f1"])
        for key, m in results.items():
            fs, mn = key.split("__")
            w.writerow([fs, mn, m["n_features"], m["mcc"], *m["mcc_ci95"], m["auroc"],
                        m["average_precision"], m["accuracy"], m["essential_precision"],
                        m["essential_recall"], m["essential_f1"]])

    # Out-of-fold predictions, so the scaling script can run a PAIRED test
    # against each language model on exactly the same genes.
    if not args.quick:
        np.savez_compressed(RESULTS / "oof_baseline.npz",
                            locus_tag=np.array(usable), y=y, **oof)
        print("Wrote results/oof_baseline.npz (for paired comparisons)")

    print("\nTop features by |coefficient| (standardised, logistic regression):")
    for d in importances["logistic_regression_abs_coef"][:12]:
        print(f"   {d['feature']:16} {d['coef']:+.4f}")
    print(f"\nWrote results/baseline.json and results/baseline.csv")


if __name__ == "__main__":
    main()
