"""Train on E. coli, test on B. subtilis, with no retraining.

This asks whether anything learned transfers between organisms, or whether the
models only learned E. coli's particular composition. Each representation
(classical features, and each language model's embeddings) trains one
classifier on all 3,929 E. coli genes and is then applied unchanged to 4,046
B. subtilis genes.

Two things make the comparison honest:

  * the classical features are recomputed from B. subtilis sequences with a
    B. subtilis CAI reference set, because codon usage is organism-specific and
    carrying E. coli's weights over would be a bug, not a transfer test;
  * the majority-class rate differs between the two organisms (91.0% vs 93.7%),
    so MCC is reported for both and accuracy is never compared across them.

    .venv\\Scripts\\python scripts\\evaluate_generalization.py
"""

import csv
import json
import sys
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

from src.embeddings import CACHE_DIR  # noqa: E402
from src.features import feature_names, features_for, relative_adaptiveness  # noqa: E402
from src.models import MODELS  # noqa: E402
from src.splits import SEED, load_folds  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
POOLING, STRATEGY = "mean", "chunk_mean"
RIBOSOMAL = ("rps", "rpl", "rpm")


def classifiers():
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED)),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.06, early_stopping=True,
            validation_fraction=0.15, class_weight="balanced", random_state=SEED),
    }


def evaluate(y, pred, score):
    p, r, f1, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    return {
        "mcc": round(matthews_corrcoef(y, pred), 4),
        "auroc": round(roc_auc_score(y, score), 4),
        "average_precision": round(average_precision_score(y, score), 4),
        "accuracy": round(float((pred == y).mean()), 4),
        "essential_precision": round(p[1], 4),
        "essential_recall": round(r[1], 4),
        "essential_f1": round(f1[1], 4),
    }


def bootstrap_mcc(y, pred, n=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if 0 < y[idx].sum() < len(idx):
            vals.append(matthews_corrcoef(y[idx], pred[idx]))
    return (round(float(np.percentile(vals, 2.5)), 4),
            round(float(np.percentile(vals, 97.5)), 4))


def load_organism(prefix, label_source):
    gfile = "genes.csv" if not prefix else f"{prefix}genes.csv"
    ffile = "genes.fasta" if not prefix else f"{prefix}genes.fasta"
    efile = "essentiality.csv" if not prefix else f"{prefix}essentiality.csv"
    genes = {r["locus_tag"]: r for r in csv.DictReader((PROCESSED / gfile).open())}
    seqs = {r.id: str(r.seq).upper() for r in SeqIO.parse(PROCESSED / ffile, "fasta")}
    labels = {}
    for r in csv.DictReader((PROCESSED / efile).open()):
        if r.get("source", label_source) == label_source and r["call"] in ("Essential", "Non-essential"):
            labels[r["locus_tag"]] = r["call"] == "Essential"
    tags = sorted(t for t, g in genes.items()
                  if g["biotype"] == "protein_coding" and t in labels and t in seqs)
    return genes, seqs, labels, tags


def classical_matrix(genes, seqs, tags, label, weights=None):
    """Features for one organism, with a CAI reference that is its own unless
    one is passed in.

    Recomputing the reference inside the test organism is the right default,
    since codon usage is organism-specific and carrying E. coli weights over
    would be a bug. It is also an organism-specific recalibration that the
    frozen embeddings get no equivalent of, so the alternative is computed
    too and reported beside it as `classical_ecoli_cai`."""
    if weights is None:
        reference = [seqs[t] for t in tags if genes[t]["gene_name"].startswith(RIBOSOMAL)]
        print(f"   {label}: CAI reference = {len(reference)} ribosomal genes")
        weights = relative_adaptiveness(reference)
    else:
        print(f"   {label}: CAI reference carried over from the training organism")
    X = np.array([features_for(seqs[t], weights) for t in tags], dtype=np.float64)
    return X, weights


def main():
    ec_genes, ec_seqs, ec_labels, ec_tags = load_organism("", "Goodall2018_ST1")
    bs_genes, bs_seqs, bs_labels, bs_tags = load_organism("bsub_", "SubtiWiki_v5")
    # Train on exactly the genes the main results used.
    folds = load_folds()
    ec_tags = [t for t in ec_tags if t in folds]
    y_ec = np.array([ec_labels[t] for t in ec_tags], dtype=int)
    y_bs = np.array([bs_labels[t] for t in bs_tags], dtype=int)
    print(f"train E. coli    {len(ec_tags):,} genes, {y_ec.sum()} essential, "
          f"majority {max(y_ec.mean(), 1 - y_ec.mean()):.4f}")
    print(f"test  B. subtilis {len(bs_tags):,} genes, {y_bs.sum()} essential, "
          f"majority {max(y_bs.mean(), 1 - y_bs.mean()):.4f}")

    within = {r["model"] + "__" + r["classifier"]: r
              for r in json.loads((RESULTS / "scaling.json").read_text())["results"]}
    bl = json.loads((RESULTS / "baseline.json").read_text())

    representations = {}
    print("\nBuilding classical features")
    Xe_cl, ec_weights = classical_matrix(ec_genes, ec_seqs, ec_tags, "E. coli")
    Xb_own, _ = classical_matrix(bs_genes, bs_seqs, bs_tags, "B. subtilis")
    Xb_carried, _ = classical_matrix(bs_genes, bs_seqs, bs_tags, "B. subtilis",
                                     weights=ec_weights)
    representations["classical"] = (Xe_cl, Xb_own, None)
    # The same trained classifier applied to B. subtilis features built with
    # E. coli's codon weights, so the classical pipeline gets no more
    # organism-specific information than a frozen embedding does.
    representations["classical_ecoli_cai"] = (Xe_cl, Xb_carried, None)
    for key, spec in MODELS.items():
        ec_path = CACHE_DIR / f"{key}__{POOLING}__{STRATEGY}.npz"
        bs_path = CACHE_DIR / f"bsub_{key}__{POOLING}__{STRATEGY}.npz"
        if not (ec_path.exists() and bs_path.exists()):
            print(f"!! missing embeddings for {key}, skipping")
            continue
        with np.load(ec_path) as z:
            Xe = np.stack([z[t] for t in ec_tags]).astype(np.float64)
        with np.load(bs_path) as z:
            Xb = np.stack([z[t] for t in bs_tags]).astype(np.float64)
        representations[key] = (Xe, Xb, spec["params"])

    rows = []
    print()
    for rep, (Xe, Xb, params) in representations.items():
        for clf_name in classifiers():
            model = classifiers()[clf_name]
            model.fit(Xe, y_ec)
            pred = model.predict(Xb)
            score = model.predict_proba(Xb)[:, 1]
            m = evaluate(y_bs, pred, score)
            m["mcc_ci95"] = bootstrap_mcc(y_bs, pred)
            m["representation"] = rep
            m["classifier"] = clf_name
            m["params"] = params
            key = f"{rep}__{clf_name}"
            base_rep = "classical" if rep.startswith("classical") else rep
            source = (within.get(f"{base_rep}__{clf_name}")
                      or bl["results"].get("all__" + clf_name
                                           if base_rep == "classical" else ""))
            m["within_ecoli_mcc"] = source["mcc"] if source else None
            m["mcc_drop"] = (round(m["within_ecoli_mcc"] - m["mcc"], 4)
                             if m["within_ecoli_mcc"] is not None else None)
            rows.append(m)
            print(f"  {rep:20} {clf_name:20} B.sub MCC {m['mcc']:+.4f} CI {m['mcc_ci95']}  "
                  f"E.coli {m['within_ecoli_mcc']}  drop {m['mcc_drop']}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "generalization.csv").open("w", newline="") as f:
        cols = ["representation", "classifier", "params", "within_ecoli_mcc", "mcc",
                "mcc_lo", "mcc_hi", "mcc_drop", "auroc", "average_precision", "accuracy",
                "essential_precision", "essential_recall", "essential_f1"]
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r["representation"], r["classifier"], r["params"],
                        r["within_ecoli_mcc"], r["mcc"], r["mcc_ci95"][0], r["mcc_ci95"][1],
                        r["mcc_drop"], r["auroc"], r["average_precision"], r["accuracy"],
                        r["essential_precision"], r["essential_recall"], r["essential_f1"]])
    (RESULTS / "generalization.json").write_text(json.dumps({
        "train": {"organism": "E. coli BW25113", "n": len(ec_tags), "essential": int(y_ec.sum())},
        "test": {"organism": "B. subtilis 168", "n": len(bs_tags), "essential": int(y_bs.sum()),
                 "majority_class_fraction": round(float(max(y_bs.mean(), 1 - y_bs.mean())), 4)},
        "results": rows,
    }, indent=2) + "\n")
    print("\nWrote results/generalization.csv and generalization.json")


if __name__ == "__main__":
    main()
