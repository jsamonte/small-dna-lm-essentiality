"""The scaling comparison: does more model buy more essentiality signal?

Trains the same classifiers, on the same canonical folds, over each model's
cached embeddings, and plots performance against parameter count and against
inference time. The classical-features and majority-class baselines are drawn
on both plots, because a language model that cannot clear cheap features has
not earned its parameters.

Also runs the leakage check, which is not optional. "The embeddings encode gene
length" is the most likely boring explanation for any positive result, so this
reports the correlation between each model's embedding norm and gene length,
and how well length alone does.

    .venv\\Scripts\\python scripts\\evaluate_scaling.py
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.embeddings import cache_path  # noqa: E402
from src.models import MODELS  # noqa: E402
from src.splits import N_SPLITS, SEED, load_folds  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
POOLING, STRATEGY = "mean", "chunk_mean"


def classifiers():
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED)),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.06, early_stopping=True,
            validation_fraction=0.15, class_weight="balanced", random_state=SEED),
    }


def cross_validate(X, y, fold_ids, name):
    pred = np.zeros(len(y), dtype=int)
    score = np.zeros(len(y), dtype=float)
    for fold in range(N_SPLITS):
        test = fold_ids == fold
        model = classifiers()[name]
        model.fit(X[~test], y[~test])
        pred[test] = model.predict(X[test])
        score[test] = model.predict_proba(X[test])[:, 1]
    return pred, score


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
        "non_essential_precision": round(p[0], 4),
        "non_essential_recall": round(r[0], 4),
    }


def paired_mcc_difference(y, pred_a, pred_b, n=2000, seed=SEED):
    """Bootstrap MCC(a) - MCC(b) on the SAME resampled genes.

    Comparing two separate confidence intervals and checking whether they
    overlap is a weak test: both models are scored on identical genes and
    identical folds, so the comparison should be paired. If the interval for
    the difference excludes zero, the gap is real.
    """
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if 0 < y[idx].sum() < len(idx):
            diffs.append(matthews_corrcoef(y[idx], pred_a[idx])
                         - matthews_corrcoef(y[idx], pred_b[idx]))
    diffs = np.asarray(diffs)
    return {
        "delta_mcc": round(float(np.mean(diffs)), 4),
        "ci95": (round(float(np.percentile(diffs, 2.5)), 4),
                 round(float(np.percentile(diffs, 97.5)), 4)),
        "fraction_above_zero": round(float((diffs > 0).mean()), 4),
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


def load_common():
    folds = load_folds()
    tags = sorted(folds)
    labels = {}
    for r in csv.DictReader((PROCESSED / "essentiality.csv").open()):
        if r["source"] == "Goodall2018_ST1" and r["call"] in ("Essential", "Non-essential"):
            labels[r["locus_tag"]] = r["call"] == "Essential"
    lengths = {r["locus_tag"]: int(r["length"])
               for r in csv.DictReader((PROCESSED / "genes.csv").open())}
    y = np.array([labels[t] for t in tags], dtype=int)
    fold_ids = np.array([folds[t] for t in tags])
    gene_len = np.array([lengths[t] for t in tags], dtype=float)
    return tags, y, fold_ids, gene_len


def trial_seconds(model_key):
    path = RESULTS / f"embedding_trial_{model_key}_{POOLING}_{STRATEGY}.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return round(d["projected_seconds"] / d["n_genes"], 4)


def plot(rows, baseline, majority, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for xkey, xlabel, fname, logx in (
        ("params", "Model parameters", "scaling_mcc_vs_params.png", True),
        ("seconds_per_gene", "Inference seconds per gene (CPU)", "scaling_mcc_vs_time.png", True),
    ):
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for clf, marker in (("gradient_boosting", "o"), ("logistic_regression", "s")):
            pts = sorted([r for r in rows if r["classifier"] == clf and r.get(xkey)],
                         key=lambda r: r[xkey])
            if not pts:
                continue
            x = [p[xkey] for p in pts]
            ymcc = [p["mcc"] for p in pts]
            lo = [p["mcc"] - p["mcc_ci95"][0] for p in pts]
            hi = [p["mcc_ci95"][1] - p["mcc"] for p in pts]
            ax.errorbar(x, ymcc, yerr=[lo, hi], marker=marker, capsize=3,
                        label=f"DNA LM + {clf.replace('_', ' ')}")
            if clf == "gradient_boosting":      # label once, not twice
                for i, p in enumerate(pts):
                    ax.annotate(p["model"], (p[xkey], p["mcc"]), textcoords="offset points",
                                xytext=(0, 11 if i % 2 == 0 else -17), ha="center", fontsize=7)
        best = max(baseline.values(), key=lambda m: m["mcc"])
        ax.axhline(best["mcc"], color="crimson", ls="--", lw=1.4,
                   label=f"classical features (MCC {best['mcc']:.3f})")
        ax.axhspan(best["mcc_ci95"][0], best["mcc_ci95"][1], color="crimson", alpha=0.08)
        ax.axhline(0.0, color="grey", ls=":", lw=1.2,
                   label=f"majority class (MCC 0, acc {majority:.3f})")
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Matthews correlation coefficient")
        ax.set_title("Gene essentiality in E. coli BW25113: signal vs model size")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=200)
        plt.close(fig)
        print(f"   wrote results/{fname}")


def main():
    tags, y, fold_ids, gene_len = load_common()
    print(f"{len(tags):,} genes, {y.sum()} essential, folds from cv_folds.csv")

    baseline = {}
    bl = json.loads((RESULTS / "baseline.json").read_text())
    for key, m in bl["results"].items():
        if key.startswith("all__"):
            baseline[key] = m
    majority = bl["dataset"]["majority_class_accuracy"]

    # Out-of-fold baseline predictions, for the paired comparison.
    oof_path = RESULTS / "oof_baseline.npz"
    baseline_oof = None
    if oof_path.exists():
        with np.load(oof_path, allow_pickle=True) as z:
            if list(z["locus_tag"]) == tags and np.array_equal(z["y"], y):
                baseline_oof = {k: z[k] for k in z.files if k not in ("locus_tag", "y")}
            else:
                print("!! oof_baseline.npz does not match the current gene set; "
                      "re-run scripts/baseline_features.py")

    rows, leakage, comparisons = [], {}, {}
    for model_key, spec in MODELS.items():
        path = cache_path(model_key, POOLING, STRATEGY)
        if not path.exists():
            print(f"!! no cache for {model_key}, skipping")
            continue
        with np.load(path) as z:
            X = np.stack([z[t] for t in tags]).astype(np.float64)

        norms = np.linalg.norm(X, axis=1)
        r_len = float(np.corrcoef(norms, gene_len)[0, 1])
        leakage[model_key] = {
            "pearson_r_embedding_norm_vs_gene_length": round(r_len, 4),
            "r_squared": round(r_len ** 2, 4),
        }

        for clf in classifiers():
            pred, score = cross_validate(X, y, fold_ids, clf)
            m = evaluate(y, pred, score)
            m.update({
                "model": model_key, "model_id": spec["id"], "classifier": clf,
                "params": spec["params"], "dim": X.shape[1],
                "seconds_per_gene": trial_seconds(model_key),
                "mcc_ci95": bootstrap_mcc(y, pred),
            })
            rows.append(m)
            print(f"  {model_key:20} {clf:20} MCC {m['mcc']:.4f} CI {m['mcc_ci95']} "
                  f"AUROC {m['auroc']:.4f}  ess recall {m['essential_recall']:.3f}")
            if baseline_oof is not None:
                best_key = max((k for k in baseline_oof if k.startswith("all__")),
                               key=lambda k: bl["results"][k]["mcc"])
                comparisons[f"{model_key}__{clf}"] = paired_mcc_difference(
                    y, pred, baseline_oof[best_key]) | {"versus": best_key}

    if comparisons:
        print("\nPaired comparison against the best classical baseline "
              "(negative = the language model is worse)")
        for key, c in sorted(comparisons.items(), key=lambda kv: kv[1]["delta_mcc"]):
            lo, hi = c["ci95"]
            verdict = ("BEATS baseline" if lo > 0 else
                       "worse than baseline" if hi < 0 else "indistinguishable")
            print(f"   {key:40} dMCC {c['delta_mcc']:+.4f} "
                  f"CI ({lo:+.4f}, {hi:+.4f})  {verdict}")

    # Leakage control: how far does gene length alone get you?
    len_only = bl["results"].get("length_only__gradient_boosting", {})
    print("\nLeakage check -- embedding norm vs gene length")
    for k, v in leakage.items():
        print(f"   {k:20} r={v['pearson_r_embedding_norm_vs_gene_length']:+.3f} "
              f"(r^2={v['r_squared']:.3f})")
    print(f"   length-only classifier MCC: {len_only.get('mcc')}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "scaling.csv").open("w", newline="") as f:
        cols = ["model", "model_id", "params", "dim", "classifier", "seconds_per_gene",
                "mcc", "mcc_lo", "mcc_hi", "auroc", "average_precision", "accuracy",
                "essential_precision", "essential_recall", "essential_f1",
                "non_essential_precision", "non_essential_recall"]
        w = csv.writer(f)
        w.writerow(cols)
        for r in sorted(rows, key=lambda r: (r["classifier"], r["params"])):
            w.writerow([r["model"], r["model_id"], r["params"], r["dim"], r["classifier"],
                        r["seconds_per_gene"], r["mcc"], r["mcc_ci95"][0], r["mcc_ci95"][1],
                        r["auroc"], r["average_precision"], r["accuracy"],
                        r["essential_precision"], r["essential_recall"], r["essential_f1"],
                        r["non_essential_precision"], r["non_essential_recall"]])
    (RESULTS / "scaling.json").write_text(json.dumps({
        "pooling": POOLING, "strategy": STRATEGY,
        "dataset": bl["dataset"], "baseline": baseline,
        "results": rows, "leakage": leakage, "paired_vs_baseline": comparisons,
    }, indent=2) + "\n")

    plot(rows, baseline, majority, RESULTS)
    print("\nWrote results/scaling.csv and results/scaling.json")


if __name__ == "__main__":
    main()
