"""Reproduce every number and figure in results/ from a fresh clone.

    python run_all.py --quick     smoke test, a few minutes, results are not publishable
    python run_all.py             the real thing, roughly two hours on a laptop CPU

Stages are skipped when their outputs already exist, so an interrupted run
resumes; --force redoes everything. Embeddings are cached per gene, so even a
killed embedding stage picks up where it stopped.

The only thing this cannot fetch is the Goodall et al. supplementary tables:
PMC blocks automated downloads and the publisher sits behind a bot check. The
two files are committed to data/raw/ for exactly that reason, and this script
stops with instructions if they are missing.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
EMB = ROOT / "data" / "embeddings"
MODELS = ["hyenadna-tiny", "hyenadna-tiny-d256", "nt-v2-50m", "nt-v2-100m"]

REQUIRED_RAW = {
    "data/raw/goodall2018_st1.xlsx":
        "Goodall et al. 2018 Table S1. Download from "
        "https://journals.asm.org/doi/10.1128/mbio.02096-17 (Supplemental Material).",
    "data/raw/goodall2018_st4.xlsx": "Goodall et al. 2018 Table S4, from the same page.",
}


def check_inputs():
    missing = [(p, why) for p, why in REQUIRED_RAW.items() if not (ROOT / p).exists()]
    if missing:
        print("Missing input files that cannot be downloaded automatically:\n")
        for p, why in missing:
            print(f"  {p}\n      {why}\n")
        sys.exit(1)


def run(label, args, produces, force, quick_args=None, quick=False):
    outputs = [ROOT / p for p in produces]
    if not force and all(o.exists() for o in outputs):
        print(f"[skip] {label} (outputs present)")
        return 0.0
    cmd = [PY, "-W", "ignore", str(ROOT / args[0])] + args[1:]
    if quick and quick_args:
        cmd += quick_args
    print(f"\n[run ] {label}\n       {' '.join(cmd[2:])}", flush=True)
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        sys.exit(f"FAILED: {label}")
    dt = time.perf_counter() - t0
    print(f"[done] {label} in {dt / 60:.1f} min")
    return dt


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="tiny smoke run, not publishable")
    ap.add_argument("--force", action="store_true", help="redo stages even if outputs exist")
    ap.add_argument("--skip-generalization", action="store_true")
    args = ap.parse_args()
    check_inputs()

    total = 0.0
    total += run("1. model load check", ["scripts/check_models.py"],
                 ["results/model_check.csv"], args.force)
    total += run("2. E. coli genome (BW25113)", ["scripts/download_genome.py"],
                 ["data/processed/genes.csv", "data/processed/genes.fasta"], args.force)
    total += run("3. essentiality labels", ["scripts/parse_essentiality.py", "--table", "both"],
                 ["data/processed/essentiality.csv"], args.force)
    total += run("4. classical baseline", ["scripts/baseline_features.py"],
                 ["results/baseline.csv", "results/oof_baseline.npz"], args.force,
                 quick_args=["--quick"], quick=args.quick)

    for model in MODELS:
        total += run(f"5. embeddings: {model}",
                     ["scripts/compute_embeddings.py", "--model", model],
                     [f"data/embeddings/{model}__mean__chunk_mean.npz"], args.force,
                     quick_args=["--limit", "200"], quick=args.quick)
        run(f"5t. timing trial: {model}",
            ["scripts/compute_embeddings.py", "--model", model, "--trial"],
            [f"results/embedding_trial_{model}_mean_chunk_mean.json"], args.force)

    total += run("6. scaling comparison", ["scripts/evaluate_scaling.py"],
                 ["results/scaling.csv", "results/scaling_mcc_vs_params.png"], args.force)

    if not args.skip_generalization:
        total += run("7. B. subtilis genome and labels", ["scripts/download_bsubtilis.py"],
                     ["data/processed/bsub_genes.csv", "data/processed/bsub_essentiality.csv"],
                     args.force)
        for model in MODELS:
            total += run(f"8. B. subtilis embeddings: {model}",
                         ["scripts/compute_embeddings.py", "--model", model, "--organism", "bsub"],
                         [f"data/embeddings/bsub_{model}__mean__chunk_mean.npz"], args.force,
                         quick_args=["--limit", "200"], quick=args.quick)
        total += run("9. generalization", ["scripts/evaluate_generalization.py"],
                     ["results/generalization.csv"], args.force)

    print(f"\nAll stages complete in {total / 60:.1f} min. Results in results/")
    if args.quick:
        print("NOTE: --quick was used. These numbers are a smoke test, not results.")


if __name__ == "__main__":
    main()
