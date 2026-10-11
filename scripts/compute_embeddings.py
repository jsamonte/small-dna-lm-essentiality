"""Embed every labelled gene with one model, caching as it goes.

Always run --trial first. It embeds 10 genes drawn across the length
distribution and projects the full runtime from the measured rate, so a long
job is never started blind.

    .venv\\Scripts\\python scripts\\compute_embeddings.py --model hyenadna-tiny --trial
    .venv\\Scripts\\python scripts\\compute_embeddings.py --model hyenadna-tiny
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from Bio import SeqIO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.embeddings import CACHE_DIR, POOLINGS, STRATEGIES, Embedder, EmbeddingCache  # noqa: E402
from src.models import MODELS  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


ORGANISMS = {
    # name: (genes csv, fasta, essentiality csv, cache prefix)
    "ecoli": ("genes.csv", "genes.fasta", "essentiality.csv", ""),
    "bsub": ("bsub_genes.csv", "bsub_genes.fasta", "bsub_essentiality.csv", "bsub_"),
}


def load_genes(organism="ecoli"):
    gfile, ffile, efile, _ = ORGANISMS[organism]
    genes = {r["locus_tag"]: r for r in csv.DictReader((PROCESSED / gfile).open())}
    labelled = {r["locus_tag"] for r in csv.DictReader((PROCESSED / efile).open())
                if r["call"] in ("Essential", "Non-essential")
                and r.get("source", "Goodall2018_ST1") in ("Goodall2018_ST1", "SubtiWiki_v5")}
    seqs = {r.id: str(r.seq).upper() for r in SeqIO.parse(PROCESSED / ffile, "fasta")}
    tags = sorted(t for t, g in genes.items()
                  if g["biotype"] == "protein_coding" and t in labelled and t in seqs)
    return [(t, seqs[t]) for t in tags]


def trial(embedder, items, pooling, strategy, n=10):
    """Time n genes spread across the length distribution, then project."""
    ordered = sorted(items, key=lambda kv: len(kv[1]))
    picks = [ordered[round(i * (len(ordered) - 1) / (n - 1))] for i in range(n)]
    embedder.embed(picks[0][1], pooling, strategy)          # warm-up, not timed

    rows, total = [], 0.0
    for tag, seq in picks:
        t0 = time.perf_counter()
        vec = embedder.embed(seq, pooling, strategy)
        dt = time.perf_counter() - t0
        total += dt
        rows.append({"locus_tag": tag, "bp": len(seq), "seconds": round(dt, 4), "dim": len(vec)})
        print(f"   {tag}  {len(seq):>5} bp  {dt:6.3f}s  dim {len(vec)}")

    mean_bp_all = sum(len(s) for _, s in items) / len(items)
    mean_bp_trial = sum(r["bp"] for r in rows) / len(rows)
    per_gene = total / len(rows)
    # Cost scales with length, so rescale by mean length rather than assuming
    # the trial sample is representative.
    projected = per_gene * len(items) * (mean_bp_all / mean_bp_trial)
    print(f"\n   trial mean {mean_bp_trial:.0f} bp, dataset mean {mean_bp_all:.0f} bp")
    print(f"   {per_gene:.3f} s/gene on the trial sample")
    print(f"   PROJECTED FULL RUN: {projected / 60:.1f} min for {len(items):,} genes")
    return rows, projected


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--organism", default="ecoli", choices=sorted(ORGANISMS))
    parser.add_argument("--pooling", default="mean", choices=POOLINGS)
    parser.add_argument("--strategy", default="chunk_mean", choices=STRATEGIES)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--trial", action="store_true", help="time 10 genes and project, then stop")
    parser.add_argument("--limit", type=int, help="only the first N genes (debugging)")
    args = parser.parse_args()

    items = load_genes(args.organism)
    if args.limit:
        items = items[:args.limit]
    print(f"{len(items):,} labelled protein-coding genes ({args.organism})")

    embedder = Embedder(args.model, device=args.device)
    if args.trial:
        rows, projected = trial(embedder, items, args.pooling, args.strategy)
        RESULTS.mkdir(parents=True, exist_ok=True)
        path = RESULTS / f"embedding_trial_{ORGANISMS[args.organism][3]}{args.model}_{args.pooling}_{args.strategy}.json"
        path.write_text(json.dumps({
            "model": args.model, "model_id": MODELS[args.model]["id"], "organism": args.organism,
            "pooling": args.pooling, "strategy": args.strategy,
            "device": embedder.device, "n_genes": len(items),
            "projected_seconds": round(projected, 1),
            "projected_minutes": round(projected / 60, 1),
            "trial": rows,
        }, indent=2) + "\n")
        print(f"Wrote {path.relative_to(ROOT)}")
        return

    prefix = ORGANISMS[args.organism][3]
    path = CACHE_DIR / f"{prefix}{args.model}__{args.pooling}__{args.strategy}.npz"
    cache = EmbeddingCache(path)
    t0 = time.perf_counter()
    cache = embedder.embed_many(items, args.pooling, args.strategy, cache=cache)
    dims = {v.shape[0] for v in cache.data.values()}
    print(f"\nDone in {(time.perf_counter() - t0) / 60:.1f} min. "
          f"{len(cache):,} vectors, dim {dims}, {path.stat().st_size / 1e6:.1f} MB")
    print(f"Cache: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
