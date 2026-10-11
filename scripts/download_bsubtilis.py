"""Fetch the B. subtilis 168 genome and its essentiality labels.

The second organism, used only as a held-out test: models are trained on
E. coli and applied here without retraining.

Genome: NC_000964.3, Bacillus subtilis subsp. subtilis str. 168, 4,215,606 bp.
Labels: SubtiWiki v5 (subtiwiki.uni-goettingen.de), the B. subtilis community
database, whose per-gene `essential` field is curated from the published
deletion-library work. Its counts (258 essential) agree with Koo et al. 2017
Cell Systems, which reported ~257. Rows marked "quasi" or "-" are dropped, the
same treatment given to Goodall's "Unclear" class in E. coli.

SubtiWiki is a live database with no version stamp, so the raw export is
cached and its sha256 recorded; re-running will not silently change the labels
underneath a finished result.

Outputs:
    data/raw/NC_000964.3.gb                 (gitignored)
    data/raw/subtiwiki_export.json          cached label export
    data/processed/bsub_genes.csv
    data/processed/bsub_genes.fasta
    data/processed/bsub_essentiality.csv
    data/processed/bsub_manifest.json

    .venv\\Scripts\\python scripts\\download_bsubtilis.py [--force]
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from Bio import Entrez, SeqIO

ACCESSION = "NC_000964.3"
EXPECTED_LENGTH = 4_215_606
API = "https://subtiwiki.uni-goettingen.de/v5/api/gene/export"
FIELDS = ["name", "essential", "genomic_annotation.locus_tag"]

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RNA_TYPES = ("tRNA", "rRNA", "ncRNA", "tmRNA", "misc_RNA")

ESSENTIAL_YES = {"yes"}
ESSENTIAL_NO = {"no"}
# "quasi" (6 genes) and "-" are neither clear call, so they are excluded, as
# Goodall's "Unclear" class is for E. coli.


def download_genome(force):
    path = RAW / f"{ACCESSION}.gb"
    if path.exists() and not force:
        print(f"Using cached {path.relative_to(ROOT)}")
        return path
    Entrez.email = os.environ.get("NCBI_EMAIL")
    Entrez.tool = "small-dna-lm-essentiality"
    print(f"Downloading {ACCESSION} ...", flush=True)
    with Entrez.efetch(db="nuccore", id=ACCESSION, rettype="gbwithparts", retmode="text") as h:
        path.write_text(h.read())
    return path


def download_labels(force):
    path = RAW / "subtiwiki_export.json"
    if path.exists() and not force:
        print(f"Using cached {path.relative_to(ROOT)}")
        return path
    print("Querying SubtiWiki v5 ...", flush=True)
    body = json.dumps({"fields": FIELDS, "format": "json"}).encode()
    req = urllib.request.Request(API, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        path.write_bytes(r.read())
    return path


def verify(record):
    source = next(f for f in record.features if f.type == "source")
    q = {k: v[0] for k, v in source.qualifiers.items()}
    problems = []
    if record.id != ACCESSION:
        problems.append(f"id {record.id!r}")
    if "Bacillus subtilis" not in q.get("organism", ""):
        problems.append(f"organism {q.get('organism')!r}")
    if "168" not in f"{q.get('strain', '')} {q.get('sub_species', '')} {record.description}":
        problems.append("strain 168 not named")
    if len(record.seq) != EXPECTED_LENGTH:
        problems.append(f"length {len(record.seq):,}")
    if problems:
        sys.exit("Not the expected B. subtilis record: " + "; ".join(problems))
    print(f"Verified: {record.id} | {q['organism']} | {len(record.seq):,} bp")


def biotype(gene, other_types):
    if "pseudo" in gene.qualifiers or "pseudogene" in gene.qualifiers:
        return "pseudogene"
    if "CDS" in other_types:
        return "protein_coding"
    for rna in RNA_TYPES:
        if rna in other_types:
            return rna
    return "other"


def extract_genes(record):
    types_by_tag = defaultdict(set)
    for f in record.features:
        if f.type != "gene" and "locus_tag" in f.qualifiers:
            types_by_tag[f.qualifiers["locus_tag"][0]].add(f.type)
    rows, seqs = [], []
    for f in record.features:
        if f.type != "gene" or "locus_tag" not in f.qualifiers:
            continue
        tag = f.qualifiers["locus_tag"][0]
        name = f.qualifiers.get("gene", [tag])[0]
        seq = f.extract(record.seq)
        kind = biotype(f, types_by_tag[tag])
        rows.append({
            "locus_tag": tag, "gene_name": name,
            "start": int(f.location.start) + 1, "end": int(f.location.end),
            "strand": "+" if f.location.strand == 1 else "-",
            "length": len(seq), "segments": len(f.location.parts), "biotype": kind,
        })
        seqs.append((f"{tag} {name} biotype={kind}", str(seq)))
    return rows, seqs


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    genome_path = download_genome(args.force)
    record = SeqIO.read(genome_path, "genbank")
    verify(record)
    rows, seqs = extract_genes(record)

    labels_path = download_labels(args.force)
    payload = json.loads(labels_path.read_text())["data"]
    header, records = payload["header"], payload["records"]
    i_ess = header.index("essential")
    i_tag = header.index("genomic_annotation.locus_tag")
    i_name = header.index("name")

    known = {r["locus_tag"] for r in rows}
    raw_values = Counter()
    labels, unmatched, no_tag = {}, [], 0
    for rec in records:
        tag, value = rec[i_tag], (rec[i_ess] or "").strip().lower()
        raw_values[value or "(null)"] += 1
        if not tag:
            no_tag += 1
            continue
        if value in ESSENTIAL_YES:
            call = "Essential"
        elif value in ESSENTIAL_NO:
            call = "Non-essential"
        else:
            continue                       # quasi, "-", or missing
        if tag not in known:
            unmatched.append(tag)
            continue
        labels[tag] = (call, rec[i_name])

    PROCESSED.mkdir(parents=True, exist_ok=True)
    with (PROCESSED / "bsub_genes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with (PROCESSED / "bsub_genes.fasta").open("w") as f:
        for head, seq in seqs:
            f.write(f">{head}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i:i + 80] + "\n")
    with (PROCESSED / "bsub_essentiality.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["locus_tag", "gene_name", "essential", "call", "source"])
        for tag, (call, name) in sorted(labels.items()):
            w.writerow([tag, name, call == "Essential", call, "SubtiWiki_v5"])

    by_type = Counter(r["biotype"] for r in rows)
    coding = {r["locus_tag"] for r in rows if r["biotype"] == "protein_coding"}
    usable = {t for t in labels if t in coding}
    n_ess = sum(1 for t in usable if labels[t][0] == "Essential")
    manifest = {
        "accession": ACCESSION, "organism": record.annotations.get("organism"),
        "genome_length": len(record.seq),
        "genome_sha256": hashlib.sha256(genome_path.read_bytes()).hexdigest(),
        "labels_source": API,
        "labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
        "labels_downloaded_utc": datetime.fromtimestamp(
            labels_path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
        "subtiwiki_rows": len(records),
        "subtiwiki_raw_values": dict(raw_values.most_common()),
        "rows_without_locus_tag": no_tag,
        "unmatched_locus_tags": sorted(set(unmatched))[:50],
        "unmatched_count": len(unmatched),
        "genes_total": len(rows), "genes_by_biotype": dict(by_type.most_common()),
        "labelled": len(labels),
        "usable_protein_coding": len(usable),
        "usable_essential": n_ess,
        "usable_non_essential": len(usable) - n_ess,
        "majority_class_fraction": round((len(usable) - n_ess) / len(usable), 4),
    }
    (PROCESSED / "bsub_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\nGenes: {len(rows):,} ({dict(by_type.most_common(4))})")
    print(f"SubtiWiki raw values: {dict(raw_values.most_common())}")
    print(f"Unmatched locus tags: {len(unmatched)}")
    print(f"Usable protein-coding with a clear call: {len(usable):,} "
          f"({n_ess} essential, {len(usable) - n_ess} non-essential, "
          f"majority {manifest['majority_class_fraction']:.4f})")
    print("\nWrote bsub_genes.csv, bsub_genes.fasta, bsub_essentiality.csv, bsub_manifest.json")


if __name__ == "__main__":
    main()
