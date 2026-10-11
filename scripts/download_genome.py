"""Fetch the E. coli BW25113 genome from NCBI and extract every gene.

BW25113, not MG1655. BW25113 is the parent of the Keio collection and the
strain Goodall et al. 2018 actually assayed; they mapped their TraDIS reads to
CP009273.1, so their gene list IS this annotation. Pairing their essentiality
labels with MG1655 sequences would mean 15 measured genes could not be located
at all, and would describe a strain that was never assayed. Checked 2026-10-11:
every one of Goodall's 4,313 rows resolves against this record.

MG1655 (NC_000913.3) is still downloaded, but only as a cross-reference: both
annotations carry ECK numbers, which bridge BW25113 locus tags to the familiar
b-numbers for 98.6% of genes, so published work keyed on b-numbers stays
comparable.

Outputs:
    data/raw/CP009273.1.gb                  BW25113 GenBank record (gitignored)
    data/raw/NC_000913.3.gb                 MG1655, for the b-number bridge (gitignored)
    data/processed/genes.fasta              one record per gene, id = locus tag
    data/processed/genes.csv                locus_tag, gene_name, eck, b_number, gene_id,
                                            start, end, strand, length, biotype
    data/processed/genome_manifest.json     accessions, checksums, counts

start/end are 1-based inclusive genome coordinates; sequences run 5'->3' on the
gene's own strand.

Set NCBI_EMAIL in the environment to identify yourself to NCBI, as its usage
policy asks; the download works without it.

    .venv\\Scripts\\python scripts\\download_genome.py [--force]
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from Bio import Entrez, SeqIO

ACCESSION = "CP009273.1"          # E. coli BW25113, the assayed strain
EXPECTED_LENGTH = 4_631_469
XREF_ACCESSION = "NC_000913.3"    # E. coli K-12 MG1655, for ECK -> b-number
XREF_EXPECTED_LENGTH = 4_641_652

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

ECK = re.compile(r"^ECK\d{4}$")
B_NUMBER = re.compile(r"^b\d{4}$")
RNA_TYPES = ("tRNA", "rRNA", "ncRNA", "tmRNA", "misc_RNA", "RNase_P_RNA", "SRP_RNA", "antisense_RNA")


def download(accession, force):
    path = RAW / f"{accession}.gb"
    if path.exists() and not force:
        print(f"Using cached {path.relative_to(ROOT)} (pass --force to re-download)")
        return path
    Entrez.email = os.environ.get("NCBI_EMAIL")
    Entrez.tool = "small-dna-lm-essentiality"
    print(f"Downloading {accession} from NCBI ...", flush=True)
    with Entrez.efetch(db="nuccore", id=accession, rettype="gbwithparts", retmode="text") as handle:
        text = handle.read()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def verify(record, accession, expected_length, expect_strain):
    source = next(f for f in record.features if f.type == "source")
    q = {k: v[0] for k, v in source.qualifiers.items()}
    descriptors = " ".join(filter(None, (q.get("organism"), q.get("strain"), q.get("sub_strain"))))
    problems = []
    if record.id != accession:
        problems.append(f"record id {record.id!r}, expected {accession!r}")
    if "Escherichia coli" not in q.get("organism", ""):
        problems.append(f"organism {q.get('organism')!r}")
    if expect_strain not in descriptors:
        problems.append(f"{expect_strain!r} not in {descriptors!r}")
    if len(record.seq) != expected_length:
        problems.append(f"length {len(record.seq):,}, expected {expected_length:,}")
    if problems:
        sys.exit(f"{accession} is not the expected record: " + "; ".join(problems))
    print(f"Verified: {record.id} | {descriptors} | {len(record.seq):,} bp")
    return q


def synonyms(feature):
    out = set()
    for value in feature.qualifiers.get("gene_synonym", []):
        out.update(s.strip() for s in value.split(";") if s.strip())
    return out


def first_match(names, pattern):
    return next((n for n in sorted(names) if pattern.match(n)), "")


def eck_to_b_number(record):
    """ECK number -> b-number, read off the MG1655 annotation."""
    mapping = {}
    for f in record.features:
        if f.type != "gene":
            continue
        tag = f.qualifiers["locus_tag"][0]
        if not B_NUMBER.match(tag):
            continue
        eck = first_match(synonyms(f), ECK)
        if eck:
            mapping[eck] = tag
    return mapping


def biotype(gene, other_types):
    if "pseudo" in gene.qualifiers or "pseudogene" in gene.qualifiers:
        return "pseudogene"
    if "CDS" in other_types:
        return "protein_coding"
    for rna in RNA_TYPES:
        if rna in other_types:
            return rna
    return "other"


def extract_genes(record, b_numbers):
    types_by_tag = defaultdict(set)
    for f in record.features:
        if f.type != "gene" and "locus_tag" in f.qualifiers:
            types_by_tag[f.qualifiers["locus_tag"][0]].add(f.type)

    rows, records = [], []
    for f in record.features:
        if f.type != "gene" or "locus_tag" not in f.qualifiers:
            continue
        tag = f.qualifiers["locus_tag"][0]
        name = f.qualifiers.get("gene", [tag])[0]
        eck = first_match(synonyms(f), ECK)
        gene_id = next((x.split(":", 1)[1] for x in f.qualifiers.get("db_xref", []) if x.startswith("GeneID:")), "")
        seq = f.extract(record.seq)
        if len(seq) != len(f.location):
            sys.exit(f"{tag}: extracted {len(seq)} bases but the location spans {len(f.location)}")
        kind = biotype(f, types_by_tag[tag])
        # A handful of genes (ribosomal frameshifts, split features) have compound
        # locations, so their sequence is shorter than start..end. Record the part
        # count rather than silently flattening it; length is always the sequence.
        rows.append({
            "locus_tag": tag,
            "gene_name": name,
            "eck": eck,
            "b_number": b_numbers.get(eck, ""),
            "gene_id": gene_id,
            "start": int(f.location.start) + 1,
            "end": int(f.location.end),
            "strand": "+" if f.location.strand == 1 else "-",
            "length": len(seq),
            "segments": len(f.location.parts),
            "biotype": kind,
        })
        records.append((f"{tag} {name} eck={eck} biotype={kind}", str(seq)))
    return rows, records


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args()

    path = download(ACCESSION, args.force)
    record = SeqIO.read(path, "genbank")
    source = verify(record, ACCESSION, EXPECTED_LENGTH, "BW25113")

    xref_path = download(XREF_ACCESSION, args.force)
    xref = SeqIO.read(xref_path, "genbank")
    verify(xref, XREF_ACCESSION, XREF_EXPECTED_LENGTH, "MG1655")
    b_numbers = eck_to_b_number(xref)
    print(f"ECK -> b-number pairs from {XREF_ACCESSION}: {len(b_numbers):,}")

    rows, records = extract_genes(record, b_numbers)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    with (PROCESSED / "genes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (PROCESSED / "genes.fasta").open("w") as f:
        for header, seq in records:
            f.write(f">{header}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i:i + 80] + "\n")

    tags = [r["locus_tag"] for r in rows]
    counts = Counter(r["biotype"] for r in rows)
    with_b = sum(1 for r in rows if r["b_number"])
    compound = [r["locus_tag"] for r in rows if r["segments"] > 1]
    manifest = {
        "accession": ACCESSION,
        "description": record.description,
        "organism": source["organism"],
        "strain": source.get("strain", ""),
        "genome_length": len(record.seq),
        "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "raw_downloaded_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
        "xref_accession": XREF_ACCESSION,
        "xref_sha256": hashlib.sha256(xref_path.read_bytes()).hexdigest(),
        "eck_to_b_number_pairs": len(b_numbers),
        "genes_total": len(rows),
        "genes_with_b_number": with_b,
        "genes_by_biotype": dict(counts.most_common()),
        "genes_with_compound_location": compound,
        "duplicate_locus_tags": [t for t, n in Counter(tags).items() if n > 1],
    }
    (PROCESSED / "genome_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\nGenes extracted: {len(rows):,}")
    for kind, n in counts.most_common():
        print(f"  {kind:15} {n:,}")
    print(f"With a b-number cross-reference: {with_b:,} ({with_b / len(rows):.1%})")
    print(f"Compound (multi-segment) locations: {len(compound)} {compound[:5]}")
    print(f"\nWrote {PROCESSED.relative_to(ROOT)}/genes.csv, genes.fasta, genome_manifest.json")


if __name__ == "__main__":
    main()
