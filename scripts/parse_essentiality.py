"""Join Goodall et al. 2018 essentiality calls onto the BW25113 gene table.

Source: Goodall ECA et al. (2018) "The Essential Genome of Escherichia coli
K-12", mBio 9:e02096-17, supplementary tables, downloaded from the publisher
(journals.asm.org/doi/10.1128/mbio.02096-17):

    Table S1  essentiality from the transposon library itself   <- primary
    Table S4  essentiality after outgrowth in LB                <- second condition

Both have one row per gene with an Insertion Index Score, a Log Likelihood
Ratio, and three mutually exclusive boolean columns: Essential, Non-essential,
Unclear. The tables identify genes by NAME, not by locus tag, so names are
resolved against the BW25113 annotation with the primary `gene` name taking
priority and `gene_synonym` used only as a fallback. Without that priority,
common names look ambiguous: acrB is the primary name of one gene and a
historical synonym of gyrB.

Outputs:
    data/processed/essentiality.csv   locus_tag, b_number, gene_name,
                                      source_gene, essential, call,
                                      insertion_index, log_likelihood, source
    data/processed/essentiality_join_report.json

Only rows that resolve to exactly one gene are written. Rows that cannot be
resolved, or that are ambiguous, are counted and listed in the report rather
than guessed at.

    .venv\\Scripts\\python scripts\\parse_essentiality.py [--table st1|st4|both]
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import openpyxl
from Bio import SeqIO

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
GENOME = RAW / "CP009273.1.gb"

TABLES = {
    "st1": ("goodall2018_st1.xlsx", "TraDIS transposon library (primary)"),
    "st4": ("goodall2018_st4.xlsx", "TraDIS after outgrowth in LB"),
}
CLASSES = ("Essential", "Non-essential", "Unclear")


def name_index(genbank):
    """Primary gene names and synonyms, kept separate so primary wins."""
    primary, synonym = defaultdict(set), defaultdict(set)
    record = SeqIO.read(genbank, "genbank")
    for f in record.features:
        if f.type != "gene" or "locus_tag" not in f.qualifiers:
            continue
        tag = f.qualifiers["locus_tag"][0]
        for n in f.qualifiers.get("gene", []):
            primary[n].add(tag)
        for value in f.qualifiers.get("gene_synonym", []):
            for n in value.split(";"):
                if n.strip():
                    synonym[n.strip()].add(tag)
    return primary, synonym


def resolve(name, primary, synonym):
    hits = primary.get(name) or synonym.get(name) or set()
    return next(iter(hits)) if len(hits) == 1 else None


def read_table(path):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.worksheets[0]
    header = None
    rows = []
    for raw in ws.iter_rows(values_only=True):
        cells = ["" if c is None else str(c).strip() for c in raw]
        if header is None:
            if cells[0] == "Gene":           # row 0 is a title; row 1 is the header
                header = cells
            continue
        if cells[0]:
            rows.append(dict(zip(header, cells)))
    wb.close()
    return rows


def call_of(row):
    flags = [c for c in CLASSES if row.get(c, "").lower() == "true"]
    return flags[0] if len(flags) == 1 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--table", choices=["st1", "st4", "both"], default="st1")
    args = parser.parse_args()
    wanted = ["st1", "st4"] if args.table == "both" else [args.table]

    genes = {}
    with (PROCESSED / "genes.csv").open() as f:
        for row in csv.DictReader(f):
            genes[row["locus_tag"]] = row
    primary, synonym = name_index(GENOME)

    out_rows, report = [], {}
    for key in wanted:
        filename, description = TABLES[key]
        rows = read_table(RAW / filename)
        classes = Counter()
        resolved, unresolved, malformed = {}, [], []
        for row in rows:
            name = row["Gene"]
            call = call_of(row)
            if call is None:
                malformed.append(name)
                continue
            classes[call] += 1
            tag = resolve(name, primary, synonym)
            if tag is None:
                unresolved.append(name)
                continue
            resolved.setdefault(tag, []).append((name, call, row))

        collisions = {t: [n for n, _, _ in v] for t, v in resolved.items() if len(v) > 1}
        written = Counter()
        for tag, entries in resolved.items():
            if len(entries) > 1:          # two source rows for one gene: cannot pick
                continue
            name, call, row = entries[0]
            gene = genes[tag]
            written[call] += 1
            out_rows.append({
                "locus_tag": tag,
                "b_number": gene["b_number"],
                "gene_name": gene["gene_name"],
                "source_gene": name,
                "essential": {"Essential": "True", "Non-essential": "False", "Unclear": ""}[call],
                "call": call,
                "insertion_index": row.get("Insertion Index Score", ""),
                "log_likelihood": row.get("Log Likelihood Ratio", ""),
                "source": f"Goodall2018_{key.upper()}",
            })

        report[key] = {
            "file": filename,
            "description": description,
            "rows_in_file": len(rows),
            "classes_in_file": {c: classes[c] for c in CLASSES},
            "malformed_rows": malformed,
            "unresolved_names": sorted(set(unresolved)),
            "unresolved_count": len(unresolved),
            "collisions": collisions,
            "written": {c: written[c] for c in CLASSES},
            "written_total": sum(written.values()),
        }
        print(f"\n== {key.upper()}: {description}")
        print(f"   rows in file     {len(rows)}")
        print(f"   classes in file  " + ", ".join(f"{c} {classes[c]}" for c in CLASSES))
        print(f"   unresolved names {len(unresolved)} {sorted(set(unresolved))[:10]}")
        print(f"   collisions       {len(collisions)} {list(collisions.items())[:4]}")
        print(f"   written          {sum(written.values())} "
              + ", ".join(f"{c} {written[c]}" for c in CLASSES))

    PROCESSED.mkdir(parents=True, exist_ok=True)
    with (PROCESSED / "essentiality.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)

    # Coverage of the gene table, counted on the primary source only.
    st1 = [r for r in out_rows if r["source"] == "Goodall2018_ST1"]
    if st1:
        labelled = {r["locus_tag"] for r in st1}
        coding = {t for t, g in genes.items() if g["biotype"] == "protein_coding"}
        usable = [r for r in st1 if r["call"] != "Unclear" and r["locus_tag"] in coding]
        n_ess = sum(r["call"] == "Essential" for r in usable)
        report["coverage_st1"] = {
            "genes_in_table": len(genes),
            "protein_coding": len(coding),
            "labelled": len(labelled),
            "protein_coding_labelled": len(labelled & coding),
            "protein_coding_unlabelled": len(coding - labelled),
            "usable_protein_coding_excluding_unclear": len(usable),
            "usable_essential": n_ess,
            "usable_non_essential": len(usable) - n_ess,
            "imbalance_ratio": round((len(usable) - n_ess) / n_ess, 2) if n_ess else None,
            "majority_class_fraction": round((len(usable) - n_ess) / len(usable), 4) if usable else None,
        }
        print("\n== Coverage of the gene table (ST1)")
        for k, v in report["coverage_st1"].items():
            print(f"   {k:45} {v}")

    (PROCESSED / "essentiality_join_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {PROCESSED.relative_to(ROOT)}/essentiality.csv ({len(out_rows)} rows) "
          f"and essentiality_join_report.json")


if __name__ == "__main__":
    main()
