"""Checks on the output of scripts/download_genome.py (E. coli BW25113).

The coordinate/strand tests matter most: a bug there would leave minus-strand
genes reverse-complemented wrongly, and their start and stop codons would stop
matching. That is checked per strand so a one-sided bug cannot hide.

test_strain_specific_deletions is the identity check. BW25113's published
genotype includes D(araD-araB)567, DlacZ4787 and D(rhaD-rhaB)568, so if those
genes look intact we have downloaded the wrong genome.
"""

import csv
import re
from pathlib import Path

import pytest
from Bio import SeqIO

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
STARTS = ("ATG", "GTG", "TTG")
STOPS = ("TAA", "TAG", "TGA")

if not (PROCESSED / "genes.csv").exists():
    pytest.skip("run scripts/download_genome.py first", allow_module_level=True)


@pytest.fixture(scope="module")
def genes():
    with (PROCESSED / "genes.csv").open() as f:
        return {row["locus_tag"]: row for row in csv.DictReader(f)}


@pytest.fixture(scope="module")
def seqs():
    return {r.id: str(r.seq) for r in SeqIO.parse(PROCESSED / "genes.fasta", "fasta")}


@pytest.fixture(scope="module")
def by_name(genes):
    return {g["gene_name"]: g for g in genes.values()}


def test_gene_count_in_expected_range(genes):
    assert 4200 <= len(genes) <= 4700, len(genes)
    coding = sum(g["biotype"] == "protein_coding" for g in genes.values())
    assert 4000 <= coding <= 4400, coding


def test_locus_tags_are_bw25113_and_unique(genes):
    assert all(re.fullmatch(r"BW25113_\d{4}", tag) for tag in genes)


def test_most_genes_carry_a_b_number_cross_reference(genes):
    with_b = sum(1 for g in genes.values() if g["b_number"])
    assert with_b / len(genes) >= 0.98, with_b / len(genes)
    assert all(re.fullmatch(r"b\d{4}", g["b_number"]) for g in genes.values() if g["b_number"])


def test_fasta_matches_csv(genes, seqs):
    assert seqs.keys() == genes.keys()
    assert all(len(seqs[tag]) == int(g["length"]) for tag, g in genes.items())
    # Single-segment genes span exactly their sequence; compound ones (ribosomal
    # frameshifts and split features) span more, never less.
    for tag, g in genes.items():
        span = int(g["end"]) - int(g["start"]) + 1
        if int(g["segments"]) == 1:
            assert span == int(g["length"]), (tag, span, g["length"])
        else:
            assert span > int(g["length"]), (tag, span, g["length"])


@pytest.mark.parametrize("strand", ["+", "-"])
def test_coding_genes_read_as_open_reading_frames(genes, seqs, strand):
    coding = [tag for tag, g in genes.items() if g["biotype"] == "protein_coding" and g["strand"] == strand]
    starts = sum(seqs[t][:3] in STARTS for t in coding) / len(coding)
    stops = sum(seqs[t][-3:] in STOPS for t in coding) / len(coding)
    assert starts >= 0.99, starts
    assert stops >= 0.99, stops


def test_strain_specific_deletions(by_name):
    # Deleted outright when the strain was built.
    assert "araB" not in by_name
    assert "rhaB" not in by_name
    # Disrupted rather than removed, so it survives as a pseudogene.
    assert by_name["lacZ"]["biotype"] == "pseudogene"


def test_known_genes(by_name, seqs):
    thrL = by_name["thrL"]
    assert (thrL["locus_tag"], thrL["b_number"], thrL["strand"]) == ("BW25113_0001", "b0001", "+")
    dnaA = by_name["dnaA"]
    assert dnaA["b_number"] == "b3702" and dnaA["strand"] == "-"
    # dnaA starts with GTG, a normal bacterial start codon, so check the set.
    assert seqs[dnaA["locus_tag"]][:3] in STARTS and seqs[dnaA["locus_tag"]][-3:] in STOPS
