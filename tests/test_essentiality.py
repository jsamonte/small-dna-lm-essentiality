"""Checks on the output of scripts/parse_essentiality.py.

The join is the step that can silently poison everything downstream: a label
attached to the wrong gene produces a plausible-looking but meaningless model.
These tests assert that every written row points at a real gene, that the
class counts match the published paper, and that the imbalance is what the
literature expects.
"""

import csv
import json
from collections import Counter
from pathlib import Path

import pytest

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"

if not (PROCESSED / "essentiality.csv").exists():
    pytest.skip("run scripts/parse_essentiality.py first", allow_module_level=True)


@pytest.fixture(scope="module")
def labels():
    with (PROCESSED / "essentiality.csv").open() as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def genes():
    with (PROCESSED / "genes.csv").open() as f:
        return {row["locus_tag"]: row for row in csv.DictReader(f)}


@pytest.fixture(scope="module")
def report():
    return json.loads((PROCESSED / "essentiality_join_report.json").read_text())


def st1(labels):
    return [r for r in labels if r["source"] == "Goodall2018_ST1"]


def test_every_label_points_at_a_real_gene(labels, genes):
    assert all(r["locus_tag"] in genes for r in labels)


def test_one_label_per_gene_per_source(labels):
    counts = Counter((r["source"], r["locus_tag"]) for r in labels)
    assert not [k for k, n in counts.items() if n > 1]


def test_class_counts_match_the_published_paper(report):
    # Goodall et al. 2018 Table S1 reports 358 essential genes.
    assert report["st1"]["classes_in_file"]["Essential"] == 358
    assert report["st1"]["written"]["Essential"] == 358


def test_unresolved_names_are_only_insertion_sequences(report):
    # IS elements are present in many identical copies, so a name cannot pick
    # one. They are excluded deliberately; anything else here is a real failure.
    assert all(n.startswith("ins") for n in report["st1"]["unresolved_names"])
    assert report["st1"]["unresolved_count"] <= 60


def test_no_collisions(report):
    assert report["st1"]["collisions"] == {}


def test_essential_flag_agrees_with_call(labels):
    for r in labels:
        assert r["essential"] == {"Essential": "True", "Non-essential": "False", "Unclear": ""}[r["call"]]


def test_class_imbalance_is_as_expected(report):
    cov = report["coverage_st1"]
    assert 300 <= cov["usable_essential"] <= 400, cov["usable_essential"]
    assert 8 <= cov["imbalance_ratio"] <= 15, cov["imbalance_ratio"]
    # The number every model has to beat; accuracy alone is meaningless here.
    assert 0.88 <= cov["majority_class_fraction"] <= 0.94


def test_known_essential_and_non_essential_genes(labels, genes):
    by_name = {r["gene_name"]: r for r in st1(labels)}
    # dnaA (replication initiator) and ftsZ (cell division) are textbook essential.
    assert by_name["dnaA"]["call"] == "Essential"
    assert by_name["ftsZ"]["call"] == "Essential"
    # lacZ is disrupted in BW25113 and is certainly not required for growth.
    assert by_name["lacZ"]["call"] == "Non-essential"
