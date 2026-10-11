"""Classical sequence features: the floor every language model has to clear.

Deliberately not a straw man. It includes the features a working
bioinformatician would reach for first:

  length, log length        essential genes differ systematically in length
  GC, GC1, GC2, GC3         base composition, overall and per codon position
  CAI                       codon adaptation index against ribosomal proteins,
                            a standard proxy for expression level
  Nc                        effective number of codons (Wright 1990), codon
                            bias measured without a reference set
  k-mer frequencies         1-, 2- and 3-mers, normalised

Gene length is kept because dropping it would understate the baseline, but it
correlates with essentiality and with embedding norms, so the comparison
scripts report length-only performance separately.
"""

import math
from collections import Counter
from itertools import product

BASES = "ACGT"
STOPS = {"TAA", "TAG", "TGA"}

# Standard genetic code, grouped by amino acid; used for CAI and Nc.
CODON_TABLE = {}
for _b1, _b2, _b3 in product("TCAG", repeat=3):
    _c = _b1 + _b2 + _b3
    CODON_TABLE[_c] = (
        "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
    )["TCAG".index(_b1) * 16 + "TCAG".index(_b2) * 4 + "TCAG".index(_b3)]

SYNONYMS = {}
for _codon, _aa in CODON_TABLE.items():
    SYNONYMS.setdefault(_aa, []).append(_codon)

KMERS = {k: ["".join(p) for p in product(BASES, repeat=k)] for k in (1, 2, 3)}


def codons(seq):
    return [seq[i:i + 3] for i in range(0, len(seq) - len(seq) % 3, 3)]


def gc_fraction(seq):
    counts = Counter(seq)
    total = sum(counts[b] for b in BASES)
    return (counts["G"] + counts["C"]) / total if total else 0.0


def gc_by_position(seq):
    out = []
    for offset in range(3):
        site = seq[offset::3]
        out.append(gc_fraction(site) if site else 0.0)
    return out


def relative_adaptiveness(reference_seqs):
    """Per-codon weights w from a reference set, the CAI of Sharp & Li (1987)."""
    counts = Counter()
    for seq in reference_seqs:
        counts.update(c for c in codons(seq) if c not in STOPS and set(c) <= set(BASES))
    weights = {}
    for aa, group in SYNONYMS.items():
        if aa == "*":
            continue
        group_counts = {c: counts.get(c, 0) for c in group}
        best = max(group_counts.values())
        for c in group:
            # +1 smoothing so an unseen codon does not make CAI collapse to zero.
            weights[c] = (group_counts[c] + 1) / (best + 1) if best else 1.0
    return weights


def cai(seq, weights):
    vals = [weights[c] for c in codons(seq)
            if c in weights and c not in STOPS and len(SYNONYMS[CODON_TABLE[c]]) > 1]
    if not vals:
        return 0.0
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def effective_number_of_codons(seq):
    """Nc (Wright 1990): 20 means maximum bias, 61 means none."""
    used = Counter(c for c in codons(seq) if c not in STOPS and set(c) <= set(BASES))
    by_class = {}
    for aa, group in SYNONYMS.items():
        if aa == "*" or len(group) == 1:
            continue
        n = sum(used.get(c, 0) for c in group)
        if n < 2:
            continue
        p2 = sum((used.get(c, 0) / n) ** 2 for c in group)
        f = (n * p2 - 1) / (n - 1)
        if f > 0:
            by_class.setdefault(len(group), []).append(f)
    if not by_class:
        return 61.0
    mean = {k: sum(v) / len(v) for k, v in by_class.items()}
    pooled = sum(mean.values()) / len(mean)
    # Met and Trp are invariant, then 9 two-fold, 1 three-fold, 5 four-fold and
    # 3 six-fold families. A family absent from this gene falls back to the
    # pooled mean, as Wright prescribes.
    total = 2.0
    for size, count in ((2, 9), (3, 1), (4, 5), (6, 3)):
        f = mean.get(size, pooled)
        total += count / f if f > 0 else 0
    return min(total, 61.0)


def kmer_frequencies(seq):
    out = []
    for k, alphabet in KMERS.items():
        counts = Counter(seq[i:i + k] for i in range(len(seq) - k + 1))
        total = sum(counts[m] for m in alphabet) or 1
        out.extend(counts.get(m, 0) / total for m in alphabet)
    return out


def feature_names():
    names = ["length", "log_length", "gc", "gc1", "gc2", "gc3", "cai", "nc", "n_codons"]
    for k, alphabet in KMERS.items():
        names.extend(f"kmer{k}_{m}" for m in alphabet)
    return names


def features_for(seq, weights):
    seq = seq.upper()
    gc1, gc2, gc3 = gc_by_position(seq)
    row = [
        float(len(seq)),
        math.log10(len(seq)) if seq else 0.0,
        gc_fraction(seq), gc1, gc2, gc3,
        cai(seq, weights),
        effective_number_of_codons(seq),
        float(len(seq) // 3),
    ]
    row.extend(kmer_frequencies(seq))
    return row
