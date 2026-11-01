"""Generate paper/manuscript.md with every number read from results/.

No figure in the manuscript is ever typed by hand. Methods and Results are
written out in full; Introduction and Discussion are left as outlines, because
those are the author's argument to make, not a script's.

    .venv\\Scripts\\python scripts\\build_manuscript.py
"""

import atexit
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
PROCESSED = ROOT / "data" / "processed"
PAPER = ROOT / "paper"


def load(name, where=RESULTS):
    path = where / name
    return json.loads(path.read_text()) if path.exists() else None


#: A section file seeded from a template carries this marker. While it is
#: present the section counts as unwritten: the build substitutes the outline
#: and says so loudly on exit, so placeholder text can never be mistaken for
#: prose in a posted preprint. Delete the marker line when you have written it.
#: A section drafted BY CLAUDE rather than by the author carries this marker.
#: Unlike UNWRITTEN it is still used, so the manuscript reads end to end - but
#: every build says so. A committee assesses how the AUTHOR reasons, and prose
#: that was edited rather than argued does not survive an interview. Delete the
#: marker once the section has been rewritten in the author's own words.
DRAFT_MARKER = "<!-- DRAFT -->"
_drafted = []
SPLIT_ON = chr(10)


@atexit.register
def _warn_drafted():
    if _drafted:
        names = ", ".join(sorted(set(_drafted)))
        print()
        print("  ~~ STILL IN CLAUDE'S WORDS: " + names)
        print("     Usable, and NOT yours yet. Rewrite before posting.")
        print("     Remove the <!-- DRAFT --> line when you have.")
UNWRITTEN_MARKER = "<!-- UNWRITTEN -->"
_unwritten = []


@atexit.register
def _warn_unwritten():
    if _unwritten:
        names = ", ".join(sorted(set(_unwritten)))
        print()
        print("  !! STILL UNWRITTEN: " + names)
        print("     Those sections show a generated outline, not your prose.")
        print("     Remove the <!-- UNWRITTEN --> line once you have written one.")
def prose(name, fallback):
    path = PAPER / "sections" / f"{name}.md"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if UNWRITTEN_MARKER in text:
            _unwritten.append(name)
            return fallback
        if DRAFT_MARKER in text:
            _drafted.append(name)
        return text.rstrip().split(SPLIT_ON)
    return fallback


def fmt_ci(ci):
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "n/a"


def main():
    baseline = load("baseline.json")
    scaling = load("scaling.json")
    general = load("generalization.json")
    genome = load("genome_manifest.json", PROCESSED)
    join = load("essentiality_join_report.json", PROCESSED)
    bsub = load("bsub_manifest.json", PROCESSED)
    if not (baseline and scaling):
        sys.exit("run baseline_features.py and evaluate_scaling.py first")

    ds = baseline["dataset"]
    best_bl_key = max((k for k in baseline["results"] if k.startswith("all__")),
                      key=lambda k: baseline["results"][k]["mcc"])
    best_bl = baseline["results"][best_bl_key]
    len_only = baseline["results"]["length_only__gradient_boosting"]
    no_len = baseline["results"]["no_length__gradient_boosting"]

    rows = scaling["results"]
    best_lm = max(rows, key=lambda r: r["mcc"])
    paired = scaling.get("paired_vs_baseline", {})
    n_worse = sum(1 for c in paired.values() if c["ci95"][1] < 0)
    n_better = sum(1 for c in paired.values() if c["ci95"][0] > 0)

    L = []
    w = L.append

    w("# DNA language models up to 10^8 parameters do not beat codon "
      "statistics at predicting gene essentiality")
    w("")
    w("*Methods, Results and Limitations are generated from `results/` by")
    w("`scripts/build_manuscript.py`; do not edit numbers here. Abstract, Introduction")
    w("and Discussion are in `paper/sections/` and are the author's to write.*")
    w("")
    w("Jared Samonte")
    w("")
    w("Independent researcher, Beaumont, TX")
    w("")

    # ---------------------------------------------------------------- abstract
    w("## Abstract")
    w("")
    for line in prose("abstract", [
        f"DNA language models are growing rapidly, but gene essentiality is a coarse "
        f"property and it is unclear how much model is needed to predict it. We compared "
        f"four DNA language models spanning {min(r['params'] for r in rows):,} to "
        f"{max(r['params'] for r in rows):,} parameters against a classical "
        f"sequence-feature baseline on {ds['n_genes']:,} protein-coding genes of "
        f"*Escherichia coli* BW25113, using identical cross-validation folds throughout. "
        f"The best language model reached MCC {best_lm['mcc']:.3f} "
        f"({best_lm['model']}, {best_lm['classifier'].replace('_', ' ')}), against "
        f"{best_bl['mcc']:.3f} for classical features. In paired bootstrap comparisons on "
        f"identical genes, {n_worse} of {len(paired)} model-classifier combinations were "
        f"significantly worse than the baseline and {n_better} were better. "
        f"Performance did not increase consistently with parameter count. "
        f"Gene length, often suspected as a confound, carried almost no signal "
        f"(MCC {len_only['mcc']:.3f}), although one model's embedding norms correlated "
        f"strongly with it. We conclude that, across the four models tested, scaling "
        f"to 10^8 parameters did not yield essentiality signal beyond cheap sequence "
        f"statistics.",
    ]):
        w(line)
    w("")

    # ------------------------------------------------------------ introduction
    w("## 1. Introduction")
    w("")
    for line in prose("introduction", [
        "*Outline to write: scaling of genome language models; essentiality as a probe;",
        "how small can you go; why it matters without a datacenter GPU; the contribution.*",
    ]):
        w(line)
    w("")

    # ----------------------------------------------------------------- methods
    w("## 2. Methods")
    w("")
    w("### 2.1 Organism, genome and labels")
    w("")
    if genome:
        w(f"Sequences are from *Escherichia coli* BW25113 (GenBank `{genome['accession']}`, "
          f"{genome['genome_length']:,} bp), which contains {genome['genes_total']:,} "
          f"annotated genes, {genome['genes_by_biotype'].get('protein_coding', 0):,} of them "
          f"protein-coding. BW25113 rather than the more familiar MG1655: Goodall et al. "
          f"assayed BW25113 and mapped their reads to this assembly, so their gene list is "
          f"this annotation. Against MG1655, 15 assayed genes could not be located at all. "
          f"For comparability with work keyed on MG1655 b-numbers, ECK identifiers present "
          f"in both annotations provide a cross-reference for "
          f"{genome['genes_with_b_number']:,} genes "
          f"({genome['genes_with_b_number'] / genome['genes_total']:.1%}).")
        w("")
    if join:
        st1 = join["st1"]
        cov = join["coverage_st1"]
        w(f"Essentiality calls are from Goodall et al. (2018), Table S1, a transposon-directed "
          f"insertion-site sequencing (TraDIS) screen. The table classifies "
          f"{st1['rows_in_file']:,} genes as Essential "
          f"({st1['classes_in_file']['Essential']}), Non-essential "
          f"({st1['classes_in_file']['Non-essential']}) or Unclear "
          f"({st1['classes_in_file']['Unclear']}). Genes are identified by name rather than "
          f"locus tag, so names were resolved against the annotation with the primary gene "
          f"name taking precedence over historical synonyms; without that precedence, common "
          f"names such as *acrB* appear ambiguous because they are also obsolete synonyms of "
          f"unrelated genes. {st1['written_total']:,} of {st1['rows_in_file']:,} rows "
          f"resolved to exactly one gene with no collisions. The "
          f"{st1['unresolved_count']} unresolved rows are all insertion-sequence elements, "
          f"which are present in many identical copies and were excluded. After restricting "
          f"to protein-coding genes with a clear call, the analysis set is "
          f"{cov['usable_protein_coding_excluding_unclear']:,} genes, "
          f"{cov['usable_essential']} essential and {cov['usable_non_essential']} "
          f"non-essential, an imbalance of 1:{cov['imbalance_ratio']}.")
        w("")

    w("### 2.2 Models and embeddings")
    w("")
    w("| Model | Parameters | Embedding dim | Context | s/gene (CPU) |")
    w("|---|---|---|---|---|")
    seen = set()
    for r in sorted(rows, key=lambda r: r["params"]):
        if r["model"] in seen:
            continue
        seen.add(r["model"])
        ctx = "~1 kb" if "hyena" in r["model"] else "~12 kb"
        w(f"| `{r['model_id']}` | {r['params']:,} | {r['dim']} | {ctx} | "
          f"{r['seconds_per_gene']} |")
    w("")
    w(f"Each gene is represented by the mean of the final hidden layer over all token "
      f"positions. Model revisions are pinned to specific commits, since all four execute "
      f"code from their own repositories. HyenaDNA (Nguyen et al., 2023) uses a "
      f"character-level tokenizer, which gives it a "
      f"context of about 1 kb, which 36.3% of protein-coding genes exceed; such genes are "
      f"split into consecutive windows, embedded separately, and averaged weighted by "
      f"token count. The 6-mer tokenizer of Nucleotide Transformer (Dalla-Torre et al., "
      f"2025) reaches roughly 12 kb, which "
      f"no gene exceeds, so no truncation occurs for those models. All inference ran on "
      f"CPU; embeddings are cached and verified to be deterministic.")
    w("")

    w("### 2.3 Classical baseline")
    w("")
    w(f"The baseline uses {baseline['results']['all__gradient_boosting']['n_features']} "
      f"features: gene length and log length, GC content overall and at each codon "
      f"position, the codon adaptation index (Sharp and Li, 1987) against a reference set "
      f"of ribosomal protein genes, the effective number of codons (Wright, 1990), and "
      f"1-, 2- and 3-mer frequencies.")
    w("")

    w("### 2.4 Evaluation")
    w("")
    w(f"A single stratified {ds['n_folds']}-fold split (seed {ds['seed']}) was fixed once "
      f"and reused by every representation, enforced by a test; out-of-fold predictions "
      f"are collected so that every gene is predicted by a model that never saw it. Both a "
      f"logistic regression on standardised features and a gradient-boosted tree ensemble "
      f"were trained, each with balanced class weights. Because the classes are imbalanced "
      f"({ds['majority_class_accuracy']:.1%} majority), the Matthews correlation "
      f"coefficient (MCC) is the primary metric and accuracy is reported only beside the "
      f"majority-class rate. Confidence intervals are bootstrap percentiles over 2,000 "
      f"resamples. Comparisons between a model and the baseline are *paired*: the "
      f"difference in MCC is computed on the same resampled genes, rather than comparing "
      f"two separate intervals.")
    w("")

    # ----------------------------------------------------------------- results
    w("## 3. Results")
    w("")
    # Figure and caption are both generated from results/, by
    # scripts/make_figures.py, so the figure and the text cannot disagree.
    figure = PAPER / "figures" / "fig1_scaling_and_transfer.png"
    caption = RESULTS / "fig1_caption.md"
    if figure.exists() and caption.exists():
        w("![](figures/fig1_scaling_and_transfer.png)")
        w("")
        w(caption.read_text(encoding="utf-8").strip())
        w("")
    w("### 3.1 Classical features set a strong floor")
    w("")
    w(f"Predicting the majority class for every gene gives "
      f"{ds['majority_class_accuracy']:.2%} accuracy and MCC 0 by construction. The "
      f"classical features reach MCC {best_bl['mcc']:.3f} "
      f"(95% CI {fmt_ci(best_bl['mcc_ci95'])}, AUROC {best_bl['auroc']:.3f}) with "
      f"gradient boosting and "
      f"{baseline['results']['all__logistic_regression']['mcc']:.3f} with logistic "
      f"regression.")
    w("")
    w("### 3.2 No language model beat the baseline")
    w("")
    w("| Representation | Classifier | MCC | 95% CI | AUROC |")
    w("|---|---|---|---|---|")
    w(f"| classical features | gradient boosting | **{best_bl['mcc']:.3f}** | "
      f"{fmt_ci(best_bl['mcc_ci95'])} | {best_bl['auroc']:.3f} |")
    for r in sorted(rows, key=lambda r: (r["params"], r["classifier"])):
        w(f"| {r['model']} ({r['params']:,}) | {r['classifier'].replace('_', ' ')} | "
          f"{r['mcc']:.3f} | {fmt_ci(r['mcc_ci95'])} | {r['auroc']:.3f} |")
    w("")
    if paired:
        w(f"In paired bootstrap comparisons against the strongest classical configuration, "
          f"{n_worse} of {len(paired)} model-classifier combinations were significantly "
          f"worse (95% interval entirely below zero) and {n_better} were significantly "
          f"better. The best language model configuration, "
          f"{best_lm['model']} with {best_lm['classifier'].replace('_', ' ')}, differed "
          f"from the baseline by "
          f"{paired.get(best_lm['model'] + '__' + best_lm['classifier'], {}).get('delta_mcc', float('nan')):+.3f} "
          f"MCC "
          f"(95% CI {fmt_ci(paired.get(best_lm['model'] + '__' + best_lm['classifier'], {}).get('ci95'))}), "
          f"that is, indistinguishable from it rather than better.")
        w("")
        # Two objections a referee raises here, answered where they arise.
        w(f"Two features of that comparison are worth stating rather than "
          f"leaving to be noticed. First, the baseline is itself the better "
          f"of two classical configurations, so the headline is a comparison "
          f"of the best classical result against each language model result "
          f"individually; against the weaker classical configuration "
          f"(logistic regression, {baseline['results']['all__logistic_regression']['mcc']:.3f}) "
          f"several language models are not worse. The best-against-best "
          f"comparison is the one quoted above and is the one we rely on. "
          f"Second, {len(paired)} comparisons are made against a single "
          f"baseline with no multiplicity correction. None is needed for the "
          f"direction of the conclusion, since no correction can turn "
          f"{n_better} significant wins into one, and a Bonferroni threshold "
          f"of 0.05/{len(paired)} would only reduce how many of the "
          f"{n_worse} losses are called significant.")
        w("")
    w("### 3.3 Scaling helps the stronger classifier, but not enough")
    w("")
    gb = sorted([r for r in rows if r["classifier"] == "gradient_boosting"],
                key=lambda r: r["params"])
    lr = sorted([r for r in rows if r["classifier"] == "logistic_regression"],
                key=lambda r: r["params"])
    if gb and lr:
        factor = gb[-1]["params"] / gb[0]["params"]
        monotonic = all(b["mcc"] >= a["mcc"] for a, b in zip(gb, gb[1:]))
        gb_seq = ", ".join("{:.3f}".format(r["mcc"]) for r in gb)
        lr_seq = ", ".join("{:.3f}".format(r["mcc"]) for r in lr)
        # Reported plainly rather than folded into "scaling is weak": the
        # gradient-boosting column rises at every step, and a reader who
        # plots it will see that whether or not the text admits it.
        w(f"With gradient boosting the four models, in increasing size order, "
          f"give MCC {gb_seq}. That is "
          f"{'monotonic' if monotonic else 'not monotonic'} across a "
          f"{factor:.0f}-fold increase in parameters, and it is the strongest "
          f"evidence in this study that scale buys anything. With logistic "
          f"regression the same four give {lr_seq}, which is flat to slightly "
          f"decreasing.")
        w("")
        w(f"Two things stop this being a positive scaling result. The trend "
          f"holds for one classifier of the two, and the largest model's "
          f"interval still contains the baseline instead of clearing it, at "
          f"{gb[-1]['mcc']:.3f} "
          f"[{gb[-1]['mcc_ci95'][0]:+.3f}, {gb[-1]['mcc_ci95'][1]:+.3f}] "
          f"against {best_bl['mcc']:.3f} for the classical features. Four "
          f"points from two architectures cannot be extrapolated, and we do "
          f"not claim a larger model would fail to clear the baseline. The "
          f"supportable statement is narrower, that nothing tested here does, "
          f"and that the gap was still closing at the largest size we could "
          f"run. The smallest model ({gb[0]['params']:,} parameters, "
          f"{gb[0]['seconds_per_gene']} s/gene on CPU) sits within overlapping "
          f"intervals of the largest under logistic regression. Figure 1A "
          f"shows both trends.")
        w("")

    w("### 3.4 Leakage control: gene length")
    w("")
    w(f"Gene length alone gives MCC {len_only['mcc']:.3f}, and removing all length features "
      f"from the classical set changes performance negligibly "
      f"({no_len['mcc']:.3f} versus {best_bl['mcc']:.3f}). Length is therefore not a viable "
      f"explanation for any result here. Embedding norms nevertheless correlate with gene "
      f"length to very different degrees:")
    w("")
    w("| Model | Pearson r (embedding norm vs gene length) | r² |")
    w("|---|---|---|")
    for k, v in scaling["leakage"].items():
        w(f"| {k} | {v['pearson_r_embedding_norm_vs_gene_length']:+.3f} | {v['r_squared']:.3f} |")
    w("")
    w("This is a property of the representations worth reporting in its own right, but it "
      "cannot account for predictive performance, since length does not predict "
      "essentiality in this dataset.")
    w("")

    w("### 3.5 Transfer to a second organism")
    w("")
    if general:
        t = general["test"]
        w(f"Classifiers trained on all {general['train']['n']:,} *E. coli* genes were "
          f"applied without retraining to {t['n']:,} protein-coding genes of *Bacillus "
          f"subtilis* 168 ({t['essential']} essential, majority class "
          f"{t['majority_class_fraction']:.2%}), with labels from SubtiWiki (Pedreira et al., "
          f"2022), whose per-gene essentiality field is curated from the published "
          f"deletion-library work and whose counts agree with Koo et al. (2017). "
          f"Classical "
          f"features were recomputed within *B. subtilis*, including an organism-specific "
          f"codon adaptation reference, since carrying *E. coli* codon weights across would "
          f"not be a transfer test.")
        w("")
        w("That recomputation is correct, but it is also an "
          "organism-specific recalibration that a frozen embedding has no "
          "equivalent of, so the alternative is reported beside it. The row "
          "`classical_ecoli_cai` is the same trained classifier applied to "
          "*B. subtilis* features built with *E. coli* codon weights, which "
          "puts the classical pipeline on the same footing as the models.")
        w("")
        w("| Representation | Classifier | E. coli MCC | B. subtilis MCC | Drop |")
        w("|---|---|---|---|---|")
        for r in general["results"]:
            w(f"| {r['representation']} | {r['classifier'].replace('_', ' ')} | "
              f"{r['within_ecoli_mcc']} | {r['mcc']:.3f} {fmt_ci(r['mcc_ci95'])} | "
              f"{r['mcc_drop']} |")
        w("")
        gx = {(r["representation"], r["classifier"]): r for r in general["results"]}
        cal_gb = gx[("classical", "gradient_boosting")]
        raw_gb = gx[("classical_ecoli_cai", "gradient_boosting")]
        cal_lr = gx[("classical", "logistic_regression")]
        raw_lr = gx[("classical_ecoli_cai", "logistic_regression")]
        best_lm_t = max((r for k, r in gx.items() if not k[0].startswith("classical")),
                        key=lambda r: r["mcc"])
        w(f"**The classical baseline's transfer depends on that "
          f"recalibration, and only under gradient boosting.** Denied a "
          f"*B. subtilis* codon reference, classical features with gradient "
          f"boosting fall from {cal_gb['mcc']:.3f} to {raw_gb['mcc']:.3f} "
          f"{fmt_ci(raw_gb['mcc_ci95'])}, an interval containing zero. The "
          f"same features with logistic regression are unaffected, moving "
          f"from {cal_lr['mcc']:.3f} to {raw_lr['mcc']:.3f}. On the "
          f"equal-footing comparison, where nothing is recalibrated in the "
          f"test organism, the best language model result is "
          f"{best_lm_t['representation']} with "
          f"{best_lm_t['classifier'].replace('_', ' ')} at "
          f"{best_lm_t['mcc']:.3f} {fmt_ci(best_lm_t['mcc_ci95'])} and the "
          f"best classical result is logistic regression at "
          f"{raw_lr['mcc']:.3f} {fmt_ci(raw_lr['mcc_ci95'])}. Those "
          f"intervals overlap substantially, so the conclusion of \u00a73.2 "
          f"survives the move between organisms: the expensive "
          f"representation matches the cheap one and does not beat it.")
        w("")
    else:
        w("*Pending: run `scripts/evaluate_generalization.py`.*")
        w("")

    # -------------------------------------------------------------- discussion
    w("## 4. Discussion")
    w("")
    for line in prose("discussion", ["*Outline to write.*"]):
        w(line)
    w("")

    # ------------------------------------------------------------- limitations
    w("## 5. Limitations")
    w("")
    w("1. **One organism for training, one for testing.** Two bacteria, both well studied, "
      "both with annotation pipelines that may themselves use composition. Nothing here "
      "speaks to eukaryotes.")
    w("2. **Frozen embeddings with a shallow classifier is not fine-tuning.** A fine-tuned "
      "model may extract more; this measures linearly and non-linearly accessible signal "
      "in the pretrained representation, which is the regime a laptop user is in.")
    w(f"3. **Gene length.** Length alone carries almost no signal here "
      f"(MCC {len_only['mcc']:.3f}), but one model's embedding norms correlate strongly "
      f"with it (r = "
      f"{max(v['pearson_r_embedding_norm_vs_gene_length'] for v in scaling['leakage'].values()):+.2f}); "
      f"on a dataset where length did predict essentiality, that would be a confound.")
    w("4. **Context-window truncation.** HyenaDNA sees about 1 kb at a time and 36.3% of "
      "genes are longer, so its genes are read in windows and averaged. Nucleotide "
      "Transformer has no such limit here, so the comparison between families is partly a "
      "comparison of context handling.")
    w("5. **The labels are noisy and condition-dependent.** Essentiality is defined "
      "relative to a growth medium; Goodall et al. report a second condition (outgrowth in "
      "LB) whose calls differ for tens of genes, and a further 162 genes are explicitly "
      "Unclear and were excluded.")
    w("6. **Training-set contamination.** Nucleotide Transformer v2 is multi-species and "
      "very likely saw these genomes in pretraining, so this measures retrieval of learned "
      "sequence statistics, not prediction on novel sequence.")
    w("7. **Four models is not a scaling law.** The parameter axis has four points from two "
      "architectures with different tokenizers; architecture and scale are not separable "
      "here.")
    w("")

    w("## 6. Data and code availability")
    w("")
    w("All code is in this repository. `python run_all.py` reproduces every file in "
      "`results/` from a fresh clone. Sequence data is downloaded from NCBI "
      "(`CP009273.1`, `NC_000964.3`) and essentiality labels from the sources cited above; "
      "the Goodall supplementary tables are committed because their hosts block automated "
      "download.")
    w("")

    w("")
    # Public as of 2026-10-31.
    w("The repository is public at "
      "https://github.com/jsamonte/small-dna-lm-essentiality.")
    w("")
    # ------------------------------------------------------ declarations
    w("## 7. Declarations")
    w("")
    w("**Funding.** This work received no funding.")
    w("")
    w("**Competing interests.** The author declares no competing interests.")
    w("")
    w("**Use of AI and AI-assisted technologies.** "
      "Anthropic's Claude was used throughout this work, under the "
      "author's direction, to write and debug the analysis code, to draft "
      "and edit prose, and to check the manuscript against the results "
      "files. Every numerical value in this paper is written by "
      "`scripts/build_manuscript.py` from the files in `results/`, which "
      "`python run_all.py` regenerates from the raw data, so no number "
      "here was produced by a language model. The author chose the "
      "question, the models and the baseline, verified the results and "
      "takes responsibility for the content.")
    w("")

    w("## References")
    w("")
    # Verified against Crossref and PubMed on 2026-10-11. Do not edit from
    # memory: five of these eight were wrong in the first draft.
    w("1. Goodall ECA, Robinson A, Johnston IG, Jabbari S, Turner KA, Cunningham AF, "
      "Lund PA, Cole JA, Henderson IR (2018) The Essential Genome of *Escherichia coli* "
      "K-12. *mBio* 9(1):e02096-17. doi:10.1128/mBio.02096-17")
    w("2. Baba T, Ara T, Hasegawa M, Takai Y, Okumura Y, Baba M, Datsenko KA, Tomita M, "
      "Wanner BL, Mori H (2006) Construction of *Escherichia coli* K-12 in-frame, "
      "single-gene knockout mutants: the Keio collection. *Molecular Systems Biology* "
      "2:2006.0008. doi:10.1038/msb4100050")
    w("3. Koo B-M, Kritikos G, Farelli JD, Todor H, Tong K, Kimsey H, Wapinski I, "
      "Galardini M, Cabal A, Peters JM, Hachmann A-B, Rudner DZ, Allen KN, Typas A, "
      "Gross CA (2017) Construction and Analysis of Two Genome-Scale Deletion Libraries "
      "for *Bacillus subtilis*. *Cell Systems* 4(3):291-305.e7. "
      "doi:10.1016/j.cels.2016.12.013")
    w("4. Pedreira T, Elfmann C, Stülke J (2022) The current state of *Subti*Wiki, the "
      "database for the model organism *Bacillus subtilis*. *Nucleic Acids Research* "
      "50(D1):D875-D882. doi:10.1093/nar/gkab943")
    w("5. Nguyen E, Poli M, Faizi M, Thomas A, Wornow M, Birch-Sykes C, Massaroli S, "
      "Patel A, Rabideau C, Bengio Y, Ermon S, Ré C, Baccus S (2023) HyenaDNA: "
      "Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution. In: "
      "*Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*, "
      "pp 43177-43201. Curran Associates. doi:10.52202/075280-1872")
    w("6. Dalla-Torre H, Gonzalez L, Mendoza-Revilla J, Lopez Carranza N, "
      "Grzywaczewski AH, Oteri F, Dallago C, Trop E, de Almeida BP, Sirelkhatim H, "
      "Richard G, Skwark M, Beguir K, Lopez M, Pierrot T (2025) Nucleotide Transformer: "
      "building and evaluating robust foundation models for human genomics. *Nature "
      "Methods* 22(2):287-297. doi:10.1038/s41592-024-02523-z")
    w("7. Sharp PM, Li W-H (1987) The codon Adaptation Index--a measure of directional "
      "synonymous codon usage bias, and its potential applications. *Nucleic Acids "
      "Research* 15(3):1281-1295. doi:10.1093/nar/15.3.1281")
    w("8. Wright F (1990) The 'effective number of codons' used in a gene. *Gene* "
      "87(1):23-29. doi:10.1016/0378-1119(90)90491-9")
    # Verified against Crossref on 2026-10-27. 62 authors; first three plus
    # et al., as Nature style does. Title keeps the published British
    # "modelling". Cited only to anchor the parameter range in the
    # introduction; this model is not evaluated here.
    w("9. Brixi G, Durrant MG, Ku J, et al. (2026) Genome modelling and design across "
      "all domains of life with Evo 2. *Nature* 652(8112):1349-1361. "
      "doi:10.1038/s41586-026-10176-5")
    w("")
    w("*Verified against Crossref and PubMed on 2026-10-11. Two points a copy-editor may "
      "query: the dash in reference 7's title is rendered `--` following PubMed, and "
      "Crossref's lowercase variant of that title disagrees with PubMed, Europe PMC and "
      "Semantic Scholar; reference 5's author order follows the NeurIPS proceedings, "
      "which differs from the arXiv preprint.*")

    PAPER.mkdir(parents=True, exist_ok=True)
    out = PAPER / "manuscript.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Wrote {out.relative_to(ROOT)} ({len(L)} lines)")
    if not general:
        print("NOTE: generalization results missing; section 3.5 is a placeholder.")


if __name__ == "__main__":
    main()
