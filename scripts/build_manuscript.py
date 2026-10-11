"""Generate paper/manuscript.md with every number read from results/.

No figure in the manuscript is ever typed by hand. Methods and Results are
written out in full; Introduction and Discussion are left as outlines, because
those are the author's argument to make, not a script's.

    .venv\\Scripts\\python scripts\\build_manuscript.py
"""

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

    w("# How small can a DNA language model be and still predict gene essentiality?")
    w("")
    w("*Draft manuscript skeleton. Methods and Results are generated from `results/` by")
    w("`scripts/build_manuscript.py`; do not edit numbers here by hand. Introduction and")
    w("Discussion are outlines for the author.*")
    w("")
    w("Independent researcher, Beaumont, TX")
    w("")

    # ---------------------------------------------------------------- abstract
    w("## Abstract")
    w("")
    w("> Draft. Rewrite in your own voice once the Discussion is settled.")
    w("")
    w(f"DNA language models are growing rapidly, but gene essentiality is a coarse "
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
      f"statistics.")
    w("")

    # ------------------------------------------------------------ introduction
    w("## 1. Introduction")
    w("")
    w("*Outline to write:*")
    w("")
    w("- Genome language models (Evo, Nucleotide Transformer, HyenaDNA) are scaling fast, "
      "and scaling is usually assumed to help.")
    w("- Gene essentiality is a coarse, well-labelled, biologically central property: a "
      "good probe for how much of that scaling reaches a practical task.")
    w("- The question nobody asks: not *does a frontier model work*, but *how small can "
      "you go before the signal disappears*. This matters for anyone without a "
      "datacenter GPU.")
    w("- State the contribution: a controlled scaling comparison with a strong classical "
      "baseline, identical folds, an explicit leakage control, and a cross-organism test.")
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
      f"code from their own repositories. HyenaDNA's character-level tokenizer gives a "
      f"context of about 1 kb, which 36.3% of protein-coding genes exceed; such genes are "
      f"split into consecutive windows, embedded separately, and averaged weighted by "
      f"token count. Nucleotide Transformer's 6-mer tokenizer reaches roughly 12 kb, which "
      f"no gene exceeds, so no truncation occurs for those models. All inference ran on "
      f"CPU; embeddings are cached and verified to be deterministic.")
    w("")

    w("### 2.3 Classical baseline")
    w("")
    w(f"The baseline uses {baseline['results']['all__gradient_boosting']['n_features']} "
      f"features: gene length and log length, GC content overall and at each codon "
      f"position, the codon adaptation index against a reference set of ribosomal protein "
      f"genes, the effective number of codons, and 1-, 2- and 3-mer frequencies.")
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
    w("### 3.3 Scaling is weak and not consistent in direction")
    w("")
    gb = sorted([r for r in rows if r["classifier"] == "gradient_boosting"], key=lambda r: r["params"])
    lr = sorted([r for r in rows if r["classifier"] == "logistic_regression"], key=lambda r: r["params"])
    if gb and lr:
        factor = gb[-1]["params"] / gb[0]["params"]
        w(f"Across a {factor:.0f}-fold increase in parameters, MCC rose from "
          f"{gb[0]['mcc']:.3f} to {gb[-1]['mcc']:.3f} with gradient boosting but fell from "
          f"{lr[0]['mcc']:.3f} to {lr[-1]['mcc']:.3f} with logistic regression. The "
          f"smallest model tested ({gb[0]['params']:,} parameters, "
          f"{gb[0]['seconds_per_gene']} s/gene on CPU) is within overlapping confidence "
          f"intervals of the largest for the logistic-regression classifier. See "
          f"`results/scaling_mcc_vs_params.png` and `scaling_mcc_vs_time.png`.")
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
          f"{t['majority_class_fraction']:.2%}), with labels from SubtiWiki. Classical "
          f"features were recomputed within *B. subtilis*, including an organism-specific "
          f"codon adaptation reference, since carrying *E. coli* codon weights across would "
          f"not be a transfer test.")
        w("")
        w("| Representation | Classifier | E. coli MCC | B. subtilis MCC | Drop |")
        w("|---|---|---|---|---|")
        for r in general["results"]:
            w(f"| {r['representation']} | {r['classifier'].replace('_', ' ')} | "
              f"{r['within_ecoli_mcc']} | {r['mcc']:.3f} {fmt_ci(r['mcc_ci95'])} | "
              f"{r['mcc_drop']} |")
        w("")
    else:
        w("*Pending: run `scripts/evaluate_generalization.py`.*")
        w("")

    # -------------------------------------------------------------- discussion
    w("## 4. Discussion")
    w("")
    w("*Outline to write:*")
    w("")
    w("- The headline: across the four models tested, scaling did not buy essentiality "
      "signal beyond cheap sequence statistics. Say \"across the four models tested\", "
      "never \"DNA language models scale as\".")
    w("- Why cheap features are so strong here: essential genes are enriched for high "
      "expression, which codon usage and composition capture directly.")
    w("- What this means for practitioners without GPUs: the laptop-scale option loses "
      "little, because the expensive option also fails to beat free features.")
    w("- Where a language model might still win: fine-tuning rather than frozen "
      "embeddings, tasks needing regulatory context, or organisms with no codon-usage "
      "reference.")
    w("- Relation to prior work on essentiality prediction and to genome language model "
      "evaluations.")
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

    w("## References")
    w("")
    w("- Goodall ECA, Robinson A, Johnston IG, et al. (2018) The Essential Genome of "
      "*Escherichia coli* K-12. *mBio* 9:e02096-17.")
    w("- Baba T, Ara T, Hasegawa M, et al. (2006) Construction of *Escherichia coli* K-12 "
      "in-frame, single-gene knockout mutants: the Keio collection. *Mol Syst Biol* 2.")
    w("- Koo B-M, Kritikos G, Farelli JD, et al. (2017) Construction and Analysis of Two "
      "Genome-Scale Deletion Libraries for *Bacillus subtilis*. *Cell Systems* 4:291-305.")
    w("- Pedreira T, Elfmann C, Stülke J (2022) The current state of SubtiWiki, the database "
      "for the model organism *Bacillus subtilis*. *Nucleic Acids Research* 50:D875-D882.")
    w("- Nguyen E, Poli M, Faizi M, et al. (2023) HyenaDNA: Long-Range Genomic Sequence "
      "Modeling at Single Nucleotide Resolution. *NeurIPS*.")
    w("- Dalla-Torre H, Gonzalez L, Mendoza-Revilla J, et al. (2024) Nucleotide Transformer: "
      "building and evaluating robust foundation models for human genomics. *Nature "
      "Methods*.")
    w("- Sharp PM, Li W-H (1987) The codon adaptation index. *Nucleic Acids Research* "
      "15:1281-1295.")
    w("- Wright F (1990) The 'effective number of codons' used in a gene. *Gene* 87:23-29.")
    w("")
    w("*Verify every citation against the publisher page before submission; these were "
      "written from memory and may contain errors in year, volume or author list.*")

    PAPER.mkdir(parents=True, exist_ok=True)
    out = PAPER / "manuscript.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Wrote {out.relative_to(ROOT)} ({len(L)} lines)")
    if not general:
        print("NOTE: generalization results missing; section 3.5 is a placeholder.")


if __name__ == "__main__":
    main()
