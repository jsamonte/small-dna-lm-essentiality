# How small can a DNA language model be and still predict gene essentiality?

*Draft manuscript skeleton. Methods and Results are generated from `results/` by
`scripts/build_manuscript.py`; do not edit numbers here by hand. Introduction and
Discussion are outlines for the author.*

Independent researcher, Beaumont, TX

## Abstract

> Draft. Rewrite in your own voice once the Discussion is settled.

DNA language models are growing rapidly, but gene essentiality is a coarse property and it is unclear how much model is needed to predict it. We compared four DNA language models spanning 438,144 to 97,889,132 parameters against a classical sequence-feature baseline on 3,929 protein-coding genes of *Escherichia coli* BW25113, using identical cross-validation folds throughout. The best language model reached MCC 0.309 (nt-v2-100m, gradient boosting), against 0.331 for classical features. In paired bootstrap comparisons on identical genes, 7 of 8 model-classifier combinations were significantly worse than the baseline and 0 were better. Performance did not increase consistently with parameter count. Gene length, often suspected as a confound, carried almost no signal (MCC -0.010), although one model's embedding norms correlated strongly with it. We conclude that, across the four models tested, scaling to 10^8 parameters did not yield essentiality signal beyond cheap sequence statistics.

## 1. Introduction

*Outline to write:*

- Genome language models (Evo, Nucleotide Transformer, HyenaDNA) are scaling fast, and scaling is usually assumed to help.
- Gene essentiality is a coarse, well-labelled, biologically central property: a good probe for how much of that scaling reaches a practical task.
- The question nobody asks: not *does a frontier model work*, but *how small can you go before the signal disappears*. This matters for anyone without a datacenter GPU.
- State the contribution: a controlled scaling comparison with a strong classical baseline, identical folds, an explicit leakage control, and a cross-organism test.

## 2. Methods

### 2.1 Organism, genome and labels

Sequences are from *Escherichia coli* BW25113 (GenBank `CP009273.1`, 4,631,469 bp), which contains 4,490 annotated genes, 4,131 of them protein-coding. BW25113 rather than the more familiar MG1655: Goodall et al. assayed BW25113 and mapped their reads to this assembly, so their gene list is this annotation. Against MG1655, 15 assayed genes could not be located at all. For comparability with work keyed on MG1655 b-numbers, ECK identifiers present in both annotations provide a cross-reference for 4,467 genes (99.5%).

Essentiality calls are from Goodall et al. (2018), Table S1, a transposon-directed insertion-site sequencing (TraDIS) screen. The table classifies 4,313 genes as Essential (358), Non-essential (3793) or Unclear (162). Genes are identified by name rather than locus tag, so names were resolved against the annotation with the primary gene name taking precedence over historical synonyms; without that precedence, common names such as *acrB* appear ambiguous because they are also obsolete synonyms of unrelated genes. 4,260 of 4,313 rows resolved to exactly one gene with no collisions. The 53 unresolved rows are all insertion-sequence elements, which are present in many identical copies and were excluded. After restricting to protein-coding genes with a clear call, the analysis set is 3,929 genes, 353 essential and 3576 non-essential, an imbalance of 1:10.13.

### 2.2 Models and embeddings

| Model | Parameters | Embedding dim | Context | s/gene (CPU) |
|---|---|---|---|---|
| `LongSafari/hyenadna-tiny-1k-seqlen-hf` | 438,144 | 128 | ~1 kb | 0.0346 |
| `LongSafari/hyenadna-tiny-1k-seqlen-d256-hf` | 1,645,184 | 256 | ~1 kb | 0.0571 |
| `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species` | 55,904,972 | 512 | ~12 kb | 0.2304 |
| `InstaDeepAI/nucleotide-transformer-v2-100m-multi-species` | 97,889,132 | 512 | ~12 kb | 0.4313 |

Each gene is represented by the mean of the final hidden layer over all token positions. Model revisions are pinned to specific commits, since all four execute code from their own repositories. HyenaDNA's character-level tokenizer gives a context of about 1 kb, which 36.3% of protein-coding genes exceed; such genes are split into consecutive windows, embedded separately, and averaged weighted by token count. Nucleotide Transformer's 6-mer tokenizer reaches roughly 12 kb, which no gene exceeds, so no truncation occurs for those models. All inference ran on CPU; embeddings are cached and verified to be deterministic.

### 2.3 Classical baseline

The baseline uses 93 features: gene length and log length, GC content overall and at each codon position, the codon adaptation index against a reference set of ribosomal protein genes, the effective number of codons, and 1-, 2- and 3-mer frequencies.

### 2.4 Evaluation

A single stratified 5-fold split (seed 0) was fixed once and reused by every representation, enforced by a test; out-of-fold predictions are collected so that every gene is predicted by a model that never saw it. Both a logistic regression on standardised features and a gradient-boosted tree ensemble were trained, each with balanced class weights. Because the classes are imbalanced (91.0% majority), the Matthews correlation coefficient (MCC) is the primary metric and accuracy is reported only beside the majority-class rate. Confidence intervals are bootstrap percentiles over 2,000 resamples. Comparisons between a model and the baseline are *paired*: the difference in MCC is computed on the same resampled genes, rather than comparing two separate intervals.

## 3. Results

### 3.1 Classical features set a strong floor

Predicting the majority class for every gene gives 91.02% accuracy and MCC 0 by construction. The classical features reach MCC 0.331 (95% CI [+0.288, +0.372], AUROC 0.794) with gradient boosting and 0.269 with logistic regression.

### 3.2 No language model beat the baseline

| Representation | Classifier | MCC | 95% CI | AUROC |
|---|---|---|---|---|
| classical features | gradient boosting | **0.331** | [+0.288, +0.372] | 0.794 |
| hyenadna-tiny (438,144) | gradient boosting | 0.230 | [+0.184, +0.272] | 0.725 |
| hyenadna-tiny (438,144) | logistic regression | 0.235 | [+0.199, +0.269] | 0.752 |
| hyenadna-tiny-d256 (1,645,184) | gradient boosting | 0.247 | [+0.202, +0.290] | 0.717 |
| hyenadna-tiny-d256 (1,645,184) | logistic regression | 0.256 | [+0.220, +0.291] | 0.752 |
| nt-v2-50m (55,904,972) | gradient boosting | 0.271 | [+0.224, +0.319] | 0.745 |
| nt-v2-50m (55,904,972) | logistic regression | 0.236 | [+0.196, +0.276] | 0.744 |
| nt-v2-100m (97,889,132) | gradient boosting | 0.309 | [+0.261, +0.358] | 0.757 |
| nt-v2-100m (97,889,132) | logistic regression | 0.228 | [+0.189, +0.266] | 0.726 |

In paired bootstrap comparisons against the strongest classical configuration, 7 of 8 model-classifier combinations were significantly worse (95% interval entirely below zero) and 0 were significantly better. The best language model configuration, nt-v2-100m with gradient boosting, differed from the baseline by -0.022 MCC (95% CI [-0.067, +0.023]), that is, indistinguishable from it rather than better.

### 3.3 Scaling is weak and not consistent in direction

Across a 223-fold increase in parameters, MCC rose from 0.230 to 0.309 with gradient boosting but fell from 0.235 to 0.228 with logistic regression. The smallest model tested (438,144 parameters, 0.0346 s/gene on CPU) is within overlapping confidence intervals of the largest for the logistic-regression classifier. See `results/scaling_mcc_vs_params.png` and `scaling_mcc_vs_time.png`.

### 3.4 Leakage control: gene length

Gene length alone gives MCC -0.010, and removing all length features from the classical set changes performance negligibly (0.338 versus 0.331). Length is therefore not a viable explanation for any result here. Embedding norms nevertheless correlate with gene length to very different degrees:

| Model | Pearson r (embedding norm vs gene length) | r² |
|---|---|---|
| hyenadna-tiny | -0.102 | 0.011 |
| hyenadna-tiny-d256 | +0.058 | 0.003 |
| nt-v2-50m | +0.792 | 0.627 |
| nt-v2-100m | -0.304 | 0.092 |

This is a property of the representations worth reporting in its own right, but it cannot account for predictive performance, since length does not predict essentiality in this dataset.

### 3.5 Transfer to a second organism

Classifiers trained on all 3,929 *E. coli* genes were applied without retraining to 4,046 protein-coding genes of *Bacillus subtilis* 168 (255 essential, majority class 93.70%), with labels from SubtiWiki. Classical features were recomputed within *B. subtilis*, including an organism-specific codon adaptation reference, since carrying *E. coli* codon weights across would not be a transfer test.

| Representation | Classifier | E. coli MCC | B. subtilis MCC | Drop |
|---|---|---|---|---|
| classical | logistic regression | 0.269 | 0.174 [+0.141, +0.209] | 0.0946 |
| classical | gradient boosting | 0.3314 | 0.198 [+0.144, +0.251] | 0.1332 |
| hyenadna-tiny | logistic regression | 0.2347 | 0.092 [+0.055, +0.131] | 0.1429 |
| hyenadna-tiny | gradient boosting | 0.2298 | -0.019 [-0.040, +0.006] | 0.2491 |
| hyenadna-tiny-d256 | logistic regression | 0.2561 | 0.103 [+0.066, +0.141] | 0.1528 |
| hyenadna-tiny-d256 | gradient boosting | 0.2474 | 0.003 [-0.026, +0.036] | 0.244 |
| nt-v2-50m | logistic regression | 0.2356 | 0.141 [+0.098, +0.183] | 0.0941 |
| nt-v2-50m | gradient boosting | 0.2705 | 0.144 [+0.091, +0.197] | 0.1269 |
| nt-v2-100m | logistic regression | 0.228 | 0.119 [+0.079, +0.161] | 0.1088 |
| nt-v2-100m | gradient boosting | 0.3094 | 0.190 [+0.120, +0.256] | 0.1192 |

## 4. Discussion

*Outline to write:*

- The headline: across the four models tested, scaling did not buy essentiality signal beyond cheap sequence statistics. Say "across the four models tested", never "DNA language models scale as".
- Why cheap features are so strong here: essential genes are enriched for high expression, which codon usage and composition capture directly.
- What this means for practitioners without GPUs: the laptop-scale option loses little, because the expensive option also fails to beat free features.
- Where a language model might still win: fine-tuning rather than frozen embeddings, tasks needing regulatory context, or organisms with no codon-usage reference.
- Relation to prior work on essentiality prediction and to genome language model evaluations.

## 5. Limitations

1. **One organism for training, one for testing.** Two bacteria, both well studied, both with annotation pipelines that may themselves use composition. Nothing here speaks to eukaryotes.
2. **Frozen embeddings with a shallow classifier is not fine-tuning.** A fine-tuned model may extract more; this measures linearly and non-linearly accessible signal in the pretrained representation, which is the regime a laptop user is in.
3. **Gene length.** Length alone carries almost no signal here (MCC -0.010), but one model's embedding norms correlate strongly with it (r = +0.79); on a dataset where length did predict essentiality, that would be a confound.
4. **Context-window truncation.** HyenaDNA sees about 1 kb at a time and 36.3% of genes are longer, so its genes are read in windows and averaged. Nucleotide Transformer has no such limit here, so the comparison between families is partly a comparison of context handling.
5. **The labels are noisy and condition-dependent.** Essentiality is defined relative to a growth medium; Goodall et al. report a second condition (outgrowth in LB) whose calls differ for tens of genes, and a further 162 genes are explicitly Unclear and were excluded.
6. **Training-set contamination.** Nucleotide Transformer v2 is multi-species and very likely saw these genomes in pretraining, so this measures retrieval of learned sequence statistics, not prediction on novel sequence.
7. **Four models is not a scaling law.** The parameter axis has four points from two architectures with different tokenizers; architecture and scale are not separable here.

## 6. Data and code availability

All code is in this repository. `python run_all.py` reproduces every file in `results/` from a fresh clone. Sequence data is downloaded from NCBI (`CP009273.1`, `NC_000964.3`) and essentiality labels from the sources cited above; the Goodall supplementary tables are committed because their hosts block automated download.

## References

- Goodall ECA, Robinson A, Johnston IG, et al. (2018) The Essential Genome of *Escherichia coli* K-12. *mBio* 9:e02096-17.
- Baba T, Ara T, Hasegawa M, et al. (2006) Construction of *Escherichia coli* K-12 in-frame, single-gene knockout mutants: the Keio collection. *Mol Syst Biol* 2.
- Koo B-M, Kritikos G, Farelli JD, et al. (2017) Construction and Analysis of Two Genome-Scale Deletion Libraries for *Bacillus subtilis*. *Cell Systems* 4:291-305.
- Pedreira T, Elfmann C, Stülke J (2022) The current state of SubtiWiki, the database for the model organism *Bacillus subtilis*. *Nucleic Acids Research* 50:D875-D882.
- Nguyen E, Poli M, Faizi M, et al. (2023) HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution. *NeurIPS*.
- Dalla-Torre H, Gonzalez L, Mendoza-Revilla J, et al. (2024) Nucleotide Transformer: building and evaluating robust foundation models for human genomics. *Nature Methods*.
- Sharp PM, Li W-H (1987) The codon adaptation index. *Nucleic Acids Research* 15:1281-1295.
- Wright F (1990) The 'effective number of codons' used in a gene. *Gene* 87:23-29.

*Verify every citation against the publisher page before submission; these were written from memory and may contain errors in year, volume or author list.*
