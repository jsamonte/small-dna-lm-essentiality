# small-dna-lm-essentiality

## What this is
A scaling comparison: how well do very small DNA language models (451k to 100M
parameters) predict bacterial gene essentiality, versus classical sequence
features, and versus each other?

## Scientific question
What is the smallest DNA language model that carries useful gene-essentiality
signal, and what is the accuracy cost of staying laptop-scale?

## Why it matters
Frontier genome models need datacenter GPUs. If small models retain most of the
signal, genomics on modest hardware stays viable. This is the on-premises /
edge-inference argument, quantified.

## Organism
E. coli K-12 **BW25113** (CP009273.1), not MG1655. Changed 2026-10-11: Goodall
et al. assayed BW25113 and mapped their reads to CP009273.1, so their gene list
IS this annotation. Against MG1655, 15 measured genes could not be located at
all; against BW25113, every row resolves. MG1655 (NC_000913.3) is still
downloaded, purely so ECK numbers can carry b-numbers across for comparability
(99.5% of genes). syn3A is a later generalization test, not the starting point.

## Models (verified 2026-10-09)
- LongSafari/hyenadna-tiny-1k-seqlen-hf (~451k params, trust_remote_code=True)
- LongSafari/hyenadna-tiny-1k-seqlen-d256-hf
- InstaDeepAI/nucleotide-transformer-v2-50m-multi-species
- InstaDeepAI/nucleotide-transformer-v2-100m-multi-species
Plus a classical-features baseline with no neural model at all.

## Data
- Genome: NCBI via biopython
- Labels: Goodall et al. 2018 mBio (PMC5821084) supplementary; alternatives are
  Keio collection (Baba et al. 2006) and EcoCyc essentiality pages

## Stack
Python 3.11, torch (CPU build is fine), transformers, biopython, scikit-learn,
pandas, pytest. Pinned.

## Conventions
- Device selection is explicit and logged: cpu, cuda, or mps.
- Embeddings are cached to disk keyed by (model, gene, pooling). Never recompute.
- Stratified cross-validation. Report MCC and per-class precision/recall.
- ALWAYS report the majority-class baseline next to every metric.
- Seeds set explicitly. Inference determinism verified.
- `python run_all.py --quick` uses 50 genes; full run uses all.

## Rules for Claude
- Verify each model loads before building a pipeline around it. HyenaDNA uses
  custom remote code and is the most likely thing to break.
- Run a 10-gene timing trial and report projected full runtime before any long
  job.
- Never report a metric you did not compute in a runnable script.
- Accuracy alone is forbidden in this repo. Essentiality is imbalanced (most
  genes are non-essential), so accuracy is misleading. MCC plus baseline.
- Watch for leakage: gene length correlates with essentiality and also with
  embedding statistics. Control for it explicitly.

## Decisions
- 2026-10-09: transformers pinned to 4.57.6. On 5.x the Nucleotide Transformer
  v2 remote code fails to import (`find_pruneable_heads_and_indices` was
  removed); HyenaDNA works on both.
- 2026-10-09: model revisions are pinned to commit hashes in
  scripts/check_models.py, because trust_remote_code runs the repo's code. The
  downloaded code was scanned: no network, subprocess, eval/exec or pickle.
- 2026-10-11: genome is CP009273.1 (BW25113). 4,490 genes, 4,131
  protein-coding. 99.5% carry a b-number via their ECK number. The strain's
  published deletions are visible and asserted in tests: araB and rhaB absent,
  lacZ a pseudogene. 18 genes have compound locations, so their start..end span
  exceeds their sequence length; genes.csv has a `segments` column and `length`
  is always the sequence.
- 2026-10-11: labels are Goodall et al. 2018 Table S1 (the transposon library
  itself); Table S4 is the same assay after outgrowth in LB, kept as a second
  condition. Three classes: Essential / Non-essential / Unclear. Genes are named,
  not b-numbered, so names resolve with the primary `gene` name taking priority
  over `gene_synonym` - without that, acrB looks ambiguous because it is also a
  historical synonym of gyrB.
- Join: 4,260 of 4,313 rows written, 0 collisions. The 53 unresolved are all
  insertion-sequence elements (insA, insB1, ...), which exist in many identical
  copies and are excluded deliberately.
- **The usable training set: 3,929 protein-coding genes with a clear call, 353
  essential and 3,576 non-essential. Imbalance 1:10.1, so the majority-class
  baseline is 91.02% accuracy.** That is the number to print beside every metric.
- The supplementary tables are blocked to scripts on PMC and had to come from
  journals.asm.org, which sits behind a Cloudflare check. They are committed to
  data/raw/ rather than re-downloaded.
- 2026-10-11: the canonical CV split lives in data/processed/cv_folds.csv
  (stratified 5-fold, seed 0) and is created once. Every model must load it, and
  tests/test_splits.py fails if it moves. `--quick` writes cv_folds_quick.csv
  instead, so a smoke run cannot fix the folds from a 200-gene subset.
- **Baseline to beat (results/baseline.csv): gradient boosting MCC 0.331
  (95% CI 0.288-0.372), AUROC 0.794. Logistic regression MCC 0.269.**
- **Gene length carries almost no signal: length alone gives MCC 0.041, and
  dropping all length features changes nothing (MCC 0.338 vs 0.331).** The
  leakage worry in the project plan is empirically absent here, so a positive
  language-model result cannot be dismissed as "it learned gene length". Still
  run the embedding-norm correlation check in the scaling step.
- 2026-10-11: **HyenaDNA's window is ~1 kb and 36.3% of protein-coding genes
  are longer than that**, so the long-sequence strategy is a real choice, not a
  detail: `chunk_mean` (default, embeds consecutive windows and averages),
  `truncate`, `center`. Nucleotide Transformer's 6-mer tokenizer reaches ~12 kb,
  so no gene hits its limit and the three strategies agree for it. Context
  truncation therefore affects only the HyenaDNA arm, and belongs in limitations.
- Measured embedding runtimes on this laptop (CPU, 3,929 genes): hyenadna-tiny
  2.3 min, hyenadna-tiny-d256 3.7 min, nt-v2-50m 15.1 min, nt-v2-100m 28.2 min.
- The model registry lives in src/models.py; embeddings cache to
  data/embeddings/<model>__<pooling>__<strategy>.npz, written atomically and
  checkpointed every 200 genes so an interrupted run resumes.
- 2026-10-11: **HEADLINE RESULT (results/scaling.csv, scaling.json).** No DNA
  language model beat the classical features. Best per model, MCC:
  hyenadna-tiny 0.235, tiny-d256 0.256, nt-v2-50m 0.271, nt-v2-100m 0.309,
  against a classical baseline of 0.331. A PAIRED bootstrap on identical genes
  and folds (not overlapping CIs) puts 7 of 8 model/classifier combinations
  significantly BELOW the baseline; only nt-v2-100m + gradient boosting is
  indistinguishable from it (dMCC -0.022, CI -0.067 to +0.023). None beats it.
- Scaling is weak and direction-dependent: across a 223x parameter increase,
  MCC rises 0.230 -> 0.309 with gradient boosting but FALLS 0.235 -> 0.228 with
  logistic regression. Never write "DNA language models scale as"; four models
  is not a scaling law. Say "across the four models tested".
- **Leakage check: nt-v2-50m's embedding norm correlates r=+0.79 (r^2=0.63)
  with gene length** -- a real and reportable property of that model. The other
  three are weak (-0.10, +0.06, -0.30). It does NOT explain performance,
  because length alone is not predictive here (length-only MCC -0.01), so a
  model encoding length is wasting capacity, not cheating.
- Paired comparisons need out-of-fold predictions, so baseline_features.py
  writes results/oof_baseline.npz and evaluate_scaling.py reads it. Re-run the
  baseline whenever the gene set changes or the comparison is skipped.
- 2026-10-11: **generalization (results/generalization.csv).** Trained on all
  3,929 E. coli genes, applied unchanged to 4,046 B. subtilis 168 genes
  (255 essential, majority 93.70%, labels from SubtiWiki v5 whose 258 essential
  agrees with Koo et al. 2017). EVERYTHING drops, and the ordering changes:
  classical 0.331 -> 0.198, nt-v2-100m 0.309 -> 0.190, nt-v2-50m 0.271 -> 0.144,
  hyenadna-tiny 0.230 -> -0.019 (i.e. useless). Classical features still lead,
  but nt-v2-100m is now within overlapping CIs of them. Logistic regression
  transfers BETTER than gradient boosting for the HyenaDNA models (less
  overfitting to E. coli), reversing the within-organism ordering.
- B. subtilis labels come from a LIVE database with no version stamp, so
  data/raw/subtiwiki_export.json is cached and its sha256 recorded in
  bsub_manifest.json. Do not re-download mid-analysis.
- CAI is recomputed per organism (52 ribosomal genes in E. coli, 55 in
  B. subtilis). Carrying E. coli codon weights into B. subtilis would not be a
  transfer test.
- CAI uses the 52 ribosomal protein genes (rps/rpl/rpm) as its reference set.
- Measured parameter counts (results/model_check.csv): HyenaDNA tiny 438,144
  (not ~451k), tiny-d256 1,645,184, NT v2 50M 55,904,972, NT v2 100M 97,889,132.

## This machine
- Windows 11 on ARM64: Snapdragon X Plus, 8 cores, 15.6 GB RAM. The GPU is a
  Qualcomm Adreno, which torch cannot use, so there is no cuda and no mps.
  Everything runs on cpu.
- Python is an x64 3.11 installed by uv, running under emulation, in `.venv\`.
  Run scripts with `.venv\Scripts\python`. Not the native ARM64 Python: torch
  ships only win_amd64 wheels on Windows.
