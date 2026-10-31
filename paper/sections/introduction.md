Language models trained on DNA have grown quickly, from a few hundred thousand
parameters to the 40 billion of Evo 2 (Brixi et al., 2026), and the implicit
promise of that growth is that larger models learn more biology. For most downstream tasks that promise is
rarely tested directly against the cheap alternative. A model is shown to work,
and the comparison is to another model.

Gene essentiality is a useful place to test it. Whether a bacterium can survive
losing a gene is a coarse, binary, biologically central property. Also gene
essentiality has been measured genome-wide in several organisms by transposon
mutagenesis and systematic deletion, so labels exist in quantity and are not
derived from sequence. Gene essentiality is also a property with a well-known
cheap predictor, namely that essential genes are enriched for high expression.
Furthermore, high expression leaves a clear signature in codon usage and base
composition that a short script can measure. Any claim that a language model
has learned something about essentiality has to clear that signature first.

The question we ask is therefore not whether a frontier model works, which
anyone with a sufficiently large machine can answer, but how much model is
needed before the signal appears at all as well as whether it ever exceeds what
free sequence statistics already provide. That framing matters practically.
The largest genome models require datacenter accelerators, and much of the
genomics that would benefit from them happens on modest hardware. If a model
small enough to run on a laptop retains most of the available signal, that is a
useful result for those users. If no model of any size beats counting codons,
that is a more useful result still.

Therefore, we compare four DNA language models spanning 438 thousand to 98
million parameters, drawn from two architectures with different tokenisation and
context behaviour, against a classical sequence-feature baseline. On identical
cross-validation folds, all models are evaluated as frozen feature extractors
with a shallow classifier, and we report the majority-class rate beside every
metric together with an explicit control for the most likely confound. We then
test whether anything learned transfers to a second organism without retraining.
