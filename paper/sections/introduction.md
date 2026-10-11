Language models trained on DNA have grown quickly, from a few hundred thousand
parameters to tens of billions, and the implicit promise of that growth is that
larger models learn more biology. For most downstream tasks that promise is
rarely tested directly against the cheap alternative. A model is shown to work,
and the comparison is to another model.

Gene essentiality is a useful place to test it. Whether a bacterium can survive
losing a gene is a coarse, binary, biologically central property; it has been
measured genome-wide in several organisms by transposon mutagenesis and
systematic deletion, so labels exist in quantity and are not derived from
sequence. It is also a property with a well-known cheap predictor: essential
genes are enriched for high expression, and high expression leaves a clear
signature in codon usage and base composition that a short script can measure.
Any claim that a language model has learned something about essentiality has to
clear that signature first.

The question we ask is therefore not whether a frontier model works, which
anyone with a sufficiently large machine can answer, but how much model is
needed before the signal appears at all — and whether it ever exceeds what
free sequence statistics already provide. This framing matters practically.
The largest genome models require datacenter accelerators, and much of the
genomics that would benefit from them happens on modest hardware. If a model
small enough to run on a laptop retains most of the available signal, that is
a useful result for those users. If no model of any size beats counting codons,
that is a more useful result still.

We compare four DNA language models spanning 438 thousand to 98 million
parameters, drawn from two architectures with different tokenisation and
context behaviour, against a classical sequence-feature baseline. All models
are evaluated as frozen feature extractors with a shallow classifier, on
identical cross-validation folds, with the majority-class rate reported beside
every metric and an explicit control for the most likely confound. We then test
whether anything learned transfers to a second organism without retraining.
