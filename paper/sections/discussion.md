Across the four models tested, no DNA language model extracted essentiality
signal beyond what gene length, base composition and codon statistics already
provide. Seven of eight model-classifier combinations performed significantly
worse than that baseline in paired comparisons on identical genes; the best
merely drew level. The result should be stated with its scope attached: four
models, two architectures, one training organism, frozen representations. It is
not a scaling law, and architecture is not separable from scale here.

The baseline's strength is not an accident, and it suggests why the comparison
goes the way it does. Essential genes are enriched for high expression, and
highly expressed genes have distinctive synonymous codon usage. The classical
features measure that directly: the codon adaptation index and the effective
number of codons are, in effect, hand-built expression proxies refined over
decades. A pretrained language model must recover the same signal indirectly
from a generic objective, and then survive compression into a single pooled
vector. On a task whose best-known correlate is already a cheap summary
statistic, the hand-built feature has a real advantage.

The transfer result sharpens this. Applied to *B. subtilis* without
retraining, every representation degraded, but not equally: the two smallest
models fell to chance, while the larger ones retained roughly the same
proportion of their performance as the classical features did. Within a single
organism the small models had looked respectable. They were, it appears,
fitting *E. coli*'s particular composition rather than anything about
essentiality. That is a more interesting conclusion than a flat "scaling does
not help", and it is the one place in this study where parameters bought
something: not accuracy, but generality. A reader deciding whether to run a
small model on their own hardware should weigh that. Within the organism you
have labels for, the small model is nearly as good; outside it, it is nothing.

The classifier interacts with this in a way worth noting. For the HyenaDNA
models, logistic regression transferred better than gradient boosting, the
reverse of the within-organism ordering. The flexible model exploits
organism-specific structure that does not survive the move. Evaluations that
report only the best classifier per representation would miss this entirely.

Gene length deserves comment because it is the standard objection. Here it is
not available as an explanation: length alone is no better than chance, and
removing every length feature leaves the baseline unchanged. One model's
embedding norms nevertheless track length closely, which says something about
what that representation encodes but cannot account for its predictions. On a
dataset where length did predict the label, the same correlation would be a
serious confound, and the check is cheap enough that it should be routine.

What this study does not show is that these models contain no essentiality
information. Frozen embeddings with a shallow classifier probe what is linearly
and non-linearly accessible in the pretrained representation without further
training; fine-tuning may well do better, and the question of how much better
is the natural next experiment. Nor does it speak to tasks where context
genuinely matters — regulatory prediction, operon structure, long-range
interactions — where a model that reads kilobases has an advantage no codon
count can replicate. The claim is narrower: for this task, at this scale, as
frozen features, the expensive option did not beat the free one.
