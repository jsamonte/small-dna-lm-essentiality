Across the four models tested, no DNA language model extracted essentiality
signal beyond what gene length, base composition and codon statistics already
provide. Seven of eight model-classifier combinations performed significantly
worse than that baseline in paired comparisons on identical genes; the best
merely drew level. The result should be stated with its scope attached — four
models, two architectures, one training organism, frozen representations. It is
not a scaling law, and architecture is not separable from scale here.

Moreover, the baseline's strength is not an accident, and it suggests why the
comparison goes the way it does. Essential genes are enriched for high
expression, and highly expressed genes have distinctive synonymous codon usage.
The classical features measure that directly. The codon adaptation index and
the effective number of codons are, in effect, hand-built expression proxies
refined over decades. A pretrained language model must recover the same signal
indirectly from a generic objective, and then survive compression into a single
pooled vector. On a task whose best-known correlate is already a cheap summary
statistic, the hand-built feature has a real advantage.

The transfer result sharpens this. Applied to *B. subtilis* without retraining,
every representation degraded, but not equally. More specifically, the two
smallest models fell to chance with gradient boosting and kept only about 40%
of their within-organism performance with logistic regression, while the larger
ones retained roughly the same proportion as the classical features did. Within
a single organism the small models had looked respectable. They were, it
appears, fitting *E. coli*'s particular composition rather than anything about
essentiality. However, that reading needs a control, and the control fails it
in part. Denied a *B. subtilis* codon reference and paired with gradient
boosting, the classical features do exactly the same thing, dropping from
0.198 to −0.007. Fitting the training organism's composition is therefore a
property of the combination rather than a failing peculiar to small models.
What survives is narrower and still worth having, which is that the largest
model reaches 0.190 on transfer with no organism-specific information of any
kind, and no small model here came close to that. A reader deciding whether to run a
small model on their own hardware should weigh that. Within the organism you
have labels for, the small model is nearly as good. Without labels it is much
weaker.

One alternative explanation belongs here rather than in the list of
limitations, because it bears directly on the claim just made. Nucleotide
Transformer v2 is multi-species and very likely saw both genomes during
pretraining. "The larger model generalises" and "the larger model has also
memorised *B. subtilis*" predict the same table, and nothing in this study
separates them. The transfer result should be read with that attached, and a
clean test would need an organism absent from the pretraining corpus.

Of equal importance, the classifier interacts with this in a way worth noting.
For the HyenaDNA models, logistic regression and gradient boosting were nearly
tied within *E. coli*, but on transfer gradient boosting collapsed to chance
while logistic regression retained about 40% of its performance. The flexible
model exploits organism-specific structure that does not survive the move, and
the cleanest demonstration is the classical features themselves. Under logistic
regression they transfer at 0.187 with no recalibration at all; under gradient
boosting the same features, given the same information, collapse to zero.
Evaluations that report only the best classifier per representation would miss
this entirely.

Gene length deserves comment because it is the standard objection. Here it is
not available as an explanation. Length alone carries almost no signal and
removing every length feature leaves the baseline unchanged, but one model's
embedding norms nevertheless track length closely, which says something about
what that representation encodes. However, it cannot account for its
predictions. On a dataset where length did predict the label, the same
correlation would be a serious confound, and the check is cheap enough that it
should be routine.

Altogether, what this study does not show is that these models contain no
essentiality information. Frozen embeddings with a shallow classifier probe
what is linearly and non-linearly accessible in the pretrained representation
without further training. Fine-tuning may well do better, and the question of
how much better is the natural next experiment. Nor does it speak to tasks
where context genuinely matters, such as regulatory prediction, operon
structure and long-range interactions, where a model that reads kilobases has
an advantage no codon count can replicate. The final claim, although narrower,
is that for this task, at this scale, as frozen features, in either organism,
the expensive option did not beat the free one.
