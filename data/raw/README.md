# Third-party data redistributed here

## `goodall2018_st1.xlsx`, `goodall2018_st4.xlsx`

Supplementary Tables S1 and S4 of:

> Goodall ECA, Robinson A, Johnston IG, Jabbari S, Turner KA, Cunningham AF,
> Lund PA, Cole JA, Henderson IR (2018) The Essential Genome of *Escherichia
> coli* K-12. *mBio* 9(1):e02096-17. doi:10.1128/mBio.02096-17

Copyright the authors. mBio has published open-access articles under the
Creative Commons Attribution 4.0 International licence (CC BY 4.0) since 2016,
which permits redistribution with attribution; this file is that attribution.
Confirm the licence line on the article's own landing page before relying on
it for anything beyond reuse of the kind made here.

These two files are committed rather than downloaded because the publisher
sits behind a check that blocks scripted retrieval and the PMC copies are not
machine-reachable either, so a fresh clone could not otherwise reproduce
`results/`. Everything else under `data/raw/` is ignored by git.

Nothing in these files has been altered. `scripts/parse_essentiality.py`
reads them and writes the processed tables used by the analysis.
