# Dataset

## Source

The data are from the DeepHF study:

> Wang D, Zhang C, Wang B, Li B, Wang Q, Liu D, Wang H, Zhou Y, Shi L, Lan F, et al.
> *Optimized CRISPR guide RNA design for two high-fidelity Cas9 variants by deep learning.*
> Nature Communications 10:4284 (2019). doi:10.1038/s41467-019-12281-8

That study measured sgRNA **on-target** activity, which is what the models here predict.

## Access

Sequence Read Archive accession
[PRJNA522677](https://www.ncbi.nlm.nih.gov/bioproject/522677/).

## What this work uses

- Wild-type SpCas9 activity only. The eSpCas9 and SpCas9-HF data in the same study are not
  combined with it, because their activity distributions differ.
- Each sgRNA is 20 nucleotides plus the variable PAM nucleotide, so 21 in total.
- 15% held out for testing, the remaining 85% used for hyperparameter tuning and training.

## Encoding

One-hot, giving a 21 x 4 matrix, then an embedding layer to 128 dimensions.
GC content is computed as the proportion of G and C in the sequence and appended as a single
input in the last layer.
