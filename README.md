# Random-projection ensembles vs. random forest on high-dimensional gene-expression data

Gene-expression microarrays have far more features (genes) than samples (patients). This project
compares three classifiers on three classic cancer datasets with 2,000–12,600 genes and only 62–102
samples:

- **RF:** random forest, 500 trees. The baseline.
- **SMV-MRPV:** K = 100 Gaussian random projections to d′ = 100 dimensions, one LDA classifier per
  projection, combined by **simple majority vote**.
- **NNV-MRPV:** the same projections and LDA classifiers, with their class probabilities combined by
  a small **neural network** (Dense 32 → Dropout 0.5 → Dense 128 → Dropout 0.2 → softmax, 500
  epochs). This is a stacked meta-learner.

## Results

Repeated stratified 5-fold cross-validation × 3 repeats = 15 test folds per dataset. Values are
mean ± SD over folds, in %, produced by `run_experiments.py` in GitHub Actions
(see [`results/`](results/)).

| Dataset | Samples / genes | Method | Accuracy | Balanced accuracy |
|---|---|---|---|---|
| Colon | 62 / 2,000 | RF | 84.5 ± 9.4 | 83.2 ± 11.2 |
| | | SMV-MRPV | **87.2 ± 8.8** | **86.8 ± 9.4** |
| | | NNV-MRPV | 82.3 ± 9.1 | 80.5 ± 10.6 |
| | | NNV, in-sample meta-features | 85.5 ± 7.9 | 83.9 ± 10.0 |
| Leukemia | 72 / 7,129 | RF | **96.7 ± 5.3** | 95.3 ± 7.4 |
| | | SMV-MRPV | 95.3 ± 4.4 | 93.7 ± 6.1 |
| | | NNV-MRPV | 93.1 ± 6.3 | 93.5 ± 5.2 |
| | | NNV, in-sample meta-features | **96.7 ± 3.6** | **96.3 ± 4.4** |
| Prostate | 102 / 12,600 | RF | 89.9 ± 6.2 | 89.9 ± 6.3 |
| | | SMV-MRPV | **90.2 ± 7.3** | **90.2 ± 7.3** |
| | | NNV-MRPV | 85.0 ± 8.2 | 84.9 ± 8.3 |
| | | NNV, in-sample meta-features | 88.0 ± 7.7 | 88.0 ± 7.8 |

**What the results show**

- **SMV-MRPV is the most reliable random-projection method.** It is the best on colon and prostate.
  On leukemia, random forest is slightly better.
- **All differences are smaller than the fold-to-fold spread.** The SDs are 4–11 points, so no
  method is clearly better than the others on these small datasets.
- **The neural-network combiner does not beat majority voting.** Averaged over the 15 folds, SMV is
  ahead of NNV-MRPV by 6.3 points in balanced accuracy on colon, 0.1 on leukemia and 5.2 on prostate.

## Training the meta-learner: in-sample vs. out-of-fold

A stacked meta-learner is normally trained on **out-of-fold** predictions of the base classifiers:
predictions for samples that each LDA did not see during fitting. If it is trained on in-sample
predictions instead, it learns from votes that are almost always correct. In the tests, LDA after
projection to 100 dimensions classifies its own training samples nearly perfectly, while its
accuracy on held-out samples is much lower.

`NNV-MRPV` here uses out-of-fold meta-features. Within each training fold:

1. The same K projections are kept.
2. The LDA classifiers are refitted in an inner 5-fold split.
3. The neural network is trained on their held-out probabilities.
4. The final LDA classifiers are fitted on the whole training fold and give the meta-features of
   the test samples.

The original version (in-sample meta-features) is kept as a comparison row.

**Result: out-of-fold training did not help here.** It was lower by about 3 points of balanced
accuracy on all three datasets. Over the 45 test folds it was:

- lower in 22;
- higher in 7;
- tied in 16.

So the methodological concern did not turn into a measurable loss on these data. A likely reason
(not tested here) is that the out-of-fold meta-learner is trained on noisier inputs, fitted from only
80% of an already small training fold.

## Run it

```bash
pip install -r requirements.txt
python run_experiments.py --out results              # all datasets, about 45-60 min on a CPU
python run_experiments.py --K 50 --skip-ablation     # quicker variant
python -m pytest -q tests
```

The data are downloaded from OpenML (`fetch_openml(name, version=1)` for `leukemia`, `colon` and
`prostate`). Every split, projection matrix and network is seeded, so results are reproducible up to
floating-point differences between machines.

## Changes from the first version

- **Repeated CV.** Repeated stratified CV (3 × 5 folds) replaces a single 5-fold split, and balanced
  accuracy is reported next to accuracy.
- **Seeding.** All random parts are seeded. Previously the projections and the network were not, so
  results changed from run to run.
- **Out-of-fold NNV.** The neural-network combiner now uses out-of-fold meta-features, and the
  in-sample version is kept as a comparison.
- **More trees.** Random forest uses 500 trees instead of 100.
- **Reproducible results.** Results are produced by a script and saved as CSV, instead of being
  typed into a Quarto report.

## Limitations

- **Small, well-studied datasets.** Three datasets with 62–102 samples each, so differences between
  methods are within the noise.
- **No hyperparameter tuning.** K, d′ and the network architecture are fixed, as in the first
  version of this project.
