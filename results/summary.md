Repeated stratified 5-fold CV x 3 repeats (K = 100 projections, d' = 100, NNV epochs = 500, seed = 42). Mean ± standard deviation over all splits, in %.

**colon** (62 samples, 2,000 genes, classes {'-1': 40, '1': 22})

| Method | Accuracy | Balanced accuracy |
|---|---|---|
| RF | 84.5 ± 9.4 | 83.2 ± 11.2 |
| SMV-MRPV | 87.2 ± 8.8 | 86.8 ± 9.4 |
| NNV-MRPV | 82.3 ± 9.1 | 80.5 ± 10.6 |
| NNV in-sample (ablation) | 85.5 ± 7.9 | 83.9 ± 10.0 |

**leukemia** (72 samples, 7,129 genes, classes {'ALL': 47, 'AML': 25})

| Method | Accuracy | Balanced accuracy |
|---|---|---|
| RF | 96.7 ± 5.3 | 95.3 ± 7.4 |
| SMV-MRPV | 95.3 ± 4.4 | 93.7 ± 6.1 |
| NNV-MRPV | 93.1 ± 6.3 | 93.5 ± 5.2 |
| NNV in-sample (ablation) | 96.7 ± 3.6 | 96.3 ± 4.4 |

**prostate** (102 samples, 12,600 genes, classes {'1': 52, '2': 50})

| Method | Accuracy | Balanced accuracy |
|---|---|---|
| RF | 89.9 ± 6.2 | 89.9 ± 6.3 |
| SMV-MRPV | 90.2 ± 7.3 | 90.2 ± 7.3 |
| NNV-MRPV | 85.0 ± 8.2 | 84.9 ± 8.3 |
| NNV in-sample (ablation) | 88.0 ± 7.7 | 88.0 ± 7.8 |
