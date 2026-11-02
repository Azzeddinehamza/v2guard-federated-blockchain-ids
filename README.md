# Data

The datasets are **not** included in this repository (size and licensing). Download them from the
original sources, then preprocess into `data/KDDCombined.txt`.

## NSL-KDD

- **Source:** Canadian Institute for Cybersecurity, University of New Brunswick
- **Page:** https://www.unb.ca/cic/datasets/nsl.html
- **Files:** `KDDTrain+.txt`, `KDDTest+.txt` (41 features + label, CSV without header)
- **Preprocessing used in the thesis:** concatenate train + test, label-encode `protocol_type`,
  `service`, `flag`, oversample minority classes (`RandomOverSampler`), split into 6 non-IID
  client shards, and derive binary (`normal` vs. attack) and multiclass labels.
- **Feature subset:** the 22 features selected in `src/federated_learning.py` (`selected_features`).

## UNSW-NB15

- **Source:** UNSW Canberra Cyber Range
- **Page:** https://research.unsw.edu.au/projects/unsw-nb15-dataset
- **Files:** `UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv`
- **Notes:** 49 features, 9 attack categories + normal. The thesis notebooks in
  `notebooks/` (not shipped here — see the thesis) cover the cleaning and sharding.

## Reproducing the preprocessed file

```bash
# expected by src/federated_learning.py:
#   data/KDDCombined.txt  — 41 comma-separated columns + label, no header
```

The feature names list is already embedded in the script (`feature_names`); column order must
match the standard NSL-KDD order.

## Ethics

Both datasets are public benchmark datasets of **simulated/collected network traffic**, not
personal data from real users. No real charging sessions or driver data are used anywhere in this
project.
