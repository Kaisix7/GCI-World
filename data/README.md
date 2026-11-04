# Dataset Guidelines: GCI World 2026 Home Credit Default Risk

## Overview
This directory is intended to contain the official competition datasets provided for the **GCI Global 2026 / Home Credit Default Risk Competition**.

## Strict Data Distribution Restrictions
In strict adherence to the **GCI World 2026 Competition Rules** and academic integrity guidelines:
- **Raw competition datasets must NOT be redistributed publicly or committed to version control.**
- Ensure that `train.csv` and `test.csv` remain strictly in your local environment.

## Expected Directory Setup
To run the training and evaluation pipelines, ensure the files are placed in this folder (`data/` or `input/`) with the following exact names and schemas:

```text
data/ (or input/)
├── train.csv                      # Official training data (171,202 rows, 34 columns)
├── test.csv                       # Official test data (61,500 rows, 33 columns)
├── sample_submission.csv          # Sample submission template (61,500 rows)
└── HomeCredit_columns_description.xlsx  # Official column descriptions
```

### Dataset Schemas:
- **`train.csv`**: Contains `SK_ID_CURR` (loan application identifier), 32 explanatory variables (demographic, financial, external credit scores), and `TARGET` (`1` = client with payment difficulties, `0` = all other cases).
- **`test.csv`**: Contains `SK_ID_CURR` (IDs `171202` to `232701`) and 32 explanatory variables.
- **Target Distribution**: ~8.07% positive defaults (unbalanced binary classification).

## Obtaining the Data
Authorized students and participants can download the official dataset zip from the course assignment portal:
- Course: GCI Global 2026
- Platform: [Omnicampus](https://global.omnicamp.us/)
