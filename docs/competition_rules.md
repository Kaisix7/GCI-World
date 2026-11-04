# Competition Rules & Submission Specifications

> [!NOTE]
> **Status: Concluded Competition Benchmark**
> The active competition period for **GCI Global 2026** has concluded. This document preserves the official tournament rules, validation standards, and submission format as historical benchmark specifications under which this solution was developed, audited, and evaluated.

---

## 1. Overview & Objectives
The **GCI Global 2026 Home Credit Default Risk Competition** was an educational machine learning challenge adapted from the 2018 Kaggle tournament. The primary task was to estimate default probability for loan applicants based on a tabular profile of demographic, financial, credit history, and external credit agency attributes.

- **Primary Target**: Binary target variable `TARGET`:
  - `1`: Applicant encountered payment difficulties (default).
  - `0`: Applicant repaid without significant payment difficulties.
- **Evaluation Metric**: Area Under the Receiver Operating Characteristic Curve (**ROC-AUC**), evaluated on continuous probability predictions against ground-truth labels.

---

## 2. Dataset Structure
Participants were restricted to a structured subset of the original Home Credit tables:
- `train.csv`: 171,202 loan applications with associated binary labels in `TARGET` and 32 explanatory variables.
- `test.csv`: 61,500 unlabelled loan applications with sequential identifiers `SK_ID_CURR` from `171202` to `232701`.
- `HomeCredit_columns_description.xlsx`: Feature definitions provided by the organizers.

---

## 3. Strict Tournament Regulations

### 3.1 Prohibition of External Data
No external datasets, macroeconomic indices, alternative credit databases, or scraped records were permitted at any step of data preparation, feature engineering, or modeling. All models were required to train strictly on the official input files.

### 3.2 Prohibition of Hand-Labeling & Rule-Based Overrides
Manual modification of test predictions was prohibited. Assigning arbitrary scores, hardcoded rules, or heuristic threshold overrides based on manual inspection violated tournament integrity. All submitted predictions had to be generated programmatically by reproducible machine learning models capable of scoring arbitrary unseen samples.

### 3.3 Target Leakage & Validation Discipline
All feature engineering, transformations, target encoding, and imputation parameters had to be estimated strictly from training fold partitions during cross-validation. Held-out validation folds and test data were not permitted to influence data transformation statistics.

### 3.4 Reproducibility Standard
All randomized routines (e.g. data splitting, model seed initializations, cross-validation shuffles) were required to be explicitly seeded. Models were required to yield deterministic, reproducible out-of-fold and test predictions when executed in identical library environments.

---

## 4. Leaderboard Scoring & Submission Requirements

### 4.1 Evaluation Splits
- **Public Leaderboard**: Computed during the competition on an undisclosed subset of the test dataset to track progress and prevent leaderboard overfitting.
- **Private Leaderboard**: Evaluated on the remaining test instances at tournament close to establish final standings.
- **Selection Policy**: Official ranking was determined by the participant's **last submitted file** on the Omnicampus platform.

### 4.2 Submission Format
Predictions were uploaded via the Omnicampus competition portal formatted as a two-column CSV:
- **Columns**: `SK_ID_CURR,TARGET`
- **Rows**: Exactly 61,500 rows matching the sequential test identifiers (`171202` through `232701`).
- **Data Integrity**: No missing values (`NaN`), infinities, or nulls allowed. Target predictions had to be bounded within $[0.0, 1.0]$. Invalid formatting resulted in an automatic score of `-1`.

---

## 5. Honors & Outstanding Student Criteria
Participants qualifying for Honors or Outstanding Student distinctions were required to submit complete, self-contained, and reproducible source code that generates the exact submitted CSV prediction file from raw inputs.

---

## 6. Information Sharing & Academic Attribution
- **Collaboration**: Public discussion and knowledge sharing among students were permitted and encouraged.
- **Attribution & Code Disclosure**: High-ranking participants were required to disclose their codebase. When referencing open-source techniques, public kernels, or community literature, original sources were required to be explicitly credited. Failure to cite external inspiration during code review could result in disqualification.
