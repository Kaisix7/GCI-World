<div align="center">

# 🏆 GCI Global 2026: Home Credit Default Risk
### Advanced Credit Risk Scoring via Tri-Family Gradient Boosting & Non-Linear Solvency Modeling

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Frameworks](https://img.shields.io/badge/Models-LightGBM%20%7C%20CatBoost%20%7C%20XGBoost-orange.svg)](#model-architecture)
[![Evaluation](https://img.shields.io/badge/Evaluation-ROC--AUC-purple.svg)](#validation-strategy--results)
[![Status](https://img.shields.io/badge/Competition-GCI%20World%202026-gold.svg)](https://global.omnicamp.us/)

</div>

---

## 📌 Executive Summary

This repository presents the end-to-end machine learning solution for the **GCI Global 2026 In-class Machine Learning Competition: Home Credit Default Risk**. 

Predicting customer credit default on uncollateralized cash loans is a fundamental challenge in financial risk analytics, characterized by extreme class imbalance (~8% default rate), non-linear borrower interactions, and substantial missingness across external bureau scores. 

Our solution establishes a leak-free pipeline incorporating:
1. **Domain-Engineered Financial Health & Solvency Features**: Uncovered credit ratios, net family disposable cushions, and harmonic external agency aggregations.
2. **Unsupervised Gaussian Mixture Loan Term Segmentation**: Continuous log-term cluster assignments capturing differing repayment dynamics.
3. **Tri-Family Gradient Boosting Architecture**: Ensemble of Bayesian-tuned **LightGBM**, symmetric oblivious **CatBoost**, and histogram-based **XGBoost**.
4. **Rank-Weighted Power Calibration ($p=3.5$)**: Resolving inter-model probability calibration disparities to optimize rank discrimination.

### 🌟 Verified Benchmark Performance
- **Tri-Family Pooled Out-of-Fold (OOF) ROC-AUC:** `0.76805`
- **Omnicampus Public Leaderboard Verified Record:** `0.77148`

---

## 📊 Exploratory Data Analysis & Key Findings

<div align="center">
  <img src="images/eda/target_distribution.png" width="45%" alt="Target Distribution" />
  <img src="images/eda/ext_sources_density.png" width="49%" alt="EXT_SOURCE Density" />
</div>

- **Target Imbalance**: 157,370 negative samples (91.9%) vs 13,832 default samples (8.1%). Models must focus on threshold-invariant rank ordering rather than pure classification accuracy.
- **External Risk Scores (`EXT_SOURCE_1, 2, 3`)**: As depicted above, these three normalized scores exhibit the highest individual correlation with borrower default risk, but feature significant missingness (69.5% missing in `EXT_1`, 31.9% in `EXT_3`).
- **The Missingness Bottleneck**: A crucial failure analysis revealed that 22.5% of applicants lack both `EXT_1` and `EXT_3`. On this subset, standard default risk rises to 9.08%, necessitating non-linear solvency interactions (`DTI_OVER_EXT2_SQ`) to compensate for missing agency bureau signals.

---

## 🛠️ Feature Engineering Architecture

Feature engineering is organized across five distinct financial and demographic dimensions:

| Feature Category | Formula / Definition | Rationale & Predictive Value |
| :--- | :--- | :--- |
| **Credit Burden** | $AMT\_ANNUITY / AMT\_INCOME\_TOTAL$ | Direct measure of debt servicing stress on client cash flow. |
| **Loan Duration** | $AMT\_CREDIT / AMT\_ANNUITY$ | Estimated repayment period in months. |
| **Uncovered Credit** | $\max(0, CREDIT - GOODS) / CREDIT$ | Proportion of loan exceeding asset value (unsecured consumer credit). |
| **Family Disposable Surplus** | $\max(0, INCOME - ANNUITY) / (1 + FAM\_MEMBERS)$ | Remaining capital buffer per dependent after debt service. |
| **Harmonic External Risk** | $3 / \sum (1 / (EXT_i + \epsilon))$ | Heavily penalizes candidates with an acute risk flag in any single agency. |
| **Solvency Cushion** | $(INCOME - ANNUITY) \times EXT\_HARM$ | Combines liquidity surplus with agency-verified creditworthiness. |
| **Car-to-Tenure Ratio** | $OWN\_CAR\_AGE / EMPLOYED\_YEARS$ | Evaluates maintenance liability against employment stability. |
| **GMM Loan Term Clusters** | Gaussian Mixture ($K=3$) on $\log(\text{Term})$ | Identifies consumer finance vs long-term mortgage loan regimes. |

---

## 🧠 Model Architecture & Methodology

```text
                                  ┌────────────────────────┐
                                  │   Engineered Features  │
                                  └───────────┬────────────┘
                                              │
                      ┌───────────────────────┼───────────────────────┐
                      │                       │                       │
           ┌──────────▼──────────┐ ┌──────────▼──────────┐ ┌──────────▼──────────┐
           │      LightGBM       │ │       CatBoost      │ │       XGBoost       │
           │  (Leaf-wise GBDT)   │ │  (Symmetric Trees)  │ │ (Hist-based Splitting│
           └──────────┬──────────┘ └──────────┬──────────┘ └──────────┬──────────┘
                      │                       │                       │
                      └───────────────────────┼───────────────────────┘
                                              │
                                  ┌───────────▼───────────┐
                                  │   Power Rank Blend    │
                                  │       (p = 3.5)       │
                                  └───────────┬───────────┘
                                              │
                                  ┌───────────▼───────────┐
                                  │ Final Submission CSV  │
                                  └───────────────────────┘
```

1. **LightGBM (Leaf-wise GBDT)**:
   - Tuned via Bayesian Optimization with Optuna: `num_leaves=62`, `max_depth=4`, `learning_rate=0.025`, `colsample_bytree=0.69`, `subsample=0.81`, $L_1=2.36$, $L_2=2.32$.
   - Excels at capturing high-order non-linear feature splits.
2. **CatBoost (Symmetric Oblivious Trees)**:
   - Configured with native categorical support and Lossguide growth.
   - Provides exceptional regularization on noisy high-cardinality features (`OCCUPATION_TYPE`, `ORGANIZATION_TYPE`).
3. **XGBoost (Histogram Splitting)**:
   - Fast histogram binning (`tree_method='hist'`) with shallow depth (`max_depth=4`) to provide orthogonal gradient paths.

---

## 📈 Validation Strategy & Results

Validation is conducted strictly through **5-Fold Stratified Cross-Validation (`StratifiedKFold(n_splits=5, shuffle=True)`)**. All transformations and early-stopping decisions are executed within training partitions to eliminate target and feature leakage.

<div align="center">
  <img src="images/roc_curve.png" width="47%" alt="ROC Curves" />
  <img src="images/feature_importance.png" width="47%" alt="Feature Importance" />
</div>

### Model Comparison Table (5-Fold Stratified CV)

| Model Family | Hyperparameters / Setup | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Pooled OOF ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM (Optuna v2)** | `leaves=62, depth=4, lr=0.025` | 0.7584 | 0.7601 | 0.7768 | 0.7679 | 0.7648 | **0.76602** |
| **XGBoost (Solvency Hist)** | `depth=4, colsample=0.70, lr=0.03` | 0.7582 | 0.7604 | 0.7770 | 0.7672 | 0.7632 | **0.76510** |
| **CatBoost (Symmetric)** | `depth=6, l2=5.0, lr=0.035` | 0.7574 | 0.7582 | 0.7759 | 0.7665 | 0.7638 | **0.76428** |
| **5-Seed LightGBM Blend** | Multi-seed average (5 seeds) | 0.7602 | 0.7618 | 0.7785 | 0.7694 | 0.7662 | **0.76754** |
| **Tri-Family Power Rank Ensemble** | $0.60\,\text{LGB}^{3.5} + 0.35\,\text{CB}^{3.5} + 0.05\,\text{XGB}^{3.5}$ | **0.7608** | **0.7624** | **0.7792** | **0.7699** | **0.7668** | **`0.76805`** |

> **Leaderboard Note**: Deploying the Tri-Family calibrated ensemble to the Omnicampus Public Test set achieved a verified public leaderboard score of **0.77148 ROC-AUC**.

---

## 🗂️ Repository Structure

```text
GCI-World-2026/
├── README.md                          # Comprehensive project overview & documentation
├── requirements.txt                   # Production environment dependencies
├── .gitignore                         # Data protection & build exclusions
├── tutorial.ipynb                     # Submission tutorial notebook (Honors requirement)
├── tutorial.zip                       # Compressed notebook archive for evaluation
├── notebooks/
│   ├── final_solution.ipynb           # Complete executable end-to-end notebook
│   └── tutorial_original_backup.ipynb # Original competition starting tutorial
├── src/
│   ├── feature_engineering.py         # Leak-free feature transformation module
│   ├── train.py                       # 5-fold CV training pipeline for LGB/CB/XGB
│   └── predict.py                     # Inference and rank-blending script
├── data/
│   └── README.md                      # Dataset acquisition guidelines & schema definitions
├── outputs/
│   ├── submission.csv                 # Final verified submission file (61,500 rows)
│   └── cv_results.csv                 # Systematic experiment tracking log (64 runs)
├── images/
│   ├── feature_importance.png         # Publication-grade top-20 feature importance plot
│   ├── roc_curve.png                  # Multi-model ROC comparison plot
│   └── eda/
│       ├── target_distribution.png    # Class balance visualization
│       └── ext_sources_density.png    # External agency distribution densities
└── docs/
    ├── competition_rules.md           # Formal rules, integrity & submission specifications
    └── methodology.md                 # Detailed mathematical & architectural documentation
```

---

## 🚀 Quickstart & Reproducibility Guide

### 1. Environment Setup
Clone this repository and create a Python 3.10+ virtual environment:

```bash
git clone https://github.com/your-username/GCI-World-2026.git
cd GCI-World-2026

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required packages
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Dataset Setup
In compliance with competition rules, raw data is not hosted on GitHub.
1. Download `train.csv`, `test.csv`, and `HomeCredit_columns_description.xlsx` from [Omnicampus](https://global.omnicamp.us/).
2. Place the files in `data/` or `input/` as described in [`data/README.md`](data/README.md).

### 3. Training & Generating Predictions
Run the 5-fold cross-validation pipeline across all three model architectures:

```bash
# Execute end-to-end training and inference
python src/train.py --data-dir input --output-dir outputs --seed 42
```

Alternatively, open and run the complete Jupyter Notebook:
```bash
jupyter notebook notebooks/final_solution.ipynb
```

---

## 📜 Competition Rules & Ethical Compliance

This project strictly adheres to the **GCI Global 2026 Official Tournament Regulations**:
- **Zero External Data**: Built exclusively using official `train.csv` and `test.csv`.
- **Zero Hand-Labeling**: All test predictions are 100% algorithmically generated.
- **Strict Leakage Prevention**: Encodings and statistical scaling are strictly partitioned within CV folds.
- **Attribution**: References and tutorial benchmarks from the course materials are duly cited below.

For complete regulatory details, consult [`docs/competition_rules.md`](docs/competition_rules.md).

---

## 📚 Acknowledgements & References

1. **Home Credit Group & Kaggle**: Original 2018 Home Credit Default Risk benchmark dataset [[Kaggle Link](https://www.kaggle.com/c/home-credit-default-risk)].
2. **GCI Global 2026 Course Staff & TAs**: Office hours, tutorials, and materials:
   - *Tutorial 01: Adding Features* (Handling anomalies & basic financial ratios)
   - *Tutorial 02: Advanced Feature Engineering* (Domain interactions & group aggregations)
   - *Tutorial 03: Hyperparameter Tuning* (Optuna Bayesian Optimization frameworks)
   - *Tutorial 04: Model Ensembling* (Blending paradigms & diversity optimization)
3. **Open-Source Libraries**: [LightGBM](https://github.com/microsoft/LightGBM), [CatBoost](https://github.com/catboost/catboost), [XGBoost](https://github.com/dmlc/xgboost), [Scikit-Learn](https://scikit-learn.org/).

---

<div align="center">
  <b>Developed for GCI Global 2026 • Honors Submission Ready</b>
</div>
