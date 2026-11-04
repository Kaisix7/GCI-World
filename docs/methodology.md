# Technical Methodology: Tri-Family Gradient Boosting & Non-Linear Solvency Modeling

## Overview
This document provides an in-depth breakdown of the feature engineering architecture, model training methodology, validation protocols, and post-processing ensembling implemented for the **GCI Global 2026 Home Credit Default Risk Competition**.

---

## 1. Feature Engineering Architecture

The feature matrix is constructed across five primary pillars:

### A. Core Financial Burden & Solvency Ratios
1. **`CREDIT_TO_INCOME`**: Loan amount divided by total annual client income ($AMT\_CREDIT / AMT\_INCOME\_TOTAL$). Measures overall debt exposure.
2. **`ANNUITY_TO_INCOME`**: Annual debt servicing obligation relative to income ($AMT\_ANNUITY / AMT\_INCOME\_TOTAL$). Captures monthly cash flow stress.
3. **`CREDIT_TERM`**: Total loan amount divided by payment annuity ($AMT\_CREDIT / AMT\_ANNUITY$). Proxies loan duration in months.
4. **`GOODS_TO_CREDIT`**: Goods price relative to credit extended ($AMT\_GOODS\_PRICE / AMT\_CREDIT$).
5. **`UNCOVERED_CREDIT_RATIO`**: Uncollateralized portion of the credit amount:
   $$\text{Uncovered Ratio} = \frac{\max(0, AMT\_CREDIT - AMT\_GOODS\_PRICE)}{AMT\_CREDIT}$$
   High uncovered credit ratios correlate strongly with loan default.
6. **`SURPLUS_PER_FAMILY`**: Net disposable income cushion divided by total family members:
   $$\text{Surplus per Family} = \frac{\max(0, AMT\_INCOME\_TOTAL - AMT\_ANNUITY)}{1 + CNT\_FAM\_MEMBERS}$$

### B. Normalized Lifetime & Demographic Indicators
- **`AGE_YEARS`**: Normalized age computed as $-DAYS\_BIRTH / 365.25$.
- **`EMPLOYED_YEARS`**: Cleaned employment tenure, replacing the pensioner anomaly ($365,243$) with `NaN`.
- **`EMPLOYED_TO_AGE`**: Proportion of adult life spent in active employment.
- **`CAR_AGE_TO_EMP_YEARS`**: Ratio of owned vehicle age to employment tenure. Uncovers maintenance costs relative to career stability.
- **`PHONE_TO_AGE`**, **`REGISTRATION_TO_AGE`**, **`ID_TO_AGE`**: Recency of life document updates normalized by age.

### C. External Risk Score Combinations (`EXT_SOURCE`)
The external credit agency scores (`EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`) represent the strongest individual predictors in the dataset.
- **Aggregates**: Row-wise arithmetic mean (`EXT_MEAN`), standard deviation (`EXT_STD`), minimum, and maximum.
- **`EXT_HARM`**: Flexible Harmonic Mean across available scores:
  $$H = \frac{3}{\sum_{i=1}^3 \frac{1}{EXT_i + \epsilon}}$$
  Harmonic averaging penalizes applicants with an extreme failure in even one external agency score.
- **Cross-Domain Solvency Interactions**:
  - `DTI_OVER_EXT_PRODUCT`: Debt-to-income ratio scaled by the product of external scores.
  - `DTI_OVER_EXT2_SQ`: Debt-to-income scaled by the square of `EXT_SOURCE_2` (designed specifically for applicants missing agency 1 and 3).
  - `SOLVENCY_CUSHION`: Net disposable income multiplied by harmonic external credit score.

### D. Gaussian Mixture Loan Term Clustering (GMM)
Unsupervised Gaussian Mixture Modeling ($K=3$ components) fit on $\log(AMT\_CREDIT / AMT\_ANNUITY)$ cleanly segments loans into short-term consumer credit, medium-term cash loans, and long-term financing regimes.

---

## 2. Multi-Model Architecture: Tri-Family Gradient Boosting

To minimize model bias and maximize error diversity, three distinct gradient boosting implementations were deployed:

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

1. **LightGBM (Leaf-wise Growth)**:
   - Configured with Bayesian-optimized hyperparameters: `num_leaves=62`, `max_depth=4`, `learning_rate=0.025`, `subsample=0.81`, `colsample_bytree=0.69`, $L_1=2.36$, $L_2=2.32$.
   - Captures deep asymmetric interactions efficiently.
2. **CatBoost (Symmetric Oblivious Trees)**:
   - Evaluated with native categorical handling and depth-5 to depth-6 symmetric trees.
   - Immune to overfitting on categorical combinations and robust to high missingness patterns.
3. **XGBoost (Histogram Splitting)**:
   - Histogram-based tree method (`tree_method='hist'`) with conservative depth (`max_depth=4`), colsample $0.70$, and early stopping.

---

## 3. Validation Strategy & Leakage Prevention

- **Protocol**: 5-Fold Stratified Cross-Validation (`StratifiedKFold(n_splits=5, shuffle=True)`).
- **Leakage Prevention**:
  - GMM cluster boundaries and imputer statistics are fit strictly on training fold partitions.
  - Early stopping is evaluated against held-out validation partitions, ensuring that model complexity is not inflated.
- **Metric Verification**:
  - Out-of-fold predictions are concatenated across all 5 folds to compute the true **Pooled OOF ROC-AUC**.

---

## 4. Rank-Weighted Power Ensembling

Rather than simple linear probability averaging (which distorts confidence when models have different probability calibration curves), we employ **Power Rank Ensembling** ($p=3.5$):

$$\text{Score}_i = \left( \frac{\text{Rank}(P_i)}{N} \right)^{3.5}$$

$$\text{Ensemble Pred} = w_{\text{lgb}} \cdot \text{Score}_{\text{lgb}} + w_{\text{cb}} \cdot \text{Score}_{\text{cb}} + w_{\text{xgb}} \cdot \text{Score}_{\text{xgb}}$$

Power ranking sharpens discrimination at the extreme default deciles while preserving ranking order across the middle probability range.
