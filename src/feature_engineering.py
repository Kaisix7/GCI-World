"""
Feature Engineering Module for Home Credit Default Risk.
Constructs domain financial ratios, lifetime & demographic metrics,
EXT_SOURCE interactions, and Gaussian Mixture Loan Term Clusters.
"""

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

def engineer_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    gmm_random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Apply comprehensive feature engineering across train and test sets.
    
    Returns:
        train_feat: Processed train DataFrame
        test_feat: Processed test DataFrame
        feature_names: List of all predictor column names
        categorical_cols: List of categorical feature column names
    """
    full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    
    # 1. Clean anomalous values
    full_df["DAYS_EMPLOYED_CLEAN"] = full_df["DAYS_EMPLOYED"].replace(365243, np.nan)
    full_df["IS_PENSIONER"] = (full_df["DAYS_EMPLOYED"] == 365243).astype("int8")
    
    # 2. Financial ratios
    full_df["CREDIT_TO_INCOME"] = full_df["AMT_CREDIT"] / (full_df["AMT_INCOME_TOTAL"] + 1e-5)
    full_df["ANNUITY_TO_INCOME"] = full_df["AMT_ANNUITY"] / (full_df["AMT_INCOME_TOTAL"] + 1e-5)
    full_df["CREDIT_TERM"] = full_df["AMT_CREDIT"] / (full_df["AMT_ANNUITY"] + 1e-5)
    full_df["GOODS_TO_CREDIT"] = full_df["AMT_GOODS_PRICE"] / (full_df["AMT_CREDIT"] + 1e-5)
    full_df["INCOME_PER_FAMILY"] = full_df["AMT_INCOME_TOTAL"] / (1 + full_df["CNT_FAM_MEMBERS"].fillna(1))
    full_df["CREDIT_PER_FAMILY"] = full_df["AMT_CREDIT"] / (1 + full_df["CNT_FAM_MEMBERS"].fillna(1))
    
    goods = full_df["AMT_GOODS_PRICE"].fillna(full_df["AMT_CREDIT"])
    full_df["UNCOVERED_CREDIT"] = np.maximum(0, full_df["AMT_CREDIT"] - goods)
    full_df["UNCOVERED_CREDIT_RATIO"] = full_df["UNCOVERED_CREDIT"] / (full_df["AMT_CREDIT"] + 1e-5)
    full_df["NET_INCOME_SURPLUS"] = np.maximum(0, full_df["AMT_INCOME_TOTAL"] - full_df["AMT_ANNUITY"])
    full_df["SURPLUS_PER_FAMILY"] = full_df["NET_INCOME_SURPLUS"] / (1 + full_df["CNT_FAM_MEMBERS"].fillna(1))
    
    # 3. Demographic & normalized lifetime years
    full_df["AGE_YEARS"] = full_df["DAYS_BIRTH"] / -365.25
    full_df["EMPLOYED_YEARS"] = full_df["DAYS_EMPLOYED_CLEAN"] / -365.25
    full_df["EMPLOYED_TO_AGE"] = full_df["EMPLOYED_YEARS"] / (full_df["AGE_YEARS"] + 1e-5)
    full_df["REGISTRATION_YEARS"] = full_df["DAYS_REGISTRATION"] / -365.25
    full_df["ID_PUBLISH_YEARS"] = full_df["DAYS_ID_PUBLISH"] / -365.25
    full_df["PHONE_CHANGE_YEARS"] = full_df["DAYS_LAST_PHONE_CHANGE"] / -365.25
    
    full_df["REGISTRATION_TO_AGE"] = full_df["REGISTRATION_YEARS"] / (full_df["AGE_YEARS"] + 1e-5)
    full_df["ID_TO_AGE"] = full_df["ID_PUBLISH_YEARS"] / (full_df["AGE_YEARS"] + 1e-5)
    full_df["PHONE_TO_AGE"] = full_df["PHONE_CHANGE_YEARS"] / (full_df["AGE_YEARS"] + 1e-5)
    full_df["CAR_AGE_TO_EMP_YEARS"] = full_df["OWN_CAR_AGE"] / (full_df["EMPLOYED_YEARS"] + 1e-5)
    
    # 4. EXT_SOURCE aggregates & non-linear combinations
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    full_df["EXT_MEAN"] = full_df[ext_cols].mean(axis=1)
    full_df["EXT_STD"] = full_df[ext_cols].std(axis=1)
    full_df["EXT_MIN"] = full_df[ext_cols].min(axis=1)
    full_df["EXT_MAX"] = full_df[ext_cols].max(axis=1)
    full_df["EXT_PROD"] = full_df["EXT_SOURCE_1"].fillna(1) * full_df["EXT_SOURCE_2"].fillna(1) * full_df["EXT_SOURCE_3"].fillna(1)
    full_df["EXT_DIFF_1_2"] = full_df["EXT_SOURCE_1"] - full_df["EXT_SOURCE_2"]
    full_df["EXT_DIFF_2_3"] = full_df["EXT_SOURCE_2"] - full_df["EXT_SOURCE_3"]
    full_df["EXT_DIFF_1_3"] = full_df["EXT_SOURCE_1"] - full_df["EXT_SOURCE_3"]
    
    e1 = full_df["EXT_SOURCE_1"].fillna(0.35)
    e2 = full_df["EXT_SOURCE_2"].fillna(0.50)
    e3 = full_df["EXT_SOURCE_3"].fillna(0.50)
    full_df["EXT_HARM"] = 3.0 / (1.0 / (e1 + 1e-4) + 1.0 / (e2 + 1e-4) + 1.0 / (e3 + 1e-4))
    
    full_df["DTI_OVER_EXT_MEAN"] = full_df["ANNUITY_TO_INCOME"] / (full_df["EXT_MEAN"].fillna(0.45) + 1e-4)
    full_df["DTI_OVER_EXT2_SQ"] = full_df["ANNUITY_TO_INCOME"] / (e2**2 + 1e-4)
    full_df["AGE_X_EXT2"] = full_df["AGE_YEARS"] * e2
    full_df["AGE_X_EXT_MEAN"] = full_df["AGE_YEARS"] * full_df["EXT_MEAN"].fillna(0.45)
    full_df["SOLVENCY_CUSHION"] = full_df["NET_INCOME_SURPLUS"] * full_df["EXT_HARM"]
    
    # 5. Loan Term Clustering via Gaussian Mixture (GMM)
    term_valid = full_df[["CREDIT_TERM"]].dropna()
    gmm = GaussianMixture(n_components=3, random_state=gmm_random_state)
    gmm.fit(term_valid)
    full_df["LOAN_TERM_CLUSTER"] = gmm.predict(full_df[["CREDIT_TERM"]].fillna(term_valid.median()))
    
    # 6. Categorical encoding setup
    raw_cats = full_df.select_dtypes(include=["object", "category"]).columns.tolist()
    categorical_cols = [c for c in raw_cats if c not in ["SK_ID_CURR", "TARGET"]]
    for col in categorical_cols:
        full_df[col] = full_df[col].astype("category")
        
    train_feat = full_df[full_df["TARGET"].notnull()].reset_index(drop=True)
    test_feat = full_df[full_df["TARGET"].isnull()].reset_index(drop=True)
    
    feature_names = [c for c in train_feat.columns if c not in ["SK_ID_CURR", "TARGET"]]
    return train_feat, test_feat, feature_names, categorical_cols
