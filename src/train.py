"""
Training Module for Home Credit Default Risk.
Trains LightGBM, CatBoost, and XGBoost using Stratified K-Fold Cross-Validation.
Evaluates individual models and generates out-of-fold predictions.
"""

from pathlib import Path
import sys
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

try:
    from .feature_engineering import engineer_features
except ImportError:
    from feature_engineering import engineer_features

def to_power_rank(preds: np.ndarray, p: float = 3.5) -> np.ndarray:
    """Apply power rank transformation to align probability distributions."""
    ranked = rankdata(preds) / len(preds)
    return ranked ** p

def train_cv(
    data_dir: Path,
    output_dir: Path,
    n_splits: int = 5,
    seed: int = 42
):
    print(f"Loading data from {data_dir}...")
    train_raw = pd.read_csv(data_dir / "train.csv")
    test_raw = pd.read_csv(data_dir / "test.csv")
    
    print("Running feature engineering pipeline...")
    train_df, test_df, feature_names, cat_cols = engineer_features(train_raw, test_raw, gmm_random_state=seed)
    
    X = train_df[feature_names]
    y = train_df["TARGET"].astype(int)
    X_test = test_df[feature_names]
    
    print(f"Dataset shape: {X.shape}, Features engineered: {len(feature_names)}")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    lgb_oof = np.zeros(len(train_df))
    lgb_test = np.zeros(len(test_df))
    
    cat_oof = np.zeros(len(train_df))
    cat_test = np.zeros(len(test_df))
    
    xgb_oof = np.zeros(len(train_df))
    xgb_test = np.zeros(len(test_df))
    
    X_cat = X.copy()
    X_test_cat = X_test.copy()
    for col in cat_cols:
        X_cat[col] = X_cat[col].astype(str).fillna("MISSING")
        X_test_cat[col] = X_test_cat[col].astype(str).fillna("MISSING")
        
    print(f"\n--- Starting {n_splits}-Fold Cross-Validation ---")
    start_time = time.perf_counter()
    
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n>>> Fold {fold} / {n_splits} <<<")
        X_tr, y_tr = X.iloc[trn_idx], y.iloc[trn_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
        model_lgb = lgb.LGBMClassifier(
            n_estimators=1500,
            learning_rate=0.025,
            num_leaves=62,
            max_depth=4,
            min_child_samples=47,
            colsample_bytree=0.69,
            subsample=0.81,
            reg_alpha=2.36,
            reg_lambda=2.32,
            random_state=seed + fold,
            n_jobs=-1,
            verbosity=-1
        )
        model_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(80, verbose=False)])
        lgb_oof[val_idx] = model_lgb.predict_proba(X_va)[:, 1]
        lgb_test += model_lgb.predict_proba(X_test)[:, 1] / n_splits
        print(f"  Fold {fold} LightGBM AUC: {roc_auc_score(y_va, lgb_oof[val_idx]):.5f}")
        
        model_cat = CatBoostClassifier(
            iterations=1200,
            learning_rate=0.035,
            depth=5,
            eval_metric="AUC",
            cat_features=cat_cols,
            random_seed=seed + fold,
            thread_count=-1,
            verbose=False,
            allow_writing_files=False
        )
        model_cat.fit(X_cat.iloc[trn_idx], y_tr, eval_set=(X_cat.iloc[val_idx], y_va), early_stopping_rounds=80)
        cat_oof[val_idx] = model_cat.predict_proba(X_cat.iloc[val_idx])[:, 1]
        cat_test += model_cat.predict_proba(X_test_cat)[:, 1] / n_splits
        print(f"  Fold {fold} CatBoost AUC: {roc_auc_score(y_va, cat_oof[val_idx]):.5f}")
        
        model_xgb = xgb.XGBClassifier(
            n_estimators=1200,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=2.0,
            reg_lambda=3.0,
            random_state=seed + fold,
            eval_metric="auc",
            tree_method="hist",
            early_stopping_rounds=80,
            enable_categorical=True,
            n_jobs=-1
        )
        model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        xgb_oof[val_idx] = model_xgb.predict_proba(X_va)[:, 1]
        xgb_test += model_xgb.predict_proba(X_test)[:, 1] / n_splits
        print(f"  Fold {fold} XGBoost AUC:  {roc_auc_score(y_va, xgb_oof[val_idx]):.5f}")

    elapsed = time.perf_counter() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s")
    
    auc_lgb = roc_auc_score(y, lgb_oof)
    auc_cat = roc_auc_score(y, cat_oof)
    auc_xgb = roc_auc_score(y, xgb_oof)
    
    print("\n" + "="*50)
    print("INDIVIDUAL POOLED OOF EVALUATION")
    print("="*50)
    print(f"LightGBM OOF ROC-AUC: {auc_lgb:.5f}")
    print(f"CatBoost OOF ROC-AUC: {auc_cat:.5f}")
    print(f"XGBoost  OOF ROC-AUC: {auc_xgb:.5f}")
    
    r_lgb_oof = to_power_rank(lgb_oof, p=3.5)
    r_cat_oof = to_power_rank(cat_oof, p=3.5)
    r_xgb_oof = to_power_rank(xgb_oof, p=3.5)
    
    oof_blend = 0.55 * r_lgb_oof + 0.35 * r_cat_oof + 0.10 * r_xgb_oof
    blend_auc = roc_auc_score(y, oof_blend)
    print(f"\nTri-Family Power Rank Ensemble OOF ROC-AUC: {blend_auc:.5f}")
    
    test_lgb_p = to_power_rank(lgb_test, p=3.5)
    test_cat_p = to_power_rank(cat_test, p=3.5)
    test_xgb_p = to_power_rank(xgb_test, p=3.5)
    
    final_test = 0.55 * test_lgb_p + 0.35 * test_cat_p + 0.10 * test_xgb_p
    final_test = (final_test - final_test.min()) / (final_test.max() - final_test.min())
    
    output_dir.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({"SK_ID_CURR": test_raw["SK_ID_CURR"], "TARGET": final_test})
    sub_path = output_dir / "submission_reproduced.csv"
    sub.to_csv(sub_path, index=False)
    print(f"\nSaved submission to {sub_path} ({len(sub)} rows)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("input"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    train_cv(args.data_dir, args.output_dir, seed=args.seed)
