from __future__ import annotations
from typing import Optional
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

def detect_problem_type(y) -> str:
    """Infer problem type from target vector."""
    if y.dtype.kind in {"O"}: # strings/objects
        return "classification"


    # If numeric but few unique integer-like values → classification
    unique_vals = np.unique(y.dropna()) if hasattr(y, "dropna") else np.unique(y)
    if unique_vals.size <= 20 and np.all(np.equal(np.mod(unique_vals, 1), 0)):
        return "classification"


    return "regression"




def get_cv(problem_type: str, n_splits: int, random_state: int):
    if problem_type == "classification":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)




def get_default_scoring(problem_type: str) -> str:
    return "accuracy" if problem_type == "classification" else "neg_root_mean_squared_error"




def safe_import_xgboost(task: str, random_state: int):
    """Return configured xgboost model or None if xgboost not installed."""
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except Exception:
        return None


    common = dict(n_estimators=400, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=random_state, n_jobs=-1)
    if task == "cls":
        return XGBClassifier(**common, eval_metric="logloss")
    return XGBRegressor(**common)




def safe_import_lightgbm(task: str, random_state: int):
    """Return configured lightgbm model or None if lightgbm not installed."""
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
    except Exception:
        return None


    common = dict(n_estimators=500, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=random_state)
    if task == "cls":
        return LGBMClassifier(**common)
    return LGBMRegressor(**common)