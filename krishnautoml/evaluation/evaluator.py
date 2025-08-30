from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

class Evaluator:
    def __init__(self, problem_type: str) -> None:
        self.problem_type = problem_type


    def evaluate(self, model, X, y, metrics: Optional[List[str]] = None) -> Dict[str, float]:
        if self.problem_type == "classification":
            return self._eval_classification(model, X, y, metrics)
        return self._eval_regression(model, X, y, metrics)
    
    # Classification metrics
    def _eval_classification(self, model, X, y, metrics: Optional[List[str]]) -> Dict[str, float]:
        y_pred = model.predict(X)
        out: Dict[str, float] = {}


        chosen = set(m.lower() for m in metrics) if metrics else {"accuracy", "f1", "roc_auc"}
        if "accuracy" in chosen:
            out["accuracy"] = float(accuracy_score(y, y_pred))
        if "f1" in chosen:
            # macro handles multiclass fairly
            out["f1_macro"] = float(f1_score(y, y_pred, average="macro"))
        if "roc_auc" in chosen:
            try:
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X)
                    if proba.shape[1] == 2:
                        out["roc_auc"] = float(roc_auc_score(y, proba[:, 1]))
                    else:
                        out["roc_auc_ovr"] = float(roc_auc_score(y, proba, multi_class="ovr"))
                elif hasattr(model, "decision_function"):
                    score = model.decision_function(X)
                    if score.ndim == 1:
                        out["roc_auc"] = float(roc_auc_score(y, score))
                    else:
                        out["roc_auc_ovr"] = float(roc_auc_score(y, score, multi_class="ovr"))
            except Exception as e:
                out["roc_auc"] = float("nan")
                print(f"⚠️ ROC-AUC computation failed: {e}")
        return out
    
    
    # Regression metrics
    def _eval_regression(self, model, X, y, metrics: Optional[List[str]]) -> Dict[str, float]:
        y_pred = model.predict(X)
        chosen = set(m.lower() for m in metrics) if metrics else {"r2", "rmse", "mae"}
        out: Dict[str, float] = {}
        if "r2" in chosen:
            out["r2"] = float(r2_score(y, y_pred))
        if "rmse" in chosen:
            out["rmse"] = float(np.sqrt(mean_squared_error(y, y_pred)))
        if "mae" in chosen:
            out["mae"] = float(mean_absolute_error(y, y_pred))
        return out
    
    