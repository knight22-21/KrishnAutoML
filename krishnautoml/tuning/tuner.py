from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
from sklearn.model_selection import cross_val_score


from ..utils.helpers import get_cv, get_default_scoring

class Tuner:
    """
    Phase 1 tuner: train with default hyperparameters and select the model
    with the best CV mean score. Then refit best on full data.
    """


    def __init__(self, problem_type: str, n_splits: int = 5, random_state: int = 42) -> None:
        self.problem_type = problem_type
        self.n_splits = n_splits
        self.random_state = random_state


    def run(self, candidates: Dict[str, object], X: np.ndarray, y: np.ndarray) -> Tuple[Dict[str, float], object]:
        cv = get_cv(self.problem_type, self.n_splits, self.random_state)
        scoring = get_default_scoring(self.problem_type)


        scores: Dict[str, float] = {}
        best_name = None
        best_score = -np.inf
        best_model = None


        for name, model in candidates.items():
            try:
                cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
                mean_score = float(np.mean(cv_scores))
                scores[name] = mean_score
                if mean_score > best_score:
                    best_score = mean_score
                    best_name = name
                    best_model = model
            except Exception as e:
                scores[name] = float("nan")
                print(f"⚠️ Skipping {name} due to error: {e}")


        if best_model is None:
            raise RuntimeError("All candidate models failed during CV.")


        # Fit best model on full data
        best_model.fit(X, y)
        return {"cv_scores": scores, "best_model": best_name, "best_score": best_score}, best_model

