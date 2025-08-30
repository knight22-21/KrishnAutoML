import pandas as pd
from typing import Optional, Dict, Any, List, Tuple


from .data.loader import DataLoader
from .data.preprocessing import Preprocessor
from .models.model_factory import ModelFactory
from .tuning.tuner import Tuner
from .evaluation.evaluator import Evaluator
from .eda.eda_report import EDAReport
from .utils.helpers import detect_problem_type
from joblib import dump



class KrishnAutoML:
    """
    Main entry point for the KrishnAutoML pipeline.
    Automates: loading, preprocessing, model selection, evaluation, saving.
    """


    def __init__(
        self,
        target: str,
        problem_type: str = "auto",
        generate_eda: bool = False, # reserved for Phase 2
        random_state: int = 42,
        n_splits: int = 5,
    ) -> None:
        self.target = target
        self.problem_type = problem_type
        self.generate_eda = generate_eda
        self.random_state = random_state
        self.n_splits = n_splits


        self._loader = DataLoader()
        self._preproc = Preprocessor()


        self.X = None
        self.y = None
        self.best_model = None
        self.results: Dict[str, Any] = {}


    # Fluent API
    def load_data(self, data: pd.DataFrame | str) -> "KrishnAutoML":
        """Accepts a CSV file path or pandas DataFrame; splits X, y."""
        self.X, self.y = self._loader.load(data, self.target)
        if self.problem_type == "auto":
            self.problem_type = detect_problem_type(self.y)
        return self


    def preprocess(self) -> "KrishnAutoML":
        """Fit the preprocessing pipeline and transform X."""
        self.X = self._preproc.fit_transform(self.X, self.y)
        return self


    def train_models(self, models: Optional[List[str]] = None) -> "KrishnAutoML":
        """Train multiple candidate models (defaults vary by problem type)."""
        factory = ModelFactory(problem_type=self.problem_type, random_state=self.random_state)
        candidates = factory.get_models(models)


        tuner = Tuner(cv=self.n_splits, random_state=self.random_state)
        self.results, self.best_model = tuner.run(candidates, self.X, self.y, problem_type=self.problem_type)
        return self


    def evaluate(self, metrics: Optional[List[str]] = None) -> Dict[str, float]:
        evaluator = Evaluator(problem_type=self.problem_type)
        scores = evaluator.evaluate(self.best_model, self.X, self.y, metrics=metrics)
        self.results["evaluation"] = scores
        return scores


    def save(self, path: str = "best_model.pkl") -> None:
        dump(self.best_model, path)
        print(f"Model saved at {path}")
        
    
    def run_eda(self) -> "KrishnAutoML":
        """Generates an EDA report."""
        if self._loader.X_orig is None or self._loader.y_orig is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        eda_report = EDAReport()
        eda_report.generate(self._loader.X_orig, self._loader.y_orig)
        
        return self