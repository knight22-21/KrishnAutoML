'''from krishnautoml.data.feature_engineering import FeatureEngineer
import pandas as pd

df = pd.DataFrame({
    "age": [25, 32, 40],
    "city": ["Delhi", "Mumbai", "Delhi"],
    "review": ["Good product, loved it", "Not bad but could improve", "Terrible experience, never again"],
    "target": [1, 0, 0]
})

X = df.drop(columns=["target"])
y = df["target"]

fe = FeatureEngineer()
X_transformed = fe.fit_transform(X, y)

print(X_transformed.shape)  # Should include numeric + encoded categorical + TF-IDF
'''


from krishnautoml.reporting.report_generator import ReportGenerator

metrics = {"accuracy": 0.85, "precision": 0.84, "recall": 0.83, "f1": 0.83}
plots = ["reports/evaluation/confusion_matrix.png", "reports/evaluation/roc_curve.png"]
model_info = {"name": "XGBoostClassifier", "params": {"max_depth": 5, "learning_rate": 0.1}}

reporter = ReportGenerator()
report_path = reporter.generate_report(
    project_name="Titanic_Survival",
    metrics=metrics,
    plots=plots,
    eda_report="reports/eda/eda_report.html",
    model_info=model_info
)

print("Report generated at:", report_path)
