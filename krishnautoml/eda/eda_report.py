# krishnautoml/eda/eda_report.py

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class EDAReport:
    def __init__(self):
        pass

    def generate(self, X: pd.DataFrame, y: pd.Series, output_dir="reports/eda"):
        os.makedirs(output_dir, exist_ok=True)

        df = X.copy()
        df["__target__"] = y

        # 1. Basic summary
        summary = df.describe(include="all").transpose()
        summary.to_csv(os.path.join(output_dir, "summary.csv"))

        # 2. Missing values
        missing = df.isnull().mean().sort_values(ascending=False)
        missing.to_csv(os.path.join(output_dir, "missing_values.csv"))

        # 3. Target distribution
        plt.figure(figsize=(6,4))
        sns.histplot(y, kde=True)
        plt.title("Target Distribution")
        plt.savefig(os.path.join(output_dir, "target_distribution.png"))
        plt.close()

        # 4. Correlation heatmap (numeric only)
        plt.figure(figsize=(10,8))
        corr = df.corr(numeric_only=True)
        sns.heatmap(corr, annot=False, cmap="coolwarm")
        plt.title("Feature Correlation Heatmap")
        plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
        plt.close()

        print(f"EDA report generated in {output_dir}")
