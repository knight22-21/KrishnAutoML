from krishnautoml.data.feature_engineering import FeatureEngineer
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
