from krishnautoml import KrishnAutoML
import pandas as pd


def test_pipeline_runs(tmp_path):
    data = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": ["a", "b", "a", "b", "a"],
            "target": [0, 1, 0, 1, 0],
        }
    )
    automl = KrishnAutoML(target="target")
    automl.load_data(data)
    automl.preprocess().train_models()
    metrics = automl.evaluate()
    assert isinstance(metrics, dict)
