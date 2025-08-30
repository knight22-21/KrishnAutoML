from krishnautoml import KrishnAutoML

# All the steps leading up to evaluation and saving
automl = KrishnAutoML(target="Survived", problem_type="auto")
automl.load_data("data/titanic.csv") \
      .preprocess() \
      .run_eda() \
      .train_models()

# Now, call the evaluation and save methods as separate steps
automl.evaluate()
automl.save("best_model.pkl")