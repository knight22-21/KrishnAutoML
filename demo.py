from krishnautoml import KrishnAutoML

automl = KrishnAutoML(target="Survived", problem_type="auto")

automl.load_data("data/titanic.csv") \
      .preprocess() \
      .run_eda() \
      .train_models()

automl.evaluate()
automl.save("best_model.pkl")
automl.generate_report(project_name="Titanic_Survival")
