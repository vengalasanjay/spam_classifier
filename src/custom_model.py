import mlflow.pyfunc

class SpamClassifierWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer

    def predict(self, context, model_input):
        vec = self.vectorizer.transform(model_input)
        return self.model.predict_proba(vec)
