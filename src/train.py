# src/train.py

import pandas as pd
import mlflow
import mlflow.pyfunc

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

from src.utils import clean_text
from src.custom_model import SpamClassifierWrapper  # 👈 Wrapper class

def train_and_log_model():
    # 🔗 Setup MLflow tracking
    mlflow.set_tracking_uri("sqlite:///mlruns_store/mlflow.db")
    mlflow.set_experiment("Spam Detection Training")

    # 📥 Load data
    df = pd.read_csv("data/sms_spam.tsv", sep="\t", header=None, names=["label", "message"])
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df['message'] = df['message'].apply(clean_text)

    # ✂️ Split data
    X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

    # 🔡 Vectorize
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 🧠 Train
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # 🔍 Evaluate
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print("✅ Accuracy:", round(acc, 4))
    print("📊 Classification Report:\n", classification_report(y_test, y_pred))

    # 📦 Log to MLflow
    with mlflow.start_run() as run:
        mlflow.log_param("model", "MultinomialNB")
        mlflow.log_param("vectorizer", "CountVectorizer(ngram_range=(1,2), stop_words='english')")
        mlflow.log_metric("accuracy", acc)

        # 🧊 Log model with custom wrapper
        mlflow.pyfunc.log_model(
            artifact_path="spam_model",
            python_model=SpamClassifierWrapper(model, vectorizer),
            registered_model_name="SpamClassifierModel"
        )

        print(f"🚀 Model registered to MLflow: SpamClassifierModel (Run ID: {run.info.run_id})")

if __name__ == "__main__":
    train_and_log_model()
