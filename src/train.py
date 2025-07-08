import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from src.utils import clean_text

def train_and_save_model():
    print("📦 Loading dataset from data/sms_spam.tsv...")
    df = pd.read_csv("data/sms_spam.tsv", sep="\t", header=None, names=["label", "message"])

    print("📄 Sample data:\n", df.head())

    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df['message'] = df['message'].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n✅ Accuracy:", round(accuracy, 4))
    print("📊 Classification Report:\n", classification_report(y_test, y_pred))

    with open("models/spam_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("models/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("\n📁 Model and vectorizer saved to models/")
