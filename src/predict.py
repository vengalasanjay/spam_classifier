import pickle
from src.utils import clean_text

def load_model_and_vectorizer():
    with open("models/spam_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def predict_message(message, threshold=0.3):
    model, vectorizer = load_model_and_vectorizer()
    cleaned = clean_text(message)
    vec = vectorizer.transform([cleaned])
    prob = model.predict_proba(vec)[0][1]
    label = "Spam 🚫" if prob >= threshold else "Not Spam ✅"
    return f"{label} (Confidence: {round(prob * 100, 2)}%)"
