import mlflow
from src.train import train_and_save_model
from src.predict import predict_message

# Enable MLflow autologging manually if desired
mlflow.set_experiment("Spam Detection Predictions")

print("\n🔍 Custom Predictions:")
messages = [
    "Your account has been temporarily suspended. Verify your login immediately.",
    "Don't miss our Black Friday Mega Sale – 70% off all electronics!",
    "Hey John, just checking in about the meeting agenda for tomorrow.",
    "You’ve inherited $5 million dollars. Contact our attorney to claim now.",
    "Reminder: Your Amazon package will be delivered tomorrow.",
    "URGENT: Your system has been infected. Click here to fix it now.",
    "Thanks for applying to our internship program. We'll get back to you shortly.",
    "Earn money from home with zero investment. Sign up today!",
    "Final warning! Your Netflix subscription will be canceled.",
    "Can we reschedule our dentist appointment for next week?"
]

with mlflow.start_run():
    for i, msg in enumerate(messages, 1):
        label, confidence = predict_message(msg)
        mlflow.log_metric(f"Confidence_Message_{i}", confidence)
        print(f"{i}. '{msg}' → {label} (Confidence: {confidence}%)")
