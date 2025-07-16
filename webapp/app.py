# webapp/app.py
from flask import Flask, render_template, request
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import predict_message  
# from flask import Flask, request, render_template
# from src.predict import predict_message  # ✅ Now uses updated logic

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None

    if request.method == "POST":
        msg = request.form["message"]
        result, confidence = predict_message(msg)
    
    return render_template("index.html", result=result, confidence=confidence)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
