from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd 

app = Flask(__name__)

# Load trained model and scaler
with open("heart_disease_bundle.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
scaler = data["scaler"]

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    probability = None

    if request.method == "POST":
        features = [
            float(request.form["age"]),
            int(request.form["sex"]),
            int(request.form["chest_pain"]),
            float(request.form["bp"]),
            float(request.form["chol"]),
            int(request.form["fbs"]),
            int(request.form["ekg"]),
            float(request.form["max_hr"]),
            int(request.form["ex_angina"]),
            float(request.form["st_dep"]),
            int(request.form["slope"]),
            int(request.form["vessels"]),
            int(request.form["thallium"])
        ]

        features = np.array([features])
        features_scaled = scaler.transform(features)

        pred = model.predict(features_scaled)[0]
        prob = model.predict_proba(features_scaled)[0]

        if pred == 1:
            result = "❤️ Heart Disease PRESENT"
            probability = f"{prob[1]*100:.2f}%"
        else:
            result = "💚 Heart Disease ABSENT"
            probability = f"{prob[0]*100:.2f}%"

    return render_template(
        "index.html",
        result=result,
        probability=probability
    )

if __name__ == "__main__":
    app.run(debug=True)
