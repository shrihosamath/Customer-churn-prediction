from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    tenure = float(request.form["tenure"])
    monthly = float(request.form["monthly"])
    total = float(request.form["total"])

    features = np.array([[tenure, monthly, total]])
    scaled = scaler.transform(features)

    prediction = model.predict(scaled)[0]

    result = "❌ Customer WILL Churn" if prediction == 1 else "✅ Customer will NOT Churn"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
