# app.py
# Developed by Bindu Bhargavi Bayaneni

from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None

    if request.method == "POST":
        # Get values from form
        gender = int(request.form["gender"])
        married = int(request.form["married"])
        education = int(request.form["education"])
        applicant_income = float(request.form["applicant_income"])
        coapplicant_income = float(request.form["coapplicant_income"])
        loan_amount = float(request.form["loan_amount"])
        loan_term = float(request.form["loan_term"])
        credit_history = int(request.form["credit_history"])
        property_area = int(request.form["property_area"])

        # Create feature array
        input_data = np.array([[gender, married, education, applicant_income,
                                coapplicant_income, loan_amount, loan_term,
                                credit_history, property_area]])

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        prediction = "Loan Approved" if pred == 1 else "Loan Rejected"
        probability = round(prob * 100, 2)

    return render_template("index.html", prediction=prediction, probability=probability)

if __name__ == "__main__":
    app.run(debug=True)
