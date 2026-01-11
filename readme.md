# Loan Approval Prediction System

**Developed by Bindu Bhargavi Bayaneni**

## 📌 Project Overview

This project is an end-to-end Machine Learning and Web Application system that predicts whether a loan will be **Approved or Rejected** based on applicant details.
It demonstrates the complete ML lifecycle: data preprocessing, feature selection, model training, evaluation, cross-validation, and deployment using Flask.

---

## 🔧 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Flask
- HTML, CSS, Bootstrap

---

## 📊 Dataset

A realistic **synthetic loan dataset** is generated inside `train_model.py` with the following features:

- Gender
- Married
- Education
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Amount Term
- Credit History
- Property Area
- Loan Status (Target)

---

## 🧹 Data Preprocessing

- Categorical features encoded using `LabelEncoder`
- Numerical features scaled using `StandardScaler`
- Missing values handled implicitly through generation logic
- Features normalized before model training

---

## 🎯 Feature Selection

Feature importance is computed using **Random Forest**.
Important features influencing loan approval:

1. Credit History
2. Applicant Income
3. Loan Amount
4. Property Area
5. Education

These were selected based on high importance scores from the trained ensemble model.

---

## 🤖 Models Trained

- Logistic Regression
- Decision Tree
- Random Forest

The best model is selected based on **highest accuracy** on the test set.

---

## 📈 Evaluation Metrics

For each model, the following metrics are calculated:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

A **Confusion Matrix Heatmap** is also visualized.

---

## 🔁 5-Fold Cross Validation

5-Fold Cross Validation is performed on the best model:

- Mean Accuracy
- Standard Deviation

### Why Cross Validation?

It ensures the model is not overfitting and performs consistently across different data splits.

---

## 💾 Model Saving

The trained model and scaler are saved using Joblib:

- `loan_model.pkl`
- `scaler.pkl`

---

## 🌐 Web Application (Flask)

A user-friendly web interface allows users to:

- Enter applicant details
- Predict Loan Approval
- View Approval Probability

---

## ▶ How to Run the Project

### Step 1: Install Dependencies

```bash
pip install flask pandas numpy scikit-learn matplotlib seaborn joblib
```

### Step 2: Train the Model

```bash
python train_model.py
```

### Step 3: Run the Web App

```bash
python app.py
```

### Step 4: Open in Browser

```
http://127.0.0.1:5000/
```

---

## 🗂 Project Structure

```
loan_prediction_project/
│
├── train_model.py
├── app.py
├── loan_model.pkl
├── scaler.pkl
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
└── README.md
```

---

## 🎤 Interview Explanation Summary

### Why Logistic / Random Forest?

- Logistic Regression: Interpretable baseline
- Random Forest: Handles non-linearity and feature interactions

### Why Precision, Recall, F1?

- Precision: How many approved loans were actually correct
- Recall: How many eligible customers were correctly approved
- F1: Balance between Precision & Recall

### Why Feature Scaling?

- Required for distance-based optimization and faster convergence

### Why Flask?

- Lightweight, fast, perfect for ML model deployment

---
