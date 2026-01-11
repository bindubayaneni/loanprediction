# train_model.py
# Developed by Bindu Bhargavi Bayaneni

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# 1. Generate Realistic Loan Dataset
np.random.seed(42)
n = 2000

gender = np.random.choice(["Male", "Female"], n)
married = np.random.choice(["Yes", "No"], n)
education = np.random.choice(["Graduate", "Not Graduate"], n)
app_income = np.random.randint(2000, 50000, n)
co_income = np.random.randint(0, 20000, n)
loan_amount = np.random.randint(50, 700, n)
loan_term = np.random.choice([120, 180, 240, 300, 360], n)
credit_history = np.random.choice([0, 1], n, p=[0.25, 0.75])
property_area = np.random.choice(["Urban", "Semiurban", "Rural"], n)

# Rule-based approval logic
loan_status = []
for i in range(n):
    if credit_history[i] == 1 and app_income[i] > 6000 and loan_amount[i] < 400:
        loan_status.append(1)
    elif credit_history[i] == 1 and property_area[i] in ["Urban", "Semiurban"] and app_income[i] > 8000:
        loan_status.append(1)
    else:
        loan_status.append(0)

data = pd.DataFrame({
    "Gender": gender,
    "Married": married,
    "Education": education,
    "ApplicantIncome": app_income,
    "CoapplicantIncome": co_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_term,
    "Credit_History": credit_history,
    "Property_Area": property_area,
    "Loan_Status": loan_status
})

print("\nDataset Sample:")
print(data.head())


# 2. Preprocessing

le = LabelEncoder()
cat_cols = ["Gender", "Married", "Education", "Property_Area"]
for col in cat_cols:
    data[col] = le.fit_transform(data[col])

X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 3. Feature Importance

rf_temp = RandomForestClassifier(n_estimators=200, random_state=42)
rf_temp.fit(X, y)

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_temp.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:")
print(importance_df)


# 4. Train Test Split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# 5. Model Training

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=6),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=10),
    "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, eval_metric='logloss')
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[name] = acc

    print(f"\n{name} Performance:")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))


# 6. Best Model Selection

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print("\nBest Model Selected:", best_model_name)

# 7. Confusion Matrix Heatmap

cm = confusion_matrix(y_test, best_model.predict(X_test))

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# 8. 5-Fold Cross Validation

cv_scores = cross_val_score(best_model, X_scaled, y, cv=5)

print("\n5-Fold Cross Validation Results:")
print("Mean Accuracy:", cv_scores.mean())
print("Standard Deviation:", cv_scores.std())


# 9. Save Model
 
joblib.dump(best_model, "loan_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel saved as loan_model.pkl")
print("Scaler saved as scaler.pkl")
