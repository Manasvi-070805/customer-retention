from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# -----------------------------
# 1. Load the dataset
# -----------------------------
df = pd.read_csv("customer_retention_reviews.csv")

# We use simple customer/service features.
# review_text and sentiment are intentionally not used here to keep
# this beginner-friendly and avoid text processing.
features = [
    "age", "gender", "city", "plan_type", "tenure_months",
    "monthly_bill", "total_spend", "usage_hours_per_month",
    "support_tickets", "avg_response_hours", "payment_delay_days",
    "auto_renewal", "satisfaction_score"
]

X = df[features]
y = df["churn"]

categorical_features = ["gender", "city", "plan_type"]
numerical_features = [c for c in features if c not in categorical_features]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# -----------------------------
# 2. Train the model
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))


@app.route("/")
def home():
    return render_template(
        "index.html",
        accuracy=round(accuracy * 100, 2),
        total_customers=len(df),
        churned=int(df["churn"].sum())
    )


@app.route("/predict", methods=["POST"])
def predict():
    # Read values from the HTML form
    customer = pd.DataFrame([{
        "age": int(request.form["age"]),
        "gender": request.form["gender"],
        "city": request.form["city"],
        "plan_type": request.form["plan_type"],
        "tenure_months": int(request.form["tenure_months"]),
        "monthly_bill": float(request.form["monthly_bill"]),
        "total_spend": float(request.form["total_spend"]),
        "usage_hours_per_month": float(request.form["usage_hours_per_month"]),
        "support_tickets": int(request.form["support_tickets"]),
        "avg_response_hours": float(request.form["avg_response_hours"]),
        "payment_delay_days": int(request.form["payment_delay_days"]),
        "auto_renewal": int(request.form["auto_renewal"]),
        "satisfaction_score": int(request.form["satisfaction_score"])
    }])

    prediction = int(model.predict(customer)[0])
    probability = float(model.predict_proba(customer)[0][1])

    if prediction == 1:
        result = "Likely to Churn"
        message = "Consider contacting this customer with a retention offer or support follow-up."
    else:
        result = "Likely to Stay"
        message = "This customer currently appears to have a lower churn risk."

    return render_template(
        "result.html",
        result=result,
        probability=round(probability * 100, 2),
        message=message
    )


if __name__ == "__main__":
    app.run(debug=True)
