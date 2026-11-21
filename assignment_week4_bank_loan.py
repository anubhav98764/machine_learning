# Option 2: Bank Loan Approval Prediction

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


loan = pd.read_csv("train_u6lujuX_CVtuZ9i (1).csv")

# Remove the identifier and convert approval labels to numeric values.
loan = loan.drop(columns=["Loan_ID"])
loan["Loan_Status"] = loan["Loan_Status"].map({"N": 0, "Y": 1})
loan = loan.dropna(subset=["Loan_Status"])

y = loan["Loan_Status"]
X = loan.drop(columns=["Loan_Status"])

numeric_features = X.select_dtypes(include="number").columns.tolist()
categorical_features = X.select_dtypes(exclude="number").columns.tolist()

# Clean missing values and convert categorical values to numeric columns.
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

random_forest = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )),
])
random_forest.fit(X_train, y_train)

predictions = random_forest.predict(X_test)
print("Random Forest Accuracy:",
      accuracy_score(y_test, predictions))

cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(cm)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Rejected", "Approved"],
).plot(cmap="Blues")
plt.title("Bank Loan Approval - Random Forest")
plt.tight_layout()
plt.show()

# Predict whether a new applicant should be approved.
new_applicant = pd.DataFrame([{
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 1500,
    "LoanAmount": 150,
    "Loan_Amount_Term": 360,
    "Credit_History": 1,
    "Property_Area": "Urban",
}])

approval = random_forest.predict(new_applicant)[0]
approval_probability = random_forest.predict_proba(new_applicant)[0, 1]
print("\nNew applicant prediction:",
      "Approved" if approval == 1 else "Rejected")
print("Approval probability:",
      f"{approval_probability:.2%}")
