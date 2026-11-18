# Assignment section------
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score


# Load housing dataset
housing = pd.read_csv("housing.csv")   

# Separate features and target
X_housing = housing.drop("price", axis=1)
y_housing = housing["price"]

# Train-test split
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_housing, y_housing, test_size=0.2, random_state=42
)

# Train Linear Regression model
lr_housing = LinearRegression()
lr_housing.fit(X_train_h, y_train_h)

# Predict prices
y_pred_h = lr_housing.predict(X_test_h)

# Evaluate R² score
print("Housing Dataset - R² Score:", r2_score(y_test_h, y_pred_h))

# Plot predicted vs actual values
plt.scatter(y_test_h, y_pred_h, alpha=0.6)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Housing Dataset: Predicted vs Actual Price")
plt.show()


# Load Titanic dataset
titanic = pd.read_csv("titanic.csv")

# Basic preprocessing
titanic["Sex"] = titanic["Sex"].map({"male": 0, "female": 1})
titanic["Age"].fillna(titanic["Age"].median(), inplace=True)
titanic["Embarked"].fillna("S", inplace=True)

# Convert categorical column 'Embarked' into dummy variables
titanic = pd.get_dummies(titanic, columns=["Embarked"])

# Select features
X_titanic = titanic[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
                     "Embarked_C", "Embarked_Q", "Embarked_S"]]
y_titanic = titanic["Survived"]

# Train-test split
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
    X_titanic, y_titanic, test_size=0.2, random_state=42
)

# Train Logistic Regression model
log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train_t, y_train_t)

# Predict survival
y_pred_t = log_model.predict(X_test_t)

# Evaluate accuracy
print("Titanic Dataset - Accuracy:", accuracy_score(y_test_t, y_pred_t))



# MINI PROJECT 2: HOUSE PRICE PREDICTION (Kaggle Dataset)


# Load Kaggle House Prices dataset
kaggle = pd.read_csv("train.csv")

# Handle missing values (simple approach)
kaggle.fillna(kaggle.median(numeric_only=True), inplace=True)

# Convert categorical columns to numeric using one-hot encoding
kaggle = pd.get_dummies(kaggle)

# Separate features and target
X_kaggle = kaggle.drop("SalePrice", axis=1)
y_kaggle = kaggle["SalePrice"]

# Train-test split
X_train_k, X_test_k, y_train_k, y_test_k = train_test_split(
    X_kaggle, y_kaggle, test_size=0.2, random_state=42
)

# Train Linear Regression model
lr_kaggle = LinearRegression()
lr_kaggle.fit(X_train_k, y_train_k)

# Predict house prices
y_pred_k = lr_kaggle.predict(X_test_k)

# Evaluate R² score
print("Kaggle House Prices - R² Score:", r2_score(y_test_k, y_pred_k))

# Plot predicted vs actual values
plt.scatter(y_test_k, y_pred_k, alpha=0.6)
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")
plt.title("Kaggle House Prices: Predicted vs Actual")
plt.show()
