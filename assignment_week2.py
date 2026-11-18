# Practice set question
#  Train a logistic regression model on Titanic dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
df = pd.read_csv('Titanic-Dataset.csv')

df = df[["Survived","Pclass","Sex","Age","Fare"]].dropna()
df["Sex"] = df["Sex"].map({"male": 0,"female" : 1})

X = df.drop("Survived", axis = 1)
y = df["Survived"]

# Train-test split
X_train,X_test,y_train,y_test =  train_test_split(X,y,test_size= 0.2,random_state = 42)
# scaler feature
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# Logistics Regression
log_reg = LogisticRegression()
log_reg.fit(X_train_scaled , y_train)

# 2...
#  Create confusion matrix and accuracy score
y_pred = log_reg.predict(X_test_scaled)
cm = confusion_matrix(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)

print("Confusion Matrix :", cm)
print("Accuracy Matrix :",acc)
