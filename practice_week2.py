# Practice set question
#  Train a logistic regression model on Titanic dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
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

# 3...
# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_acc = accuracy_score(y_test, dt.predict(X_test))

# KNN (requires scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
knn_acc = accuracy_score(y_test, knn.predict(X_test_scaled))

print("Decision Tree Accuracy:", dt_acc)
print("KNN Accuracy:", knn_acc)

# 4.....
depths = range(1, 21)
train_acc = []
test_acc = []

for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    train_acc.append(dt.score(X_train, y_train))
    test_acc.append(dt.score(X_test, y_test))





