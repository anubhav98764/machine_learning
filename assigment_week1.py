import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load a dataset using Pandas and summarize basic stats (.info(), .describe()).
df = pd.read_csv('Titanic-Dataset.csv')
data = df.info()
desc = df.describe()
# check missing value
empty = df.isnull().sum() 

# Handle missing data using mean/median imputation
mean_age = df['Age'].mean()
df['Age_mean'] = df['Age'].fillna(mean_age)

# meadian
median_age = df['Age'].median()
df['Age_median'] = df['Age'].fillna(median_age)
print(df[['Age','Age_mean','Age_median']].head())

# Encode categorical variables using LabelEncoder & OneHotEncoder
x = LabelEncoder()
label_encoder = df['lebel_sex'] = x.fit_transform(df['Sex'])
print(label_encoder)

one_hot = pd.get_dummies(df,columns = ['Sex'],drop_first= True)
print(one_hot.head())

# Task of Mini project
# visualize data by using matplotlib
plt.figure(figsize=(8,5))
plt.hist(df['Age'],bins = 30,color = 'green',edgecolor = 'red')
plt.title("Age Distribution Graph")
plt.xlabel('Age')
plt.ylabel('No of passenger')
plt.show()

# saving a output in cleaned new dataset
df.to_csv('Titanic_cleaned.csv',index = False)