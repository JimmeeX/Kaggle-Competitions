# Week 6 AI Course

import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# Input File
data = pd.read_csv("train.csv")
print(len(data)) # Col
print(data.count()) # Rows in Each Column
#print(data.describe())

print(data["Sex"].unique()) # Print Num of different inputs
print(data["Sex"].value_counts())

# Delete
dataDelete = data.dropna().count()
#print(dataDelete)

# Fill in Values (mean/median)
x = data.fillna(data.median()).fillna(data.mode().iloc[0])
x["Cabin"] = x["Cabin"].fillna("Missing")
x = x.fillna(x.mode().iloc[0])

#print(x.count())
# Find Type
#print(data.dtypes)


# Dummification/Factorise: Male = 0
#									Female = 1
x = pd.get_dummies(x, columns = ["Pclass", "Sex", "Embarked"]) # Categorical Variables
#print(x)

# Delete whole column (no impact to the end result)
x.pop("Name")
del x["Ticket"]
x.pop("Cabin")

# x["Name"].str.split(", ").str[1].str.splot(".").str[0] #Get Mrs, Ms, etc
#(x.dtypes=="object").index[x.dtypes=="object"]

y = x.pop("Survived")

#print(x.dtypes)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# Random Forest 

from sklearn.ensemble import RandomForestClassifier as rfc #OR Regressor
model = rfc(n_estimators=1000) # Parameters
model.fit(x_train, y_train)
print(sum(model.predict(x_test) ==y_test)/len(y_test)) # Accuracy





