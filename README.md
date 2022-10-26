# Machine-Learning-Project-2
##"first we have to import some  libraries"
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

##"download diabetes.csv from kaggle"
##"read the file"
df=pd.read_csv("diabetes.csv")
df.head()

##"check the null values in data"
data=sns.heatmap(df.isnull(),cmap="viridis")
##"create heatmap of it"

##"check the dependency of column on each other"
df.corr()

##"convert 0 to no diabetes and 1 to diabetes in outcome column"
df["Outcome"]=df["Outcome"].map({1:"diabetes",0:"no diabetes"})
df.head()

from sklearn.model_selection import train_test_split
x=np.array(df[["Pregnancies","Glucose","BloodPressure","Insulin","BMI","Age"]])
y=np.array(df[["Outcome"]])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=10)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))

features=np.array([[0,100,120,23,50,20]])
print(model.predict(features))
