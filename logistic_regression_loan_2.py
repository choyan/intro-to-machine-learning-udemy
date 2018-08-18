import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# sigmoid function
# p(i) = 1 / (1 + exp[-(b0+b1*x)])

credit_data = pd.read_csv("creditset.csv")
# print(credit_data.head())
# print(credit_data.describe())
# print(credit_data.corr())

features = credit_data[["income", "age", "loan"]]
targetVariable = credit_data.default

# print(targetVariable)
featureTrain, featureTest, targetTrain, targetTest = train_test_split(features, targetVariable, test_size=0.8)

model = LogisticRegression()
fitModel = model.fit(featureTrain, targetTrain)
prediction = fitModel.predict(featureTest)

print( confusion_matrix(targetTest, prediction)) #https://machinelearningmastery.com/confusion-matrix-machine-learning/
print( accuracy_score(targetTest, prediction))