import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

trainingVectors = pd.read_csv("trainingPadded.csv")
trainingLabels = pd.read_csv("trainingLabels.csv")
evalVectors = pd.read_csv("evalPadded.csv")
evalLabels = pd.read_csv('evalLabels.csv')

trainingLabels = trainingLabels.iloc[:,1]
trainingVectors = trainingVectors.iloc[:,1:]
evalLabels = evalLabels.iloc[:,1]
evalVectors = evalVectors.iloc[:,1:]
NBModel = MultinomialNB()
NBModel.fit(trainingVectors,trainingLabels)
NBPreds = NBModel.predict(evalVectors)
print(classification_report(evalLabels,NBPreds))

