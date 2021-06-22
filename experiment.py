import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
trainingRatio = 0.2
dataset = pd.read_csv('result-clean.csv')
trainingSentences,evaluateSentences,trainingLabels,evaluateLabels = train_test_split(dataset['Teks'],dataset['label'],test_size=trainingRatio,random_state=98)
print(dataset['label'].value_counts())

class_weights = list(class_weight.compute_class_weight('balanced', np.unique(dataset['label']),dataset['label']))
class_weights.sort()
weights = {}
for index, weight in enumerate(class_weights):
    weights[index] = weight

print(weights)
