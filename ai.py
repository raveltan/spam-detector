import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import random

# 0 = Normal text, 1 = Spam, 2 = Promotionals, 3 = OTP

# Defining variables
oov = "<OOV>"
numWords = 5000
paddingType = "post"
truncateType = "post"
maxLength = 100
embeddingDim = 64
trainingRatio = 0.8
dataset = pd.read_csv("result-clean.csv")

# Splitting dataset into training and evaluating
trainSize = int(len(dataset) * trainingRatio)
trainingIndexes = [i for i in range(len(dataset))]
evaluateIndexes = []
trainingSentences = []
trainingLabels = []
evaluateSentences = []
evaluateLabels = []


for i in range(len(dataset)-trainSize):
    index = random.randint(0,len(trainingIndexes)-1)
    trainingIndexes.pop(index)
    evaluateIndexes.append(index)

for i in trainingIndexes:
    trainingSentences.append(dataset["Teks"][i])
    trainingLabels.append(dataset["label"][i])

for i in evaluateIndexes:
    evaluateSentences.append(dataset["Teks"][i])
    evaluateLabels.append(dataset["label"][i])

trainingSentences = np.array(trainingSentences)
trainingLabels = np.array(trainingLabels)
evaluateSentences = np.array(evaluateSentences)
evaluateLabels = np.array(evaluateLabels)
"""
trainingSentences = np.array(dataset["Teks"][0:trainSize])
trainingLabels = np.array(dataset["label"][0:trainSize])
evaluateSentences = np.array(dataset["Teks"][trainSize:])
evaluateLabels = np.array(dataset["label"][trainSize:])"""

# Initiating tokenizer
tokenizer = Tokenizer(num_words = numWords, oov_token = oov)
tokenizer.fit_on_texts(trainingSentences)
word_index = tokenizer.word_index

# Putting tokens into sequences
sequences = tokenizer.texts_to_sequences(trainingSentences)
padded = np.array(pad_sequences(sequences,maxlen = maxLength, padding = paddingType, truncating = truncateType))
evaluateSequences = tokenizer.texts_to_sequences(evaluateSentences)
evaluatePadded = np.array(pad_sequences(evaluateSequences,maxlen = maxLength, padding = paddingType, truncating = truncateType))

# Defining model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(numWords, embeddingDim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embeddingDim)),
    tf.keras.layers.Dense(embeddingDim, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
num_epochs = 6
model.fit(padded, trainingLabels, epochs=num_epochs, validation_data=(evaluatePadded, evaluateLabels))

# Below are the code used to save the model trained from running this code
# In order for the AI to be exported into a saved model, the code below can be uncommented

# with open('tokenizer.pickle', 'wb') as handle:
   #  pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
# model.save("my_model.h5")

while True:
    testA = input(">")
    test = []
    test.append(testA)
    sequences = tokenizer.texts_to_sequences(test)
    padded = pad_sequences(sequences, maxlen = maxLength, padding=paddingType)
    pred = model.predict(padded)
    labels = ['Normal SMS','Spam','Promo','OTP']
    print(labels[np.argmax(pred)])


