import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle
from sklearn.utils import class_weight

# 0 = Normal text, 1 = Spam, 2 = Promotionals, 3 = OTP

# Defining variables
oov = "<OOV>"
numWords = 5000
paddingType = "post"
truncateType = "post"
maxLength = 100
embeddingDim = 64
trainingRatio = 0.2
dataset = pd.read_csv("result-clean2.csv")

# Split data by library
trainingSentences,valSentences,trainingLabels,valLabels = train_test_split(dataset['Teks'],dataset['label'],test_size=trainingRatio,random_state=42)
trainingSentences = np.array(trainingSentences)
trainingLabels = np.array(trainingLabels)
valSentences = np.array(valSentences)
valLabels = np.array(valLabels)

class_weights = list(class_weight.compute_class_weight('balanced', np.unique(dataset['label']),dataset['label']))
class_weights.sort()
weights = {}
for index, weight in enumerate(class_weights):
    weights[index] = weight

# Initiating tokenizer
tokenizer = Tokenizer(num_words = numWords, oov_token = oov)
tokenizer.fit_on_texts(trainingSentences)
word_index = tokenizer.word_index

# Putting tokens into sequences
trainingSequences = tokenizer.texts_to_sequences(trainingSentences)
trainingPadded = np.array(pad_sequences(trainingSequences,maxlen = maxLength, padding = paddingType, truncating = truncateType))
valSequences = tokenizer.texts_to_sequences(valSentences)
valPadded = np.array(pad_sequences(valSequences,maxlen = maxLength, padding = paddingType, truncating = truncateType))

# # Defining model
# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(numWords, embeddingDim),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embeddingDim)),
#     tf.keras.layers.Dense(embeddingDim, activation='relu'),
#     tf.keras.layers.Dense(2, activation='softmax')
# ])

# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# num_epochs = 8
# history = model.fit(trainingPadded, trainingLabels, epochs=num_epochs, validation_data=(valPadded, valLabels), class_weight=weights)

#model.save("my_model2.h5")
model = tf.keras.models.load_model('my_model2.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Evaluation
evalDataset = pd.read_csv("new2.csv")
evalFeatures = np.array(evalDataset['Teks'])
evalLabels = np.array(evalDataset['label'])
evalSequences = tokenizer.texts_to_sequences(evalFeatures)
evalPadded = np.array(pad_sequences(evalSequences,maxlen=maxLength,padding=paddingType))
model.evaluate(evalPadded,evalLabels)