import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# 0 = Normal text, 1 = Spam, 2 = Promotionals, 3 = OTP

# Defining variables
oov = "<OOV>"
numWords = 5000
paddingType = "post"
truncateType = "post"
maxLength = 100
embeddingDim = 64
trainingRatio = 0.2
dataset = pd.read_csv("result-clean.csv")


# Split data by library
trainingSentences,evaluateSentences,trainingLabels,evaluateLabels = train_test_split(dataset['Teks'],dataset['label'],test_size=trainingRatio,random_state=42)
trainingSentences = np.array(trainingSentences)
trainingLabels = np.array(trainingLabels)
evaluateSentences = np.array(evaluateSentences)
evaluateLabels = np.array(evaluateLabels)

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
history = model.fit(padded, trainingLabels, epochs=num_epochs, validation_data=(evaluatePadded, evaluateLabels), class_weight=weights)

# Uncomment the following 2 blocks of code to generate graph
# # Plot for accuracy
# plt.subplot(1,2,1)
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'eval'], loc='upper left')

# # Plot for loss
# plt.subplot(1,2,2)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend()
# plt.show()

# Below are the code used to save the model trained from running this code
# In order for the AI to be exported into a saved model, the code below can be uncommented

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
model.save("my_model.h5")

while True:
    testA = input(">")
    test = []
    test.append(testA)
    sequences = tokenizer.texts_to_sequences(test)
    padded = pad_sequences(sequences, maxlen = maxLength, padding=paddingType)
    pred = model.predict(padded)
    labels = ['Normal SMS','Spam','Promo','OTP']
    print(labels[np.argmax(pred)])
    print(pred)


