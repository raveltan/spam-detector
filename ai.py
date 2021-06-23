import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import class_weight
from sklearn.metrics import roc_curve,classification_report
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


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
rocDataset = pd.read_csv("result-clean2.csv")

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

# Defining model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(numWords, embeddingDim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embeddingDim)),
    tf.keras.layers.Dense(embeddingDim, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
num_epochs = 6
history = model.fit(trainingPadded, trainingLabels, epochs=num_epochs, validation_data=(valPadded, valLabels), class_weight=weights)

# Uncomment to run evaluation and print classification reports on test dataset
evalDataset = pd.read_csv("new.csv")
evalFeatures = np.array(evalDataset['Teks'])
evalLabels = np.array(evalDataset['label'])
evalSequences = tokenizer.texts_to_sequences(evalFeatures)
evalPadded = np.array(pad_sequences(evalSequences,maxlen=maxLength,padding=paddingType))
model.evaluate(evalPadded,evalLabels)
evalPred = model.predict(evalPadded)
evalPredNew = []
for i in evalPred:
    evalPredNew.append(np.argmax(i))
print(classification_report(evalLabels,evalPredNew))

# Uncomment the following blocks of code to generate graph
# # Plot for accuracy
plt.subplot(1,3,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'eval'], loc='upper left')

# # Plot for loss
plt.subplot(1,3,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()

# # Plot for ROC
rocEvalDataset = pd.read_csv("new2.csv")
rocLSTMModel = tf.keras.models.load_model('my_model2.h5')
rocFeatures = np.array(rocEvalDataset['Teks'])
rocLabels = np.array(rocEvalDataset['label'])
rocLSTMSequences = tokenizer.texts_to_sequences(rocFeatures)
rocLSTMPadded = np.array(pad_sequences(rocLSTMSequences,maxlen=maxLength,padding=paddingType))
rocLSTMPreds = rocLSTMModel.predict(rocLSTMPadded,verbose = 1)
LSTMPredsNew = []
for i in rocLSTMPreds:
    LSTMPredsNew.append(i[1])
LSTMFPR,LSTMTPR,LSTMThreshold = roc_curve(rocLabels, LSTMPredsNew)

# Defining Naive-Bayes classifier
rocNBModel = make_pipeline(TfidfVectorizer(),MultinomialNB())
rocNBModel.fit(rocDataset['Teks'],rocDataset['label'])
rocNBPreds = rocNBModel.predict_proba(rocFeatures)
NBPredsNew = []
for i in rocNBPreds:
    NBPredsNew.append(i[1])
NBFPR,NBTPR,NBThreshold = roc_curve(rocLabels, NBPredsNew)

plt.subplot(1,3,3)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(LSTMFPR, LSTMTPR, label='LSTM')
plt.plot(NBFPR, NBTPR, label='Naive-Bayes')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

# Below are the code used to save the model trained from running this code
# In order for the AI to be exported into a saved model, the code below can be uncommented

# with open('tokenizer.pickle', 'wb') as handle:
#     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
#model.save("my_model.h5")

# while True:
#     testA = input(">")
#     test = []
#     test.append(testA)
#     sequences = tokenizer.texts_to_sequences(test)
#     padded = pad_sequences(sequences, maxlen = maxLength, padding=paddingType)
#     pred = model.predict(padded)
#     labels = ['Normal SMS','Spam','Promo','OTP']
#     print(labels[np.argmax(pred)])
#     print(pred)


