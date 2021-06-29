import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import roc_curve,classification_report,accuracy_score
from imblearn.over_sampling import RandomOverSampler,SMOTE
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB


# 0 = Normal text, 1 = Spam, 2 = Promotionals, 3 = OTP

# Defining variables
oov = "<OOV>"
numWords = 5000
paddingType = "post"
truncateType = "post"
maxLength = 100
embeddingDim = 64
trainingRatio = 0.3
dataset = pd.read_csv("combined.csv")
rocDataset = pd.read_csv("result-clean2.csv")
oversampler = RandomOverSampler()

# Split data by library
pretrainingSentences,evalSentences,pretrainingLabels,evalLabels = train_test_split(dataset['Teks'],dataset['label'],test_size=trainingRatio,random_state=42)
trainingSentences, valSentences,trainingLabels, valLabels = train_test_split(pretrainingSentences, pretrainingLabels,test_size = 0.2,random_state = 42)

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
trainingPadded, trainingLabels = oversampler.fit_resample(trainingPadded,trainingLabels)
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
num_epochs = 10
history = model.fit(trainingPadded, trainingLabels, epochs=num_epochs, validation_data=(valPadded, valLabels),class_weight=weights)
# model = tf.keras.models.load_model('my_model.h5')

# Uncomment to run evaluation and print classification reports on test dataset
evalSequences = tokenizer.texts_to_sequences(evalSentences)
evalPadded = np.array(pad_sequences(evalSequences,maxlen=maxLength,padding=paddingType))
evalPadded, evalLabels = oversampler.fit_resample(evalPadded,evalLabels)
evalPred = model.predict(evalPadded)
evalPredNew = []
for i in evalPred:
    evalPredNew.append(np.argmax(i))
print("\nBidirectional LSTM Classification Report")
print(classification_report(evalLabels,evalPredNew))
acc = accuracy_score(evalLabels,evalPredNew)
print("Average accuracy: " + str(acc))

# Uncomment the following blocks of code to generate graph
# # Plot for accuracy
fig = plt.figure()
acc = fig.add_subplot(1,2,1)
acc.plot(history.history['accuracy'])
acc.plot(history.history['val_accuracy'])
acc.set_aspect(1.0/acc.get_data_ratio())
acc.set_title('model accuracy')
acc.set_ylabel('accuracy')
acc.set_xlabel('epoch')
acc.legend(['train', 'eval'], loc='upper left')
# # Plot for loss
loss = fig.add_subplot(1,2,2)
loss.plot(history.history['loss'])
loss.plot(history.history['val_loss'])
loss.set_aspect(1.0/loss.get_data_ratio())
loss.set_title('model loss')
loss.set_ylabel('loss')
loss.set_xlabel('epoch')
fig.show()

# # Plot for ROC
# binaryEvalDataset = pd.read_csv("new2.csv")
# binaryLSTMModel = tf.keras.models.load_model('my_model2.h5')
# binaryFeatures = np.array(binaryEvalDataset['Teks'])
# binaryLabels = np.array(binaryEvalDataset['label'])
# binarySequences = tokenizer.texts_to_sequences(binaryFeatures)
# binaryPadded = np.array(pad_sequences(binarySequences,maxlen=maxLength,padding=paddingType))
# binaryPadded, binaryLabels = oversampler.fit_resample(binaryPadded, binaryLabels)

# binaryPreds = binaryLSTMModel.predict(binaryPadded)
# LSTMPredsNew = []
# for i in binaryPreds:
#     LSTMPredsNew.append(i[1])
# LSTMFPR,LSTMTPR,LSTMThreshold = roc_curve(binaryLabels, LSTMPredsNew)

# # Defining binary Naive-Bayes classifier for ROC
# trainingVectors = pd.read_csv("trainingPadded2.csv")
# trainingLabels = pd.read_csv("trainingLabels2.csv")
# trainingLabels = trainingLabels.iloc[:,1]
# trainingVectors = trainingVectors.iloc[:,1:]

# binaryNBModel = MultinomialNB()
# binaryNBModel.fit(trainingVectors,trainingLabels)

# binaryPreds = binaryNBModel.predict_proba(binaryPadded)
# NBPredsNew = []
# for i in binaryPreds:
#     NBPredsNew.append(i[1])
# NBFPR,NBTPR,NBThreshold = roc_curve(binaryLabels,NBPredsNew)

# plt.subplot(1,3,3)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(LSTMFPR, LSTMTPR, label='LSTM')
# plt.plot(NBFPR, NBTPR, label='Naive-Bayes')
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve')
# plt.legend(loc='best')
plt.show()

# Below are the code used to save the model trained from running this code
# In order for the AI to be exported into a saved model, the code below can be uncommented

# with open('tokenizer.pickle', 'wb') as handle:
#     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
if acc > 0.93:
    model.save("my_model.h5")

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


