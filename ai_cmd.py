import tensorflow as tf 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import os
import pandas as pd

oov = "<OOV>"
numWords = 5000
paddingType = "post"
truncateType = "post"
maxLength = 100
model = tf.keras.models.load_model('my_model.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Uncomment to run evaluation on test dataset
new = pd.read_csv("new.csv")
features = np.array(new['Teks'])
labels = np.array(new['label'])

sequences = tokenizer.texts_to_sequences(features)
padded = np.array(pad_sequences(sequences,maxlen=maxLength,padding=paddingType))
model.evaluate(padded,labels)

while True:
    text = input(">")
    if text == "cls":
        os.system("cls")
    elif text == "exit()":
        break
    else:
        sequences = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequences, maxlen = maxLength, padding=paddingType)
        pred = model.predict(padded)
        labels = ['Normal SMS','Spam','Promo','OTP']
        print(labels[np.argmax(pred)])
        print(pred)




