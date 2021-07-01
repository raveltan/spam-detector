from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler 
from imblearn.pipeline import Pipeline
from matplotlib import pyplot
from sklearn.naive_bayes import MultinomialNB
from numpy import where
import pandas as pd
import nltk                       # the natural langauage toolkit, open-source NLP
import pandas as pd               # pandas dataframe
import re                         # regular expression
from nltk.corpus import stopwords  
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

df_train=pd.read_csv('combined.csv')
df_train.info()
X_train, X_test, y_train, y_test = train_test_split(df_train['Teks'], df_train['label'], 
                                                    test_size=0.30, stratify=df_train['label'],random_state = 42)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print ('Dimension of TF-IDF vector :' , X_train_tfidf.shape)


clf = MultinomialNB().fit(X_train_tfidf, y_train)

X_new_counts = count_vect.transform(X_test)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)


sm = SMOTE()
sm_xtrain_tfidf, sm_train_y = sm.fit_resample(X_new_tfidf, y_test)
ros = RandomOverSampler()
ros_xtrain_tfidf, ros_train_y = ros.fit_resample(X_new_tfidf, y_test)

print('Original dataset shape %s' % Counter(y_train))
print('Resampled dataset shape %s' % Counter(ros_train_y))



predicted = clf.predict(sm_xtrain_tfidf)
counter  = 0

for doc, category in zip(X_test, predicted):
    print('%r => %s' % (doc, category))
    if(counter == 10):
        break
    counter += 1   
    
print(classification_report(sm_train_y,predicted))