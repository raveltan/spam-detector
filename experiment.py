import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('C:/Users/yowen/Downloads/result-clean (1).csv')
x_train,x_test,y_train,y_test = train_test_split(dataset['Teks'],dataset['label'],test_size=0.2,random_state=42)

print(len(x_train),len(x_test),len(y_test),len(y_train))
