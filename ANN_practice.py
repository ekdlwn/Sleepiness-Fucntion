import numpy as np
import pandas as pd
import os

dataset_path = 'C:\\Users\\ekdlw\\Desktop\\20.07.16\\DROZY\\Final_result_2.csv'
dataset = pd.read_csv(dataset_path)
print(dataset_path)

X_len = len(dataset.columns) - 1 
y_len = len(dataset.columns) - 1
X = dataset.iloc[:, 0:X_len]
y = dataset.iloc[:, y_len]

#%% 데이터 분리
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#%% Feature Scaling
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
one_hot_y_train = to_categorical(y_train)
one_hot_y_test = to_categorical(y_test)
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
print(y_train)

#%% 인공 신경망 만들기

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
# Initialising the ANN
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 12, activation = 'tanh', input_dim = 3))
# Adding the second hidden layer
classifier.add(Dense(units = 12, activation = 'tanh'))
# Adding the output layer
classifier.add(Dense(units = 3, activation = 'softmax'))
# Compiling the ANN
opt=optimizers.Adam(lr=0.000001)
classifier.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set
classifier.fit(X_train, one_hot_y_train, batch_size = 10, epochs =1000)



#%% 5. 모델 평가하기
loss_and_metrics = classifier.evaluate(X_test, one_hot_y_test, batch_size=32)
print('')
print('loss_and_metrics : ' + str(loss_and_metrics))
#%%

# Final_dataset.to_csv('C:\\Users\\ekdlw\\Desktop\\20.07.16\\DROZY\\Final_result.csv')


