import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import BatchNormalization
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


xy=np.loadtxt('C:\\Users\\ekdlw\\Desktop\\20.07.16\\DROZY\\Final_raw_result.csv',delimiter=',',dtype=np.float32)
x_data=xy[:,0:-1]
y_data=xy[:,[-1]]

nb_classes=10


x=np.array([[[1],[2],[3],[4],[5]]])
print(x)

print("X_train 크기: {}".format(x.shape))
Y_one_hot = tf.one_hot(x, nb_classes)  # one hot
print("one_hot:", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
print("reshape one_hot:", Y_one_hot)
xx=Y_one_hot[1,:]
print(xx)
#%%
'''
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,test_size=0.2, random_state=90)

Y_one_hot = tf.one_hot(y_train, nb_classes)  # one hot
print("one_hot:", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
train_onehot=Y_one_hot
print("reshape one_hot:", train_onehot)


print("xy 크기: {}".format(xy.shape))
print("X_train 크기: {}".format(x_train.shape))
print("y_train 크기: {}".format(y_train.shape))
print("X_test 크기: {}".format(x_test.shape))
print("y_test 크기: {}".format(y_test.shape))
print("x_train", x_train)
print("y_train", y_train)
print("y_test", y_test)


train_x=x_train
dataframe=pd.DataFrame(train_x)
dataframe.to_csv("train_x.csv")
train_y=y_train
dataframe=pd.DataFrame(train_y)
dataframe.to_csv("train_y.csv")
test_x=x_test
dataframe=pd.DataFrame(test_x)
dataframe.to_csv("test_x.csv")
test_y=y_test
dataframe=pd.DataFrame(test_y)
dataframe.to_csv("test_y.csv")

# log 풀고 오차 계산하는 함수 정의
def my_metric_fn(y_true, y_pred):
    squared_difference = tf.abs(pow(10,y_true) - pow(10,y_pred))/pow(10,y_true) * 100
    return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`

#train_start
print(x_train)
#print(x_train.shape)
print(y_train)

#ann model
model = tf.keras.Sequential([
    #tf.keras.layers.Dense(20, activation='sigmoid',use_bias=True,kernel_initializer='random_normal',bias_initializer='random_normal'),  
    tf.keras.layers.Dense(10, activation='tanh',use_bias=True,kernel_initializer='random_normal',bias_initializer='random_normal'),
    #tf.keras.layers.Dense(10, activation='softmax',use_bias=True,kernel_initializer='random_normal',bias_initializer='random_normal'),
    #tf.keras.layers.Dropout(0.2), #과적합 방지
    #tf.keras.layers.Dense(20,activation='sigmoid',use_bias=True, kernel_initializer='VarianceScaling',bias_initializer='VarianceScaling'),
    #tf.keras.layers.Dense(20, activation='softsign',use_bias=True, kernel_initializer='random_normal',bias_initializer='random_normal'),
    #tf.keras.layers.Dropout(0.2), #과적합 방지
    tf.keras.layers.Dense(10, activation='softmax', use_bias=True)
])

adams = keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=adams,loss='categorical_crossentropy')
#model.compile(optimizer='adam',loss='mean_absolute_error', metrics=[my_metric_fn])
#model.compile(optimizer='adam',loss='mean_squared_logarithmic_error', metrics=[my_metric_fn])

model.fit(x_train, train_onehot, epochs=1000)

# test data 확인
prediction_y=model.evaluate(x_test, verbose=0)

pre_y=prediction_y
dataframe=pd.DataFrame(pre_y)
dataframe.to_csv("pre_y.csv")

# model 어떻게 생겼는 지 확인
model.summary()
'''
