import pandas as pd
import numpy as np

RCAdata = pd.read_csv('root_cause_analysis.csv')

print(RCAdata.head())

RCAdata.info()

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

labelencoder = LabelEncoder()
RCAdata['ROOT_CAUSE'] = labelencoder.fit_transform(RCAdata['ROOT_CAUSE'])
print(RCAdata.head())

X_data = RCAdata[['CPU_LOAD','MEMORY_LEAK_LOAD','DELAY','ERROR_1000','ERROR_1001','ERROR_1002','ERROR_1003']]
y_data = RCAdata['ROOT_CAUSE']

print(X_data.head())
print(y_data.head())

import tensorflow as tf
from tensorflow import keras

y_data = keras.utils.to_categorical(y_data, 3)

print(y_data)
print(X_data.info())

X_data = X_data.to_numpy()
print(X_data)

X_train, X_test,y_train,y_test = train_test_split(X_data,y_data,test_size = 0.1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

model = keras.Sequential()

model.add(keras.layers.Dense(128,input_shape=(7,),activation='relu'))
model.add(keras.layers.Dense(128,activation='relu'))
model.add(keras.layers.Dense(3,activation='softmax'))

model.compile(loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()

history=model.fit(X_train,y_train,epochs=100,batch_size=16,verbose=1,validation_split=0.2)

import matplotlib.pyplot as plt

pd.DataFrame(history.history)["accuracy"].plot(figsize=(8, 5))
plt.title('Training and Validation Accuracy')
plt.show()

model.evaluate(X_test,y_test)

model.save('RCA_model.keras')