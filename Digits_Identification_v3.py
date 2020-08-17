# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 23:46:02 2020

@author: somabhupal
"""
import os
import pandas as pd
import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
#from scipy.misc import imread
from matplotlib.pyplot import imread
#import cv2
# the data, split between train and test sets

#root_dir = os.path.abspath('../..')
ch_wdr = os.chdir('C:/Users/Data_Science_with_Python/Digits_Identification')
data_dir = os.getcwd()
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

temp = []
for img_name in train.filename:
    image_path = os.path.join(data_dir, 'Images', 'train', img_name)
    img = imread(image_path, '.png')
    img = img.flatten()/255
    img = img.astype('float32')
    temp.append(img)
    #print(img.size)
train_x = np.stack(temp)    
train_x = train_x.reshape(-1, 3136).astype('float32')
print(train_x.shape)
train_y = keras.utils.np_utils.to_categorical(train.label.values)
print(train_y.shape)

# Initializes a sequential model
model = Sequential()

# First layer
model.add(Dense(100, activation='relu', input_shape=(3136,)))

# Second layer
model.add(Dense(50, activation='relu'))

# Output layer
model.add(Dense(10,activation = 'softmax'))

# Compile the model
model.compile(optimizer='adam', 
           loss='categorical_crossentropy', 
           metrics=['accuracy'])

# Reshape the data to two-dimensional array
#train_data = train_data.reshape(50, 784)

# Fit the model
model.fit(train_x, train_y, validation_split=0.2, epochs=10)

temp = []
for img_name in test.filename:
    image_path = os.path.join(data_dir, 'Images', 'test', img_name)
    img = imread(image_path, '.png')
    img = img.flatten()/255
    img = img.astype('float32')
    temp.append(img)

test_x = np.stack(temp)    
test_x = test_x.reshape(-1, 3136).astype('float32')

pred = model.predict_classes(test_x)

df = test.filename
pred = pd.DataFrame(pred,columns = ['label'])
pred = pd.concat([df,pred], axis = 1)
pred.to_csv('submission.csv',index = False)
