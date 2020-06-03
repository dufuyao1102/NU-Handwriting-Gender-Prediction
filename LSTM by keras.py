# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 00:29:18 2020

@author: du
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from numpy import concatenate



train_data = pd.read_csv('c:/users/du/Desktop/SPRING 2020/Machine learning/projecr/train/train.csv')
train_ans = pd.read_csv('c:/users/du/Desktop/SPRING 2020/Machine learning/projecr/train_answers.csv')

train_ans = train_ans.iloc[:,1]
y = np.repeat(train_ans.to_numpy(), 4)
All_features = (train_data.iloc[:, 5:5020]).to_numpy()
test_idx = np.arange(3, 1128, 4)
train_idx = np.delete(np.arange(1128), test_idx)

test_X = All_features[test_idx, :]
train_X = All_features[train_idx, :]
train_y = np.repeat(train_ans.to_numpy(), 3)
# Change into 3-dimension by adding time stamp so that can be used in LSTM
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

test_y = train_ans
layer_num = 90

def fit_network(train_X, train_y, test_X, test_y, layer_num):
    np.random.seed(90)
    model = Sequential()
    model.add(LSTM(layer_num, activation='sigmoid', input_shape=(train_X.shape[1],train_X.shape[2])))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=500, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()
    # make a prediction
    yhat_train = model.predict(train_X)
    yhat_train = (yhat_train > 0.5).astype(int)
    train_XS=train_X
    train_XS = train_XS.reshape((train_XS.shape[0], train_XS.shape[2]))
    yhat_train = concatenate((yhat_train, train_XS[:, 1:]), axis=1)
    yhat_train = yhat_train[:,0]
    
    yhat_test = model.predict(test_X)
    yhat_test = (yhat_test > 0.5).astype(int)
    test_X = test_X.reshape((test_X.shape[0], train_X.shape[2]))
    yhat_test = concatenate((yhat_test, test_X[:, 1:]), axis=1)
    
    yhat_test = yhat_test[:,0]
    

    # calculate accuracy
    accuracy_train = sum(yhat_train == train_y)/len(train_y)
    accuracy_test = sum(yhat_test == test_y)/len(test_y)
    print('Train accuracy: %.3f' % accuracy_train)
    print('Test accuracy: %.3f' % accuracy_test )


fit_network(train_X,train_y,test_X,test_y, layer_num)

# Train accuracy: 0.740
# Test accuracy: 0.745