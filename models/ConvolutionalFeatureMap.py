# -*- coding: utf-8 -*-


"""
Generating Convolutional Feature maps from the genome sequences

"""
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
import pandas as pd
import numpy as np

np.random.seed(123)


print "Build model"

model = Sequential()
#input is 10 vectors of 32 dimensions
model.add(Conv1D(64, 3, activation='relu', input_shape=(10, 32)))
# now model.output_shape == (None, 10, 64)

# add a new conv1d on top
model.add(Conv1D(32, 3, activation='relu'))
# now model.output_shape == (None, 10, 32)


print model.output_shape

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print model.summary

