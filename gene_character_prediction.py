"""
Next Character prediction for genome sequence

@author: Venkata
"""

from utils import losses
from utils.preprocessing import one_hot_encoding_sequences
from models.baseNN import neuralnets
from models.baseNN import Config
from data_generation import data_generators
from data_generation.GenemeDataFetcher import read_genome_sequence_fromFile, writeFeatureToDisk
import data_generation.DataGenerator
from data_generation.DataGenerator import read_genome_fasta, getData
from data_generation import roi_helpers

import numpy as np
from numpy import random

import sys
import time

from sklearn.model_selection import train_test_split

#from random import random
from numpy import array
from numpy import cumsum
from matplotlib import pyplot
from pandas import DataFrame
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers import Dropout
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

np.random.seed(9)
sys.setrecursionlimit(40000)
c = Config.Config()
genome_ids = ['511145.12', '100226.15', '107806.10', '1028307.3']

#sequence length
input_length = 50

chars = sorted(['a','t','g','c'])
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

print(char_to_int)
print(int_to_char)


#total number of characters
n_chars = 0
n_vocab = len(chars)
X = []
y = []

#populating X and target y.
for genome_id in genome_ids[0:3]:
	#genome_id = genome_ids[0]


	wholeSequence = read_genome_fasta(str(genome_id))

	n_chars+=len(wholeSequence)

	num_of_substrings = len(wholeSequence)/input_length
	start = 0

	for i in range(1, num_of_substrings):

		#creating subsequence of length = input_length
		try:
			subseq = wholeSequence[start:i*input_length]
			#print subseq
			#print [char_to_int[char] for char in subseq]
			#exit(0)
			#outputChar is the middle character in the string
			outputChar = wholeSequence[(start+i*input_length)/2]
			#outputChar = wholeSequence[(i*input_length)+1]
			X.append([char_to_int[char] for char in subseq])
			y.append(char_to_int[outputChar])
		except:
			continue
		start = i*input_length



#reshaping X to num_sequences x inputh_length x 1
X = np.reshape(X, (len(X), input_length, 1))

#normalizing
X = X/float(n_vocab)


#one-hot encode the ouput variable
y = np_utils.to_categorical(y)

from sklearn.utils import shuffle
X, y = shuffle(X, y, random_state = 6)


print('X_shape : {0}, y_shape : {1}').format(X.shape, y.shape)

#train_test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

#Sequential model
model = Sequential()
model.add(Bidirectional(LSTM(100, return_sequences=True),input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(100)))
model.add(Dropout(0.2))
model.add(Dense(y_train.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['acc'])

#filepath="weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"

#checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
#callbacks_list = [checkpoint]


a = model.fit(X, y, epochs=4, batch_size=128, validation_split = 0.20)#, callbacks = callbacks_list)

for i in range(10):
	print X_test[i].shape
	testing_sample = np.reshape(X_test[i,:,:], (1, input_length, 1))
	print int_to_char[np.argmax(model.predict(testing_sample))], ',', y_test[i]

#res = model.evaluate(X_test, y_test, batch_sizw = 512, verbose = 1)

#print res
"""
print type(a)
print a['acc']
print a['loss']
print a['val_acc']
print a['']

"""