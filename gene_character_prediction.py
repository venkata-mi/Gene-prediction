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



np.random.seed(9)
sys.setrecursionlimit(40000)


c = Config.Config()

counter = 1
gene_location_dict = {}
genome_ids = ['511145.12', '100226.15', '107806.10', '1028307.3']
#whole sequence splitted into multiple smaller sequences.
sequence = []

class_count = {}
class_mapping = {}

allSequences = {}

num_of_bg_instances = 0

#sequence length
input_length = 100

chars = sorted(['a','t','g','c'])
char_to_int = dict((c, i) for i, c in enumerate(chars))

#total number of characters
n_chars = 0
n_vocab = len(chars)
X = []
y = []
for genome_id in genome_ids[0:1]:
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
			#outputChar = wholeSequence[(start+i*input_length)/2]
			outputChar = wholeSequence[(i*input_length)+1]
			X.append([char_to_int[char] for char in subseq])
			y.append(char_to_int[outputChar])
		except:
			continue
		start = i*input_length




X = np.reshape(X, (len(X), input_length, 1))

#normalizing
X = X/float(n_vocab)


#one-hot encode the ouput variable
y = np_utils.to_categorical(y)


print('X_shape : {0}, y_shape : {1}').format(X.shape, y.shape)



model = Sequential()
model.add(Bidirectional(LSTM(512, return_sequences=True),input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(512)))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['acc'])


model.fit(X, y, epochs=2, batch_size=128)