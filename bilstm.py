"""
Bi-directional LSTM to predict genes from bacterial genome sequences

seq-to-seq model 
takes the ? x 10000 x 4 input sequence and maps it to ? x 10000 X 1 output. which containes 1's in gene regions and 0's in non-gene regions. 
@author: Venkata Pillutla
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

#This variable can decide whether to add background region samples in the dataset or not.
create_bg_samples = False

for genome_id in genome_ids[0:1]:
	#genome_id = genome_ids[0]


	wholeSequence = read_genome_fasta(str(genome_id))
	dataFrame = getData(str(genome_id))


	for index, row in dataFrame.iterrows():
		#gene_start <=10000 and gene_end <=10000
		if int(row['start']) <= (counter * c.seqlen) and int(row['end'])<= (counter * c.seqlen):
			if counter-1 not in gene_location_dict.keys():
				gene_location_dict[counter-1] = []
			gene_location_dict[counter-1].append((row['start'],row['end']))

		#gene starts before 10000 but gene ends after 10000 characters
		elif int(row['start']) <= (counter * c.seqlen) and (int(row['end']) > (counter * c.seqlen)):
			if counter-1 not in gene_location_dict.keys():
				gene_location_dict[counter-1] = []
			


			gene_location_dict[counter-1].append((row['start'],counter*c.seqlen))
			#placing the ending portion of the gene in the next sequence
			gene_location_dict[counter] = []
			gene_location_dict[counter].append(((counter)*c.seqlen, row['end']))
			counter = counter + 1
		elif int(row['start']) > (counter *c.seqlen) and (int(row['start'] < (2 * counter * c.seqlen))):
			counter = counter +1
			if (int(row['end']) > (counter * c.seqlen)) and (int(row['end'] < (2 * counter * c.seqlen))):
				if counter-1 not in gene_location_dict.keys():
					gene_location_dict[counter-1] = []
				gene_location_dict[counter-1].append((row['start'], row['end']))


	#dividing the complete genome sequence into strings of length = c.seqlen
	for i in range(0, len(wholeSequence), c.seqlen):
		sequence.append(wholeSequence[i: i+c.seqlen])

	try:
		#last element null - issue should be fixed instead of hard coding
		for seqid, seq in enumerate(sequence[0:-1]):

			#print 'currently processing '+str(seqid)+' of '+str(len(sequence)-1)+' sequences.'
			if seqid not in allSequences.keys():


				"""
				a = []
				a.append(seq)
				a = one_hot_encoding_sequences(a, c.seqlen)
				a = np.array(a)
				
				
				try:
					a = a.reshape((c.seqlen, 4))
				except:
					#check
					continue
				"""
				allSequences[seqid] = {}

				allSequences[seqid]['seqid'] = seqid
				
				allSequences[seqid]['sequence'] = seq
				allSequences[seqid]['width'] = c.seqlen
				allSequences[seqid]['height'] = 1
				allSequences[seqid]['bboxes'] = []
				if np.random.randint(0,6) > 0:
					allSequences[seqid]['seqset'] = 'trainval'
				else:
					allSequences[seqid]['seqset'] = 'test'

				#adding ground truth bounding boxes
				if seqid in gene_location_dict.keys():
					
					#no gene sequence in this portion of the genome
					if len(gene_location_dict[seqid]) == 0:
						print('INFO: sequenceid {0} doesnt contain a gene.').format(seqid)
						class_name = 'bg'
						if class_name not in class_count:
							class_count[class_name] = 1
						else:
							class_count[class_name] += 1

						if class_name not in class_mapping:
							class_mapping[class_name] = len(class_mapping)	

					#gene exists in this portion of the genome					
					else:
						previous_x1 = 0
						previous_x2 = 0
						for bbox in gene_location_dict[seqid]:
							class_name = 1
							
							if seqid > 0:
								#class = 1
								current_x1 = int(bbox[0])-(seqid * c.seqlen)
								current_x2 = int(bbox[1])-(seqid * c.seqlen)
								if current_x1 <0 or current_x1 >c.seqlen or current_x2 <0 or current_x2 >c.seqlen:
									#print seqid
									#print('case 1 : {0}, {1}').format(current_x1, current_x2)
									continue
								allSequences[seqid]['bboxes'].append({'class': int(1), 'x1': current_x1 , 'x2': current_x2, 'y1': int(0),'y2': int(1)})
								

							else:
								current_x1 = int(bbox[0])
								current_x2 = int(bbox[1])
								if current_x1 <0 or current_x1 >c.seqlen or current_x2 <0 or current_x2 >c.seqlen:
									#print seqid
									#print('case 2 : {0}, {1}').format(current_x1, current_x2)
									continue
								allSequences[seqid]['bboxes'].append({'class': int(1), 'x1': current_x1 , 'x2': current_x2, 'y1': int(0),'y2': int(1)})
							
							
							#creating background bboxes

							if create_bg_samples == True and previous_x2 > 0 and current_x1 > previous_x2:
								num_of_bg_instances+=1
								allSequences[seqid]['bboxes'].append({'class': 'bg', 'x1': previous_x2 , 'x2': current_x1, 'y1': int(0),'y2': int(1)})
								if 'bg' not in class_count:
									class_count['bg'] = 1
								else:
									class_count['bg'] += 1

							previous_x2 = current_x2
							previous_x1 = current_x1

							if class_name not in class_count:
								class_count[class_name] = 1
							else:
								class_count[class_name] += 1

							if class_name not in class_mapping:
								class_mapping[class_name] = len(class_mapping)

				#allSequences[seqid]['bboxes'].append({'class': int(class_name), 'x1': int(x1) , 'x2': int(x2), 'y1': int(y1),'y2': int(y2)})


	except Exception as e:
		print(e)


class_mapping['bg'] = len(class_mapping)


print('class_mapping {}').format(class_mapping)
##each data point is a dictionary with seqid starting from 0, sequence, height = 1, width = c.seqlen, bboxes = list of single item dict


print('total num of instances = {}').format(len(allSequences))
print('num of background instances = {}').format(num_of_bg_instances)



all_data = []
for key in allSequences:
	all_data.append(allSequences[key])

random.shuffle(all_data)
num_sequences = len(all_data)

#Creating label for each sequence. creating a string of the same length as the sequence and replacing genes with 1's and remaining portion withs 0's.
for seqid, seqdic in enumerate(all_data):
	
	sequence = seqdic.get('sequence')
	label = np.zeros(len(sequence))
	temp = 0
	for location in seqdic.get('bboxes'):
		label[temp:location.get('x1')] = 0
		
		label[location.get('x1'):location.get('x2')] = 1
		
		temp = location.get('x2')

	all_data[seqid]['label'] = label


train_seqs = [s for s in all_data if s['seqset'] == 'trainval']
val_seqs = [s for s in all_data if s['seqset'] == 'test']

print('Num train samples {}'.format(len(train_seqs)))
print('Num val samples {}'.format(len(val_seqs)))

sequenceWidth = len(sequence[0])

#X = np.array(one_hot_encoding_sequences(sequence, c.seqlen))

#X = X.reshape(X.shape[0], 1, sequenceWidth, 4)

#np.random.shuffle(X)

word_to_int_input = dict((c, i) for i, c in enumerate(['a','t','g','c']))

print word_to_int_input


X = []
y = []
for seq in all_data:
	try:
		"""
		a = []
		a.append(seq['sequence'])
		a = one_hot_encoding_sequences(a, c.seqlen)
		a = np.array(a)
		a = np.transpose(a, (0, 2, 1))
		a = np.squeeze(a, axis = 0)
		"""
		a = [word_to_int_input[char] for char in seq['sequence']]
		

		X.append(a)
		#a.reshape(1, c.seqlen, 4)
		#X.append(a)
		y.append(seq['label'])
	except:
		continue


X = np.array(X)
#np.stack(X, axis = 0)
X = X.reshape(X.shape[0], c.seqlen, 1)
y = np.array(y)
y = y.reshape(y.shape[0], c.seqlen, 1)

print X.shape, ',', y.shape
print X[0].shape
print type(train_seqs)
print all_data[1]['label']

print np.where(all_data[1]['label']>0)


"""
max_features = sequenceWidth
embedding_size = 528
hidden_size = 320


words = ['a', 't', 'g', 'c']
word2ind = {word: index for index, word in enumerate(words)}
ind2word = {index: word for index, word in enumerate(words)}

print word2ind
print ind2word
"""
X_train = X[:400, :,:]
y_train = y[:400,:,:]

X_test, y_test = X[400:, :, :], y[400:, :,:]
# define problem properties
n_timesteps = c.seqlen
# define LSTM
model = Sequential()
model.add(Bidirectional(LSTM(150, return_sequences=True), input_shape=(c.seqlen, 1)))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

print(model.summary())
no_of_epochs = 1
# train LSTM
for epoch in range(no_of_epochs):
	# generate new random sequence
	#X,y = get_sequence(n_timesteps)
	# fit model for one epoch on this sequence
	print('epoch {0}/ {1}').format(epoch, no_of_epochs)
	model.fit(X_train, y_train, epochs=1, batch_size=50, verbose=2)
	#model.evaluate(X_test, y_test, verbose = 1)
	a = model.predict(np.reshape(X[401,:,:], (1, c.seqlen, 1)))
	print a.shape
	#print('epoch {0} validation loss = {1}').format(epoch, val_loss)

"""
# evaluate LSTM
X_test, y_test = X[100:160, :, :], y[100:160, :,:]
yhat = model.predict_classes(X, verbose=0)
print('predictions')
for i in range(10):
	print('Expected:', y[0, i], 'Predicted', yhat[0, i])
"""



