import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import one_hot
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import pandas as pd


data_location = 'gene_prediction/data'



# fix random seed for reproducibility
numpy.random.seed(7)

#window size for splitting sequences
window_size = 100

Fh = open(data_location+"/genome_sequences/511145.12.txt", "r") 

#complete genome sequence
genome_sequence = Fh.read().encode('UTF-8')
Fh.close()


#Gene locations in the genome sequence; list of tuples with starting and ending positions of the gene
#geneLocations = [(343,2799),(2801,3733),(3734,5020),(5288,5530),(6529,7959),(15445,16557)]
geneLocations = []

#maximum length - remaining length will be padded or truncated
max_genome_length = 100

top_words = 10

embedding_vecor_length = 32


def getGeneLocations(genomeId):
	df = pd.read_csv(data_location+'/'+str(genomeId)+'.csv')
	for index, row in df.iterrows():
		geneLocations.append((row['start'],row['end']))

def splitGenomeByWindow(genomeSequence, windowSize):
	"""
	conv filter style window movement
	"""
	y = []
	diction = {}
	sequences = []
	for currentPosition in range(0, len(genomeSequence)-windowSize+1):
		currentWindowSequence =  genomeSequence[currentPosition:currentPosition+windowSize]
		sequences.append(currentWindowSequence)
		#append 1 if the next character is part of a gene, 0 otherwise.
		y.append(genomeSequence[currentPosition+windowSize+1])



	return X,y



def splitGenomeByWindow(genomeSequence, windowSize, listOfGeneStartEndTuples):
	"""
	generating training samples by sliding a window on the whole genome sequence.
	"""
	diction = {}
	sequences = []
	output = []

	print 'stage1'
	for currentPosition in range(0, len(genomeSequence)-windowSize+1):
		currentWindowSequence =  genomeSequence[currentPosition:currentPosition+windowSize]
		sequences.append(currentWindowSequence)
		output.append(genomeSequence[currentPosition+windowSize-1])
		"""
		for start,end in listOfGeneStartEndTuples:
			if start>=currentPosition and start<=currentPosition+windowSize:
				#gene starting inside the window
				diction[currentWindowSequence] = 1
			elif start<=currentPosition and end<=currentPosition+windowSize:
				diction[currentWindowSequence] = 1
			elif start>=currentPosition and start <=currentPosition + windowSize:
				diction[currentWindowSequence] = 1

		
	X = []
	y = []
	print 'stage2'
	for sequence in sequences:
		#returns 1 if sequence has a gene, 0 otherwise
		X.append(sequence)
		y.append(diction.get(sequence,0))

	print 'stage3'
	return X,y
	"""
	return sequences, output

getGeneLocations(511145.12)

X, y = splitGenomeByWindow(genome_sequence, window_size, geneLocations)

print 'X_size', len(X)
print 'y_size', len(y)

print y[10:70]

le = preprocessing.LabelEncoder()
#le.fit(['a', 't', 'g', 'c'])

y_transformed = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, test_size=0.33, random_state=42)

def one_hot_encoding_sequences(seqs):
	CHARS = 'acgt'
	CHARS_COUNT = len(CHARS)

	maxlen = max(map(len, seqs))
	res = numpy.zeros((len(seqs), CHARS_COUNT * maxlen), dtype=numpy.uint8)

	for si, seq in enumerate(seqs):
	    seqlen = len(seq)
	    arr = numpy.chararray((seqlen,), buffer=seq)
	    for ii, char in enumerate(CHARS):
	        res[si][ii*seqlen:(ii+1)*seqlen][arr == char] = 1

	return res





X_train = sequence.pad_sequences(one_hot_encoding_sequences(X_train), maxlen=max_genome_length)
X_test = sequence.pad_sequences(one_hot_encoding_sequences(X_test), maxlen=max_genome_length)

X_train = numpy.array(X_train)
y_train = numpy.array(y_train)

print X_train.shape
print y_train.shape

print y_train

# create the model

model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_genome_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=64)



# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

