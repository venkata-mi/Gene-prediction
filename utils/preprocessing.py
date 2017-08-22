import numpy as np
from sklearn import preprocessing



def one_hot_encoding_sequences(seqs, sequenceLength):
	"""
	input: genome sequences
	output: one_hot encoded sequence array
	"""

	le = preprocessing.LabelEncoder()

	one_hot_sequences = []

	le.fit_transform(['a','t','g','c'])
	for si, seq in enumerate(seqs):
	    seqlen = len(seq)
	    arr = np.chararray((seqlen,), buffer=seq)
	    a = le.transform(arr)
	    
	    a = np.array(a)
	    
	    b = np.zeros((len(a), 4))
	    b[np.arange(len(a)), a] = 1

	    #b = np.array(b)
	    b = b.transpose()

	    if b.shape[1] == sequenceLength:
	    	one_hot_sequences.append(b)

	return one_hot_sequences

#print one_hot_encoding_sequences(['atgctgc','gctatgc'])

#print numpy.arange(8).reshape((4,8/4), order = 'F')

