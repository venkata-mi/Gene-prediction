import pandas as pd

#utils
def findingMaxgenomeLength(df):
	"""
	Method to find the length of the largest genome to pad the remaining genomes and reashaping them to the same length.
	"""

	max_length = 0
	for index, row in df.iterrows():
		if len(row['genome_sequence']) > max_length:
			max_length = len(row['genome_sequence'])
	return max_length


def splitGenomeByWindow(genomeSequence, windowSize, listOfGeneStartEndTuples):
	diction = {}
	sequences = []
	for currentPosition in range(0, len(genomeSequence)-windowSize+1):
		currentWindowSequence =  genomeSequence[currentPosition:currentPosition+windowSize]
		sequences.append(currentWindowSequence)
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

	for sequence in sequences:
		#returns 1 if sequence has a gene, 0 otherwise
		X.append(sequence)
		y.append(diction.get(sequence,0))

	return X,y


def splitGenomeByWindowwithNextChar(genomeSequence, windowSize):
	"""
	conv filter style window movement
	"""
	y = []
	diction = {}
	sequences = []
	for currentPosition in range(0, len(genomeSequence)-windowSize+1):
		currentWindowSequence =  genomeSequence[currentPosition:currentPosition+windowSize]
		sequences.append(currentWindowSequence)
		try:
			y.append(genomeSequence[currentPosition+windowSize+1])
		except:
			y.append(0)

	return X,y




print splitGenomeByWindowwithNextChar('abcdefghi',3)