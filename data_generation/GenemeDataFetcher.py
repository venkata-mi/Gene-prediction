import DataGenerator
import pandas as pd

GenomeId = 511145.12


def read_genome_sequence_fromFile(genomeId):
	try:
		F = open('sample_data/genome_sequences/'+genomeId+'.txt','r') 
		sequence = F.read()
		F.close()

		sequence = sequence.encode('UTF-8')
		sequence = sequence.replace('\n','')
		sequence = sequence.replace('  </p> </body></html>','')
		return sequence
	except:
		print 'ERROR: Genome Sequence doesnot exist'
		return ""

def write_genome_sequence_toFile(genomeId, genomeSequence = ""):
	"""
	Function to write the genome sequence to file with the name genomeId.txt. 
	"""
	text_file = open(str(genomeId)+".txt", "w")
	if genomeSequence == "":
		 genomeSequence = DataGenerator.getGenomeSequence(str(genomeId))
		 genomeSequence = genomeSequence.replace('\n','')

	text_file.write(genomeSequence)
	text_file.close()


def fetchDataForGenome(GenomeId, writeToFile = False):
	genomeSequence = DataGenerator.getGenomeSequence(str(GenomeId))

	genomeSequence = genomeSequence.replace('\n','')


	df = DataGenerator.getFeaturesForGenome(str(GenomeId), True)

	#retaining only the start, end, and strand columns in the dataframe

	target_df = df[['start','end','strand']]
	del df


	target_df['subsequence']= target_df.apply(lambda row: genomeSequence[row['start']-1:row['end']-1], axis = 1)

	if writeToFile == True:
		target_df.to_csv('/home/pillutla/gene_prediction/data/'+GenomeId+'.csv')

	return target_df





def writeFeatureToDisk(genomeId):
	try:
		df = DataGenerator.getFeaturesForGenome(str(genomeId), True)
		target_df = df[['start','end','strand']]

		del df

		target_df.to_csv('sample_data/'+genomeId+'.csv')
		return True
	except:
		print 'ERROR: writing features of'+ str(genomeId)+' to disk failed. Verify GenomeDataFetcher.writeFeaturesToDisk'
		return False

