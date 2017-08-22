"""
Created on Mon Jun 26 2017.

@author: Venkata Sai Sriram Pillutla

"""

from bs4 import BeautifulSoup
import urllib
import pandas as pd
from Bio import SeqIO

#FTP URL for PATRIC
PatricURL = 'ftp://ftp.patricbrc.org/patric2/genomes/'


def read_genome_fasta(genome_id):
    path = 'sample_data/genome_sequences/{}.fna'.format(genome_id)
    records = list(SeqIO.parse(path, "fasta"))

    return str(records[0].seq)


def getData(genomeId):
    #Reading dataset

    #try:
    data_frame = pd.read_table('sample_data/genome_sequences/'+str(genomeId)+'.PATRIC.features.tab')

    #keeping only CDS rows
    data_frame = data_frame[(data_frame.feature_type == 'CDS')]
    return data_frame[['start','end','strand']]
    #except:
    #    print 'ERROR: Error reading start, end, strand values of '+ str(genomeId)+ ' from disk. Check if the csv file exists. If not please create it using writeFeaturesToDisk()'




def getGenomeSequence(genomeId):
    """
    This method fetches the genome sequence based on genomeid from PATRIC

    Parameter: genomeId
    """
    
    r = urllib.urlopen(PatricURL+genomeId+'/'+genomeId+'.fna').read()
    soup = BeautifulSoup(r)
    #print type(soup)

    genomeSequence = soup.prettify().split('| '+genomeId+']')[1]
    return genomeSequence.replace('\n', '')


def getFeaturesForGenome(genomeId, CDS_ONLY):
    """
    This method gets the features for a particular genomeId frfom PATRIC

    Parameters

    genomeId: UniqueId for the genome
    CDS_ONLY: retrieve only CDS features
    """
    data_table = pd.read_table(PatricURL
                               +genomeId+'/'+genomeId+'.PATRIC.features.tab')

    
    print data_table.shape

    if CDS_ONLY:
        return data_table[(data_table.feature_type == 'CDS')]
        
    else:
        return data_table

 



