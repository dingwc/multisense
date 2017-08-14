# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 15:31:03 2016

@author: suny2
"""
import numpy as np
import scipy.sparse as ss
import math
import string


def load_freq(freq_filename,vocab2id):
    fid = open(freq_filename,'r')
    freq = [None for k in xrange(len(vocab2id))]
    for line in fid:
        l = line.split(',')
        i = vocab2id[l[0]]
        f = int(float(l[1].strip('\n')))
        freq[i] = f
    return np.array(freq)

def load_vocab(vocabIdxFilename):
    vocab2id = {}
    
    f = open(vocabIdxFilename,'r')
    for w in f:
        w2 = w.split(',')
        i = int(w2[1].strip())
        vocab2id[w2[0]] = i-1
    f.close()
    return vocab2id
    

    
def load_freq_vocabid(fpath):
    
    vocab_filename = 'enwiki_wordID2000.csv'
    freq_filename = 'enwiki_freq2000.csv'
        
    vocab2id = load_vocab('%s/%s' % (fpath,vocab_filename))
    freq = load_freq('%s/%s' % (fpath,freq_filename),vocab2id)
    return vocab2id, freq
    

            
def get_dist(x,y,alpha):
    
    x,y = np.array(x),np.array(y)
    x = np.array(x)
    y = np.array(y)
    d = np.dot(x,y)
    if alpha > 0.0:
        dn = np.linalg.norm(x)*np.linalg.norm(y)
        if alpha == 0.5: dn = np.sqrt(dn)
        if dn == 0: d = np.nan
        else: d /= dn
    
    return d