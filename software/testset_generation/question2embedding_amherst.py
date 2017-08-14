import numpy as np
import math
import scipy.sparse as ss
import sys

import os.path
import scipy.io as sio
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../misc')
from utils import *
    
#%%

"""
Load all of embeddings in one database
"""

def get_embeddings(filename,vocab2id):
    fid = open(filename ,'r')
    line = fid.readline().split(' ')
    
    vectorlist = [None for x in xrange(len(vocab2id))]
    while(1):
        line = fid.readline().split()
        if len(line) == 0: break
        word = line[0]
    
        ns = int(line[1].strip('\n'))
        
        gvec = fid.readline().split()
        gvec = [float(g.strip('\n'))  for g in gvec]
        d = {'global':gvec,'sense':[],'center':[],'word':word}
       
        for k in xrange(ns):
            cl = fid.readline().split()
            cl = [float(c.strip('\n'))  for c in cl]
            
            cc = fid.readline().split()
            cc = [float(c.strip('\n'))  for c in cc]
                  
            d['sense'].append(cl)
            d['center'].append(cc)
        if word in vocab2id:
            i = vocab2id[word]
            vectorlist[i] = d        
    fid.close()
    return vectorlist
    
    

"""
Function to transfer question (words) to answers (list of multisense embeddings)
"""
def compute_question2embed(qname, aname,vocab2id, vectorlist, disttype):
    
    fidq = open(qname,'r')
    nlines = int(fidq.readline())
    
    fida = open(aname,'w')
    fida.write('%d,%d\n' % (nlines,0))
    
    for l in fidq:
        line = l.split(',')
        
        testnum = int(line[0])
        linenum = int(line[1])
        label = line[2].strip()
        word = line[3].strip()
        context = [line[i].strip() for i in xrange(4,len(line))]
        
        
        cvec = None
        relcontext = 0.0
        for c in context:
            
            i = vocab2id[c]
            if  vectorlist[i] is None:
                continue
            relcontext += 1.0
            vg = np.array(vectorlist[i]['global'])
            if cvec is None: cvec = vg
            else: cvec += vg
                
        if relcontext == 0.0: continue
        cvec /= relcontext
        
        i = vocab2id[word]
        
        d = []
        for c in vectorlist[i]['center']:
            if disttype == 'cossim':
                d.append(get_dist(c,cvec,1.0))
            elif disttype == 'dotsim':
                d.append(get_dist(c,cvec,0.0))
                
        j = np.argmax(np.array(d))
        vec = vectorlist[i]['sense'][j]
        nrm = d[j]

        fida.write('%d,%d,%s,%f,' % (testnum,linenum,label,nrm))
        
        for x in vec:
            fida.write('%f,' % x)
        fida.write('\n')

    fida.close()
        
    fidq.close()
    print 'done'


 #%%
 
if __name__ == "__main__":
    
    
    
    vocabtype = 'enwiki' 

    filenames = [('','50d_10s','1.32clambda_0mv'),
                ('','50d_10s','1.32clambda_30000mv'),
                ('','300d_10s','1.32clambda_0mv'),
                ('test_','50d_3s','0mv'),
                ('test_','50d_3s','30000mv'),
                ('test_','300d_3s','0mv'),
                ('test_','300d_3s','30000mv')]



   
    
        
    vocab2id,freq = load_freq_vocabid('..') 
    V = len(vocab2id)

    #%%
    for ff in filenames:
        filename = 'new-vectors_'+ff[0]+'socher-wiki-match_'+ff[1]+'_1num_neg_100000v_0.001sam_0stopwords_'+ff[2]+'.txt'

        print filename
        vectorlist = get_embeddings(filename, vocab2id)
        print 'embedding loaded'
        
        for disttype in ['cossim','dotsim']:
            tail = ff[0]+'_'+ ff[1]+'_'+ ff[2] + '_maxsense_%s' % disttype
            print tail
        
            
            testclass = 'wordnet'
            testtype = 'relevance'
            qname = '../tests/wordnet_%s_questions_%s_withoutword.txt' % (vocabtype,testtype)  
            aname = 'embeddings/embeddings_amherst_%s_%s_%s_nowordincontext_%s.txt' % (vocabtype,testclass,testtype,tail)
            compute_question2embed(qname, aname, vocab2id, vectorlist,disttype)
            
            
            testclass = 'wordnet'
            testtype = 'comparison'
            qname = '../tests/wordnet_%s_questions_%s.txt' % (vocabtype,testtype)
            aname = 'embeddings/embeddings_amherst_%s_%s_%s_%s.txt' % (vocabtype,testclass,testtype,tail)
            compute_question2embed(qname, aname, vocab2id, vectorlist,disttype)
            
            testclass = 'princeton' 
            testtype = 'relevance'
            princeton_tail = 'test1_m10'
            qname = '../tests/princeton_%s_%s.csv' % (vocabtype,princeton_tail)  
            aname = 'embeddings/embeddings_amherst_%s_%s_%s_%s_%s.txt' % (vocabtype,testclass,princeton_tail,testtype,tail)
            compute_question2embed(qname, aname, vocab2id, vectorlist,disttype)
            
            
            princeton_tail = 'test2_m10'
            qname = '../tests/princeton_%s_%s.csv' % (vocabtype,princeton_tail)  
            aname = 'embeddings/embeddings_amherst_%s_%s_%s_%s_%s.txt' % (vocabtype,testclass,princeton_tail,testtype,tail)
            compute_question2embed(qname, aname, vocab2id, vectorlist,disttype)
            
            qname = '../tests/scws_%s_questions.txt' % vocabtype
            aname = 'embeddings/embeddings_amherst_scws_%s_%s.txt' % (vocabtype,tail)  
            compute_question2embed(qname, aname, vocab2id,  vectorlist,disttype)
            
            
            qname = '../tests/koeling_%s_questions.txt' % vocabtype
            aname = 'embeddings/embeddings_amherst_koeling_%s_%s.txt' % (vocabtype,tail)  
            compute_question2embed(qname, aname, vocab2id,  vectorlist,disttype)
            
            qname = '../tests/wsd_%s_questions.txt' % vocabtype
            aname = 'embeddings/embeddings_amherst_wsd_%s_%s.txt' % (vocabtype,tail)  
            compute_question2embed(qname, aname, vocab2id,  vectorlist,disttype)
            
         