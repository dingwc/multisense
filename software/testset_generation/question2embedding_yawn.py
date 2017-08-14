import numpy as np
import sys
sys.path.insert(0, '..')
from utils import *
import scipy.io as sio
import h5py
from scipy.sparse import csc_matrix

#%%

    

def load_embeddings_yawn(filepath, Cfilename, Wfilename):
    C = sio.loadmat(filepath + '/' + Cfilename)['C']
    V = C.shape[1]
    W = (h5py.File(filepath + '/' + Wfilename)['W'])

    I = W['ir'][...]
    J = W['jc'][...]
    val = W['data'][...]
    W = csc_matrix((val, I,J), shape=(V,V)).todense()
    return C,W
    
def compute_question2embed(qname, aname, vocab2id,C, W):
    dim = C.shape[0]
    
    fidq= open(qname,'r')
    
    nlines = int(fidq.readline())
    
    fida= open(aname,'w')
    fida.write('%d,%d\n' % (nlines,dim))
    
    for l in fidq:
        line = l.split(',')
        testnum = int(line[0])
        linenum = int(line[1])
        label = line[2].strip()
        word = line[3].strip()
        context = [line[i].strip() for i in xrange(4,len(line))]
        
        
        wid = vocab2id[word]
        
        cid = [vocab2id[c] for c in context]
     
        if len(cid) == 0:
            vec = np.zeros((dim,1))
            nrm = 0.0
            #print 'context is too small: %d' % len(cid)
            if len(cid) == 0: raise
        else:
            vec = np.dot(C[:,cid], W[cid,wid])
            
            nrm = np.linalg.norm(vec,2)
            nrm /= (np.sqrt(len(cid)+0.0))
        
        
        fida.write('%d,%d,%s,%e,' % (testnum,linenum,label,nrm))
        for x in vec:
            fida.write('%e,' % x)
        fida.write('\n')
        
    fida.close()
        
    fidq.close()
    print 'done'
    
  
if __name__ == "__main__":
    
    
    vocabtype = 'enwiki'
    vocab2id, freq = load_freq_vocabid('..')
 
    #tail_list = [tail_list[7],tail_list[11]]
    for tail in ['counts','counts_norm']:
        
        Cfilename = 'C_trainon_wiki_vocabwiki_dim100.mat' # replace with correct C matrix filename
        Wfilename = 'enwiki_%s.mat' % tail # replace with correct W matrix filename
        C,W = load_embeddings_yawn('../yawn_embeddings/',Cfilename,Wfilename)
        
        
        testclass = 'wordnet'
        testtype = 'relevance'        
        qname = '../tests/wordnet_%s_questions_%s_withoutword.txt' % (vocabtype,testtype)  
        aname = 'embeddings/embeddings_yawn_%s_%s_%s_nowordincontext_%s.txt' % (vocabtype,testclass,testtype,tail)
        compute_question2embed(qname, aname, vocab2id, C, W)
        
                        
        testclass = 'wordnet'
        testtype = 'comparison'
        qname = '../tests/wordnet_%s_questions_%s.txt' % (vocabtype,testtype)
        aname = 'embeddings/embeddings_yawn_%s_%s_%s_%s.txt' % (vocabtype,testclass,testtype,tail)
        compute_question2embed(qname, aname, vocab2id, C, W)
        
        testclass = 'princeton' 
        testtype = 'relevance'
        princeton_tail = 'test1_m10'
        qname = '../tests/princeton_%s_%s.csv' % (vocabtype,princeton_tail)  
        aname = 'embeddings/embeddings_yawn_%s_%s_%s_%s_%s.txt' % (vocabtype,testclass,princeton_tail,testtype,tail)
        compute_question2embed(qname, aname, vocab2id, C, W)
        
        
        
        princeton_tail = 'test2_m10'
        qname = '../tests/princeton_%s_%s.csv' % (vocabtype,princeton_tail)  
        aname = 'embeddings/embeddings_yawn_%s_%s_%s_%s_%s.txt' % (vocabtype,testclass,princeton_tail,testtype,tail)
        compute_question2embed(qname, aname, vocab2id, C, W)
        
        qname = '../tests/scws_%s_questions.txt' % vocabtype
        aname = 'embeddings/embeddings_yawn_scws_%s_%s.txt' % (vocabtype,tail)  
        compute_question2embed(qname, aname, vocab2id, C, W)
        
        
        qname = '../tests/koeling_%s_questions.txt' % vocabtype
        aname = 'koeling_%s_%s' % (vocabtype,tail)  
        compute_question2embed(qname, aname, vocab2id, C, W)
        
        qname = '../tests/wsd_%s_questions.txt' % vocabtype
        aname = 'wsd_%s_%s' % (vocabtype,tail)  
        compute_question2embed(qname, aname, vocab2id, C, W)
