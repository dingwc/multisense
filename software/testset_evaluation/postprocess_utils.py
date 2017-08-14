
import sys
sys.path.insert(0, '..')
from utils import get_dist
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_curve

def load_questions(fpath):
    fid = open(fpath,'r')
    fid.readline()
    questions = {}
    for f in fid:
        
        if len(f) == 0:
            break
        f = f.split(',')
        testnum = int(f[0])
        if testnum not in questions.keys():
            questions[testnum] = {}
        questions[testnum][int(f[1])] = {'label':f[2], 'word':f[3],'context':[f[i].strip() for i in range(4,len(f))]}
       
    fid.close()
    return questions

def importdata(fpath):
    # reads the data in the above format, and stores it in dict indexed
    # by test#, entries are tuples (label,embedding,norm)
    with open(fpath) as f:
        lcount=0
        for line in f:
            skipq = [] # stores the queries that are all 0
            if lcount==0:
                lcount+=1
                vals = line.strip('\n').split(',')
                dim = int(vals[1])
                D = {} # global dict
                
            else:
                lcount+=1
                vals = line.strip('\n').strip('\r').strip('').split(',')
                vals = vals[0:-1]
                testnum=int(vals[0])
                linenum = int(vals[1])
                
                # THIS HAS TO BE STANDARDISED FOR ALL 
                label = vals[2]
                dist = float(vals[3])
                emb = map(float,vals[4:])
               
                # only crete the key if the test should not be skipped
                if testnum not in skipq:
                    if testnum not in D.keys():
                        D[testnum] = {}
                    D[testnum][linenum] = {'label':label,'emb':emb,'dist':dist}
        return D
        

def get_sim(a,b,alist):
    if alist is None:
        alist=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.]
    s = [get_dist(a,b,i) for i in alist]
    return s
    

def compute_sim_scores(embeddings, alist = None):
    # compute similarity scores for all pairs
    # input : dictionary key = test_num , val = [(label,embedding,norm)]
    # output: tuples (test,label,score)
    simscores = {}
    for k in embeddings.keys():
        simscores[k] = {}
        for linenum in embeddings[k]:
            emb = embeddings[k][linenum]
            label = emb['label']
            embed = np.array(emb['emb'])
            if label=='q':
                query = embed
                continue
            sclist = get_sim(query,embed, alist)
            simscores[k][linenum] = {'label':label,'sim':sclist}
    return simscores
    

            


def get_scores(simscores):
    A  = {}
    c=0
    for k in simscores.keys():
        tmpvec = simscores[k]
        
        l = max(tmpvec.keys())
        y = []
        yhat=[]
        for i in tmpvec:
            
            sim = tmpvec[i]
            label = sim['label']
            if label=='t': 
                y.append(1)
            elif label=='f':
                y.append(0)
            yhat.append(sim['sim'])
        
        y = np.nan_to_num(np.array(y))
        yhat = np.nan_to_num(np.vstack(yhat))
        # check if there is only one class     
        if all(y==0) or all(y==1):
            precs = [1.]*yhat.shape[1]
            aucs = precs
            c+=1
        else:
        # compute scores
            precs = []
            aucs = []
            for j in xrange(yhat.shape[1]):
                precs.append(average_precision_score(y,yhat[:,j]))
                aucs.append(roc_auc_score(y,yhat[:,j]))
                p,r,t = precision_recall_curve(y,yhat[:,j])
        A[k] = {'auc':aucs,'prec':precs}
    return A,c

    
        
            
def return_avg_scores(A):
    A  = np.asarray(A)
    nT,ns = A.shape
    ns = ns
    scores = np.mean(A,axis=0)
    scores = list(scores[1:])
    return scores
        
    
    
             
    


                        
                
