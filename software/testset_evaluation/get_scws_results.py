from pprint import pprint
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import postprocess_utils as pu
import matplotlib.image as mpimg
from scipy import stats


vocabtype = 'enwiki'

  

def compute_dist(embedding,alpha):
    dist = {}
    for k,v in embedding.items():
        
        if len(v) < 2: 
            print 'thing skipped'
            continue
        e1,e2 = v[0]['emb'],v[1]['emb']
        d = np.dot(e1,e2)
        if alpha > 0.0:
            dn = np.linalg.norm(e1) * np.linalg.norm(e2)
            dn = dn ** alpha
            d = d / dn
        truth = v[0]['label']
        print truth
        dist[k] = {'truth':truth, 'score':d}
        
    return dist
    
    
def print_results(dist, questions):
    truth = []
    result = []
    for key in dist.keys():
        q = questions[key]

        tr, sc = dist[key]['truth'],dist[key]['score']
        truth.append(float(tr))
        result.append(float(sc))
        
        
        s = q[0]['word'] + ', \t' + ', '.join(q[0]['context'])
        print s
        s = q[1]['word'] + ', \t' + ', '.join(q[1]['context'])
        print s
        
        s = ("truth: %s, score: %s" % (tr,sc))
        print s
    spearman = stats.spearmanr(np.array(truth), np.array(result))[0]
    print "\nfinal scores: spearman %f\n" % (spearman)
    
    
def  run_scws(loadfile, qfilename,disttype):
    if disttype == 'cossim': alpha = 1.0
    elif disttype == 'dotsim':  alpha = 0.0
    questions = pu.load_questions(qfilename)
    embeddings = pu.importdata(loadfile)
    
    dist = compute_dist(embeddings,alpha)
    print_results(dist,questions)
    print 'done'

if __name__ == "__main__":
    loadfile = '../embeddings/yawn/embeddings/embeddings_yawn_scws_enwiki_CgloveWcounts.txt'
    qfilename = '../tests/scws_enwiki_questions.txt'
    
    run_scws(loadfile,qfilename,'cossim')