import sys
sys.path.insert(0, '..')

from utils import *
import postprocess_utils as pu
import numpy as np
from scipy import stats


#%% Settings here



vocabtype = 'enwiki' 
testclass = 'wordnet'
questions = pu.load_questions('../tests/wordnet_%s_questions_comparison.txt' % vocabtype)

#%%

def normalize_histogram(hist,bins):
    sh = 0.0
    for k in range(len(hist)):
        sh += hist[k]*(bins[k+1]-bins[k])
    
    hist = np.divide(hist, sh)
    return hist
    
def agg_scores(simscores, questions, quantscores):
    spearmans = {}
    aucs = {}
    precisions = {}
    for key in simscores.keys():
        aggunit = questions[key]
        def get_spearman(truth, guess):
            return stats.spearmanr(truth, guess)[0]
        def get_auc(key):
            return quantscores[key]['auc'][0]
        def get_prec(key): return quantscores[key]['prec'][0]
        labels = []
        sims = []
        for k in simscores[key].keys():        
            t = simscores[key][k]
            if t['label'] == 't':
                labels.append(1)
            else: labels.append(0)
            sims.append(t['sim'][0])
            aggunit[k]['sim'] = t['sim'][0]
        spearmans[key] = get_spearman(labels, sims)
        aucs[key] = get_auc(key)
        precisions[key] = get_prec(key)
        
    return spearmans, aucs, precisions
        
def print_all( questions, simscores, spearmans,  aucs, precisions):
    for key in simscores.keys():
        q = questions[key][0]
        s = "\nquery: %s, \t %s" % (q['word'], ",".join(q['context']))
        print s
        for k in simscores[key].keys():
            q = questions[key][k]
            t = simscores[key][k]
            s = t['label']+",\t"+("%.2e,\t"*len(t['sim']) % tuple(t['sim']))+ (" %s, \t %s" % (q['word'], ",".join(q['context'])))
            
            print s
        
        s = "Spearman: %f, AUC: %f, prec: %f\t\n" % (spearmans[key], aucs[key], precisions[key]) 
        print s
    spearmans = [v for v in spearmans.values() if not np.isnan(v)]
    aucs = [v for v in aucs.values() if not np.isnan(v)]
    precisions = [v for v in precisions.values() if not np.isnan(v)]
    print "Average spearman: %f, average AUC: %f, average precision: %f" % (np.mean(spearmans), np.mean(aucs),np.mean(precisions))
    
    
def run_comparison(loadname):
    for alpha in [0.0,1.0]:
        
        
        embeddings = pu.importdata(loadname)
            
        simscores = pu.compute_sim_scores(embeddings, alist = [alpha])
        quantscores,c = pu.get_scores(simscores)
        spearmans, aucs, precisions = agg_scores(simscores, questions, quantscores)
        print_all( questions, simscores, spearmans, aucs, precisions)
        print savename
if __name__ == "__main__":
    loadname = '../embeddings/stanford/embeddings/embeddings_stanford_enwiki_cityblk_wordnet_comparison.txt'
    run_comparison(loadname)