import sys
sys.path.insert(0, '..')
from utils import *
import copy
import numpy as np
import postprocess_utils as pu
from scipy import stats


vocabtype = 'enwiki'


def extract_dist_from_embeddings(embedding):
    dist = copy.deepcopy(embedding)
    for k in dist:
        for i in dist[k]:
            dist[k][i].pop('emb')
    return dist
    


def agg_all_tests(dist, questions):
    def agg_test(test,question):
        ranktest = []
        distances = []
        for linenum, line in question.items():
            if linenum not in test:
                continue
            d = {}
            d['word'] = question[linenum]['word']
            d['context'] = questions[key][linenum]['context']
            
            d['label'] = test[linenum]['label']
            d['dist'] = test[linenum]['dist']
            ranktest.append(d)
            distances.append(test[linenum]['dist'])
        return ranktest
    def evalute_ranked_test(rt):
        label = []
        sim = []
        for i in rt:
            if i['label'] == 't':
                label.append(1)
            else: label.append(0)
            sim.append(i['dist'])
        idx = np.argsort(label)
        if sim[idx[-1]] == max(sim):
            selectscore = 1
        else:
            selectscore = 0
        return stats.spearmanr(label,sim)[0],selectscore
    agg = []
    for key in dist.keys():
        if key not in questions: continue
        rt = agg_test(dist[key],questions[key])
        sp,se = evalute_ranked_test(rt)
        
        agg.append({'test':rt,'spearman':sp,'select':se})
    return agg
    
def print_results(agg):
    avr_spearman = 0.0
    avr_select = 0
    ntest = [0,0]
    for item in agg:
        
        for line in item['test']:
            s = ("%c, %f,\t" % (line['label'],line['dist'])) + line['word'] + ',\t' + ','.join(line['context'])
            print s
        s = "Overall score: spearman: %f, select: %d" % (item['spearman'], item['select'])
        if not np.isnan(item['spearman']): 
            avr_spearman += item['spearman']
            ntest[0] += 1
        if not np.isnan(item['select']): 
            avr_select += item['select']
            ntest[1] += 1
        print s
    if ntest[0] > 0:
        avr_spearman /= (ntest[0] + 0.0)
    else: avr_spearman = 0.0
    if ntest[1] > 0:
        avr_select /= (ntest[1] + 0.0)
    else: avr_select = 0.0
    print "final scores: spearman %f, select %f\n" % (avr_spearman, avr_select)
    
    
def run_relevance(loadfile, qfilename):
    
    questions = pu.load_questions(qfilename)
    embeddings = pu.importdata(loadfile)
    dist = extract_dist_from_embeddings(embeddings)
    print dist
    asdf
    agg_results = agg_all_tests(dist, questions)
    print_results(agg_results)
    print 'done'     

if __name__ == "__main__":
    qfilename = '../tests/princeton_enwiki_test1_m10.csv'
    loadfile = '../embeddings/stanford/embeddings/embeddings_stanford_enwiki_kldiv_princeton_test1_m10_relevance.txt'
    run_relevance(loadfile, qfilename)