clc
clear

addpath(genpath('stanford_files'))

load('stanford_files/vocab.mat','vocab','tfidf','numEmbeddings');
load('stanford_files/wordreps_orig.mat','oWe');
load('stanford_files/centers.mat','centers','orig2cent');
load('stanford_files/wordreps.mat','We');   % load word representations


vocabtype = 'enwiki';

%%
multiemb.vocab = vocab;
multiemb.tfidf = tfidf;
multiemb.numEmbeddings = numEmbeddings;
multiemb.oWe = oWe;
multiemb.centers = centers;
multiemb.orig2cent = orig2cent;
multiemb.We = We;



ptail_list = [{'test1_m10'},{'test2_m10'}];
withword = 'withoutword';
% minkowski,quadfrm,quaddiff,wsqdist <--these didn't work so were not run
distfun_list = [{'eucdist'},{'sqdist'},{'dotprod'},{'nrmcorr'},...
    {'corrdist'},{'angle'},{'cityblk'},{'maxdiff'},...
    {'mindiff'},{'hamming'},{'hamming_nrm'},{'intersect'},...
    {'intersectdis'},{'chisq'},{'kldiv'},{'jeffrey'}];

for i = 15%1:length(distfun_list)
    distfun = distfun_list{i}
    
%     testclass = 'wordnet';
%     testtype = 'relevance';
%     rfilename = sprintf('../tests/wordnet_%s_questions_%s_%s.txt',vocabtype, testtype, withword);
%     wfilename = sprintf('embeddings/embeddings_stanford_%s_%s_%s_%s_%s.txt',vocabtype,distfun,testclass, testtype, withword);
%     stanford_test2emb(rfilename, wfilename, 50, multiemb,distfun)
%     
%     
%     testclass = 'wordnet';
%     testtype = 'comparison';
%     rfilename = sprintf('../tests/wordnet_%s_questions_%s.txt',vocabtype, testtype);
%     wfilename = sprintf('embeddings/embeddings_stanford_%s_%s_%s_%s.txt',vocabtype,distfun,testclass, testtype);
%     stanford_test2emb(rfilename, wfilename, 50, multiemb,distfun)
    
    
    testclass = 'princeton';
    testtype = 'relevance';
    for k = 1:length(ptail_list)
        ptail = ptail_list{k};
        rfilename = sprintf('../../tests/princeton_%s_%s.csv',vocabtype, ptail);
        wfilename = sprintf('embeddings/embeddings_stanford_%s_%s_%s_%s_%s.txt',vocabtype,distfun,testclass, ptail, testtype);
        stanford_test2emb(rfilename, wfilename, 50, multiemb,distfun)
    end
%     asdf
% 
%     rfilename = sprintf('../tests/scws_%s_questions.txt',vocabtype);
%     wfilename = sprintf('embeddings/embeddings_stanford_scws_%s_%s.txt',vocabtype,distfun);
%     stanford_test2emb(rfilename, wfilename, 50, multiemb,distfun)
%       
% 
%     rfilename = sprintf('../tests/koeling_%s_questions.txt',vocabtype);
%     wfilename = sprintf('embeddings/embeddings_stanford_koeling_%s_%s.txt',vocabtype,distfun);
%     stanford_test2emb(rfilename, wfilename, 50, multiemb,distfun)
%       
%     
%     rfilename = sprintf('../tests/wsd_%s_questions_tokenize.txt',vocabtype);
%     wfilename = sprintf('embeddings/embeddings_stanford_wsd_%s_%s.txt',vocabtype,distfun);
%     stanford_test2emb_tokenized(rfilename, wfilename, 50, multiemb,distfun)
      
end
