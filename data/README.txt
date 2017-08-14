DATA

    DOCS:
       ExternalData.txt provides some links to pre-existing test sets and compared multisense embeddings. If the link is not provided, please email authors.
       
    YAWN (Yifan, Azin, Weicong, Nikhil) embeddings
        C matricies:
            C_trainon_wiki_vocabwiki_dim100.mat
            C_trainon_wiki_vocabwiki_dim50.mat <--C matrix (word2vec)
            glove_enwikiorder.mat
        W matrices   
            enwiki_counts.mat <--W matrix for counts (without dividing by frequency)
            enwiki_counts_norm.mat <--W matrix for normalized counts (as described in paper)
        Useful stats
            enwiki_freq2000.csv <-- list of word frequencies
            enwiki_wordID2000.csv <-- list of word ids
            
     - testsets (See LinksToExternalData for details)
           WCR R1 : princeton_enwiki_test1_m10.csv
           WCR R2 : princeton_enwiki_test2_m10.csv
           WCR R3 : wordnet_enwiki_questions_relevance_withoutword.txt
           SCWS : scws_enwiki_questions.txt
           our CWS : wordnet_enwiki_questions_comparison.txt
           WSC C1 : koeling_enwiki_questions.txt
           WSC C2 : wsd_enwiki_questions.txt
       
CODE

    testset_generation:
       question2embeddings_XXX.py : code showing how embeddings are generated for each type of embedding
           XXX = Stanford : Huang, Socher, Manning, Ng. Improving Word Representations via Global Contextand Multiple Word Prototypes
           XXX = Amherst : Neelakantan, Jeevan, Passos and McCallum. Efficient non-parametric estimation of multiple embeddings per word in vector space. 
           XXX = Chen : Xinxiong Chen, Zhiyuan Liu, and Maosong Sun. A unified model for word sense representation and disambiguation.
           XXX = YAWN : this work
        Under Stanford:
            stanford_files : empty directory where files from Huang, Socher, Manning, Ng embedding should be dumped
            Auxilary functions for running stanford testset generation
                get_stanfod_embedding_cleaning.m
                stanford_test2emb.m
        
    testset_evaluation:            
     - postprocessing
           get_comparison_results.py (CWS in paper)
           get_relevance_results.py (WCR in paper)
           get_scws_results.py
           postprocess_utils.py (helper functions)
           

    utils.py : necessary scripts for other code       

