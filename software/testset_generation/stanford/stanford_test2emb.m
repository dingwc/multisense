%Main function for generating stanford embeddings for each test
% rfilename : name of "question" file
% wfilename : name of output (embedding) file
% dim : dimension of embedding
% multiemb: object containing stanford-provided objects (vocab, tfidf, oWe,
%           numEmbeddings, centers, orig2cent, We
% distfun: any of the distance functions provided by stanford code. (see
%           comments in slmetric_pw.m)
function stanford_test2emb(rfilename, wfilename, dim, multiemb, distfun)

vocab = multiemb.vocab;
tfidf = multiemb.tfidf;
numEmbeddings = multiemb.numEmbeddings;
oWe = multiemb.oWe;
centers = multiemb.centers;
orig2cent = multiemb.orig2cent;
We = multiemb.We;

fid = fopen(rfilename,'r');


fidw = fopen(wfilename,'w');

nTests = str2double(fgetl(fid));
fprintf(fidw,'%d,%d\n',nTests,dim);

lid = 0;
testnum = nan;
while(1)
    lid = lid + 1;
    if mod(lid,100) == 0
        fprintf('%d out of %d\n',testnum,nTests)
    end
    wholeline = fgetl(fid);
    line = wholeline;
    if line < 0
        break
    end
    
    [testnum,line] = strtok(line,',');
    testnum = str2double(testnum);
    [linenum,line] = strtok(line,',');
    linenum = str2double(linenum);
    [label,line] = strtok(line,',');
    [word,line] = strtok(line,',');
    context = [];
    while 1
        [head,line] = strtok(line,',');
        if isempty(head)
            break
        end
        head = strtrim(head);
        context = [context, {head}];
    end
    word = strtrim(word);
    
    
    wid = find(strcmpi(vocab,word));
    if isempty(wid)
        emb = zeros(dim,1);
        dist = 0;
        error(sprintf('word not in vocab: %s', word))
        
    else
        cid = [];
        for k = 1:length(context)
            c = find(strcmpi(vocab,context{k}));
            if ~isempty(c)
                cid = [cid,c];
            end
        end
        if isempty(cid)
            error('context is empty: %s',context)
        else
            [pro,dist] = get_stanford_embedding_cleaning(wid,cid,centers, orig2cent, oWe, tfidf, distfun);
            dist = 1/dist;
            emb = We(:,wid,pro);
        end
    end
    
   
    fprintf(fidw,'%d,%d,%s,%f,',testnum,linenum,label,dist);
    fprintf(fidw,'%f,',emb);
    fprintf(fidw,'\n');

end
fclose(fid);
fclose(fidw);

'done'