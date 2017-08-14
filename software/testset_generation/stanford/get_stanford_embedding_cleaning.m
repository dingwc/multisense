%basic rehash of run.m from provided code, written as a function
function [pro,dist] = get_stanford_embedding_cleaning(id, cids, centers, orig2cent, oWe, tfidf,distfun)
    dsz = size(oWe,2);
    if orig2cent(id) == 0
        pro = 1;
        dist = 0;
    else
        c = squeeze(centers(:,orig2cent(id),:));
        tf = sparse(cids(:),ones(size(cids(:))),tfidf(cids(:)),dsz,1);
        tf = bsxfun(@rdivide,tf,sum(tf));
        contexts = reshape(oWe,50,[]) * tf;
        dist = slmetric_pw(contexts,c,distfun);
        [~,cluster] = min(dist,[],2);
        
        pro = cluster;
        dist = dist(cluster);
    end
end