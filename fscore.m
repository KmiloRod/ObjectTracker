function [f,precision,recall,overlap] = fscore(P,gtP,flag)

if nargin == 3
    overlap = zeros(size(P,1),1);
    nfn     = 0;
    for i = 1 :size(P,1)
        if (sum(isnan(P(i,:)))+sum(isnan(gtP(i,:)))) == 0 && flag(i) == 0
            overlap(i) = bboxOverlapRatio(P(i,:),gtP(i,:));
        end
%         if (flag(i) == 1 & sum(isnan(gtP(i,:)))==0) == 1
%            nfn = nfn + 1; 
%         end
        if sum(isnan(gtP(i,:)))==0 
           nfn = nfn + 1; 
        end
    end

elseif nargin == 2
    overlap = 1000*ones(size(P,1),1);
    nfn     = 0;
    for i = 1 : size(P,1)
        if sum(isnan(P(i,:)))+sum(isnan(gtP(i,:))) == 0
            overlap(i) = bboxOverlapRatio(P(i,:),gtP(i,:));
        end
        if sum(isnan(gtP(i,:)))==0 
           nfn = nfn + 1;
        end
%         if (sum(isnan(P(i,:)))~=0 & sum(isnan(gtP(i,:)))==0) == 1
%            nfn = nfn + 1; 
%         end
    end    
end
ntp = sum(overlap(overlap~=1000)>0.5)
nfp = sum(overlap(overlap~=1000)<=0.5)
nfn
precision = ntp/(ntp+nfp);
% recall    = ntp/(ntp+nfn);
recall    = ntp/nfn;

f = 2*precision*recall/(precision + recall);
