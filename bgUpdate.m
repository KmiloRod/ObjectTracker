function [hogFeatSet_nxt,bgKeys_nxt] = bgUpdate(I,method,perUpd,objbbox_curr,bgP_curr,bgKeys_curr,hogFeatSet_curr)

% I: current frame
% objbbox_curr: object bounding box at current frame I
% bgP_curr: background Patches
% objHOGfeat_curr: HOG features of the bg patches

imSize = size(I); 
Nbg  = size(bgP_curr,1);
Nobj = size(hogFeatSet_curr,1)-Nbg;
Nadd = round(Nbg*perUpd);

                    
switch method
    case 'msp' % minimum spatial proximity
        [bgP_nxt,bgKeys_nxt] = bgPatchGen(objbbox_curr,objbbox_curr,Nadd,imSize);
        dist_bgkeys = abs(bgKeys_curr*ones(1,Nadd) - ones(Nbg,1)*bgKeys_nxt');
        [~,idx_bgKeys] = min(dist_bgkeys);
        hogSbg_nxt = hogFeat(I,bgP_nxt);
        hogFeatSet_curr(Nobj+idx_bgKeys,:) = 0.35*hogSbg_nxt + 0.65*hogFeatSet_curr(Nobj+idx_bgKeys,:);
        hogFeatSet_nxt = hogFeatSet_curr;
        
    case 'mvc' % maximum variace change
        ibgP       = 1 : Nbg;
        bgPVar_nxt = patchVariance(I,bgP_curr);
        bgPoverlap = bboxOverlapRatio(bgP_curr,objbbox_curr);
        ibgP = ibgP(bgPVar_nxt > bgKeys_curr & bgPoverlap == 0); % bgKeys_nxt: variance vector
        if ~isempty(ibgP)
            disp(':)')
            [bgPVar_nxt,isort] = sort(bgPVar_nxt(ibgP),'descend');
            ibgP_nxt = ibgP(isort);
            if length(ibgP_nxt) > Nadd
                ibgP_nxt = ibgP_nxt(1:Nadd);
                bgPVar_nxt = bgPVar_nxt(1:Nadd);
            end
            bgKeys_curr(ibgP_nxt) = bgPVar_nxt;
            bgP_nxt = bgP_curr(ibgP_nxt,:);
%             hogSbg_nxt = hogFeat(I,bgP_nxt);
            hogSbg_nxt = hogNSSFeat(I,bgP_nxt,0,1); % HOG features from patches to update            
            hogFeatSet_curr(Nobj+ibgP_nxt,:) = 0.35*hogSbg_nxt + 0.65*hogFeatSet_curr(Nobj+ibgP_nxt,:);
        end
        bgKeys_nxt     = bgKeys_curr;
        hogFeatSet_nxt = hogFeatSet_curr;
end