clc
clear all
close all

profile on
path(path,'./02_jumping')
path(path,'./functions')

for i = 1 : 313  
    frameNumb = '00000';
    i_str = num2str(i);
    frameNumb(end-(length(i_str)-1):end)=i_str;
    frames(:,:,:,i) = imread(strcat(frameNumb,'.jpg'));
end
save('jumping.mat','frames')

% frames(:,:,:,2) = imread('00002.jpg');
% frames(:,:,:,3) = imread('00003.jpg');
% frames(:,:,:,4) = imread('00004.jpg');
% frames(:,:,:,5) = imread('00005.jpg');
% frames(:,:,:,6) = imread('00006.jpg');
% frames(:,:,:,7) = imread('00007.jpg');
% frames(:,:,:,8) = imread('00008.jpg');
% frames(:,:,:,9) = imread('00009.jpg');
% frames(:,:,:,10)= imread('00010.jpg');

% H = size(frames,1);
% W = size(frames,2);
% p = 0.1;
% if ~exist('frames131.mat')
%     scale = 1; % training patches on original image scale
%     nP    = 500; % max. number of patches both bg and obj may have
%     xSize = 25; ySize = xSize; % width and height of the patches
%     for i = 131%size(frames,4)   
%         imshow(uint8(frames(:,:,:,i)));
%         warning('off')
%         h     = msgbox('Select the object-square region by clicking and dragging','Object Square','modal');
%         objbbox = floor(getrect);
%         [bgP,objP,objbbox] = posAndNegPatchesV2(objbbox,H,W,p);
%         save(strcat('frames',num2str(i)),'objP','bgP','objbbox');
%     end
% end
% frames = double(frames);
% 
% %% Support Vector Classifier Initialization
% 
% load('frames131','objP','bgP','objbbox','bgbbox');
% nBins = 10;
% [h_obj,c_obj] = nccDist(frames(:,:,:,1),objP,nBins);
% [h_bg ,c_bg ] = nccDist(frames(:,:,:,1),bgP,nBins);
% tic
% C    = trainSVC_hognss(frames(:,:,:,131),objP,bgP);
% t1 = toc
% plot(c_obj,h_obj,'-.r',c_bg,h_bg,'--')
% 
% %% Detection in subsequent frames
% N           = 500;
% bestObjbbox = [];
% bestBgbbox  = [];
% objP_r      = objP;
% bgP_r       = bgP;
% % P_r         = P;
% dist1 = 100;
% dist2 = 100;
% ProbBest1 = 0.1;
% ProbBest2 = 0.1;
% Sim1 = 0.5;
% Sim2 = 0.5;
% objbbox_r   = objbbox;
% objbbox = objbbox+[-50 0 0 0];
% I = zeros(size(frames));
% 
% for f = 131:size(frames,4)
%     f
% %     for t = 1 
%               
%         if f==200
%         disp(':)')
%         end
%         if f==300
%         disp(':)')
%         end
%         frame_f = rgb2gray(uint8(frames(:,:,:,f)));
%         P1      = [];
%         P2      = [];
%         step = 2;
%         i_displacements = round(-objbbox(4)/2):step:round(objbbox(4)/2);
%         j_displacements = round(-objbbox(3)/2):step:round(objbbox(3)/2);
%         N        = length(i_displacements)*length(j_displacements);
%         objbbox_ = zeros(N,4);
%         o        = 1;
%         for i = i_displacements
%             for j = j_displacements
% %                 [objbbox_(o,:)] = moveSrchWindowNew(i,j,objbbox,W,H);
%                 [objbbox_(o,:)] = moveSrchWindowNew(i,j,objbbox);
% %                 if isempty(objbbox_)
% %                     o = o + 1;
% %                     continue;
% %                 end
%                 o = o + 1;
%             end
%         end
%         objbbox_(objbbox_(:,1)<1 | objbbox_(:,2)<1 | ...
%                  (objbbox_(:,1)+objbbox_(:,3)-1)>W | (objbbox_(:,2)+objbbox_(:,4)-1)>H,:)=[];
%         
% %         overlap   = (sum(bboxOverlapRatio(objbbox_,objbbox_))-1)';
%         [hog2   ] = hogFeat(frames(:,:,:,f),objbbox_,0);
%         [nss]     = nssFeat(frames(:,:,:,f),objbbox_);  
%         [hognss2] = [hog2,nss];
% %         bg  = objbbox_(:,1)+objbbox_(:,3)-1<139;
% %         obj = objbbox_(:,1)+objbbox_(:,3)-1>=139;
% %         testSet1  = dataset(double([hog2(obj,:);hog2(bg,:)])   , [ones(sum(obj),1);zeros(sum(bg),1)]);
% %         [E1,F1] = testc(testSet1*C.W_hog   ,'crisp',1);
% %         
%         Prob1   = (double(hog2)*C.W_hog);
%         sum(+Prob1(:,2)>0.5)
%         labels1 = Prob1*labeld;
%         OBJBBOX = objbbox_(labels1==1,:);
%         overlap = (sum(bboxOverlapRatio(OBJBBOX,OBJBBOX))-1)';
%         OBJBBOX_= OBJBBOX(overlap>max(overlap)*0.8,:);
%         %%% 
% %         if f == 43 || f == 1
% %             figure,imshow(drawPatches(uint8(frames(:,:,:,f)),objbbox_(labels1==1,:),2))
% %             figure, plot(objbbox_(labels1==1,1),overlap/(sum(labels1==1)),'.')
% %             xlabel('upper-left-x position')
% %             ylabel('overlap')
% %         end
%         %%%
% %                 Prob2   = (double(hognss2)*C.W_hognss);
% %                 labels2 = Prob2*labeld; 
%             
% %                 [Sp,Sn,Sim_tmp] = tldSimMeasure(frame_f,objbbox_,objP,bgP);
% %                 NN(o) = Sim_tmp;
% %                 PP(o) = Prob1(2);
% %                 if o > 1 && (Prob1(2) > 0.75) && Sim1 <= Sim_tmp                 
% %                     objbboxBest1 = objbbox_;
% %                     ProbBest1    = Prob1(2); 
% %                     [Sp,Sn,Sim1] = tldSimMeasure(frame_f,objbbox_,objP,bgP);
% %                 elseif o == 1
% %                     [hog2   ] = hogFeat(frames(:,:,:,f),objbbox,0);
% %                     Prob1   = (double(hog2)*C.W_hog);
% %                     objbboxBest1 = objbbox;
% %                     ProbBest1    = Prob1(2); 
% %                     [Sp,Sn,Sim1] = tldSimMeasure(frame_f,objbbox,objP,bgP);
% %                 end
% %                 
% %                 if o > 1 && Prob2(2) > 0.75 && Sim2 <= Sim_tmp  
% %                     objbboxBest2 = objbbox_;
% %                     ProbBest2    = Prob2(2);                     
% %                     [Sp,Sn,Sim2] = tldSimMeasure(frame_f,objbbox_,objP,bgP);
% %                 elseif o == 1
% %                     [nss]     = nssFeat(frames(:,:,:,f),objbbox_);  
% %                     [hognss2] = [hog2,nss];
% %                     Prob2   = (double(hognss2)*C.W_hognss);
% %                     objbboxBest2 = objbbox;
% %                     ProbBest2    = Prob2(2); 
% %                     [Sp,Sn,Sim2] = tldSimMeasure(frame_f,objbbox,objP,bgP);
% %                 end        
% 
% %         x = round(median(OBJBBOX(P1(:,2)>0.5,1)));
% %         y = round(median(OBJBBOX(P1(:,2)>0.5,2)));
% %         objbboxBest1 = [x,y,OBJBBOX(1,3),OBJBBOX(1,4)];        
% %         x = round(median(OBJBBOX(P2==1,1)));
% %         y = round(median(OBJBBOX(P2==1,2)));
% %         objbboxBest2 = [x,y,OBJBBOX(1,3),OBJBBOX(1,4)]; 
%         objbbox = round(median(OBJBBOX_));
%         I(:,:,:,f) = drawPatches(frames(:,:,:,f),objbbox,1);
% %         I(:,:,:,f) = drawPatches(I(:,:,:,f),objbboxBest2,2);  
% 
% 
% end