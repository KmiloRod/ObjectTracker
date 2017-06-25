
fileID = fopen('gt.txt');

tline = fgetl(fileID);
gtP   = [];
while tline ~= -1
    i0    = 1;
    bbox  = [];
    for i = 1 : length(tline)
        if strcmp(tline(i),',')
        bbox = [bbox,str2num(tline(i0:i-1))];
        i0   = i+1;
        elseif i == length(tline)
           bbox = [bbox,str2num(tline(i0:length(tline)))];
        end   
    end
    gtP   = [gtP; round([bbox(1),bbox(2),(bbox(3)-bbox(1))+1,(bbox(4)-bbox(2))+1])];
    tline = fgetl(fileID);
end
fclose(fileID);
save('pedestrian2_gt.mat','gtP')