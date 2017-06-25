function wP = slidingWindow(x0,y0,xf,yf,objW,objH,overlap)

% wStep = round(objW/2);
% hStep = round(objH/2);
% wStep = round(objW/5);
% hStep = round(objH/5);
% wStep = round(objW/3);
% hStep = round(objH/3);
% wStep = round(objW/4);
% hStep = round(objH/4);
wStep = round(objW*overlap);
hStep = round(objH*overlap);
% upper left coordinates of the bounding boxes
xl = x0:wStep:xf;
yl = y0:hStep:yf;
% upper right coordinates of the bounding boxes
xr = xf:-wStep:x0;
yr = yf:-hStep:y0;

wP  = zeros(length(xl)*length(yl)+length(xr)*length(yr),4);
ind = 1;
for i = 1 : length(xl)
    for j = 1 : length(yl)        
        wP(ind,  :) = [xl(i),yl(j),objW,objH];
        wP(ind+1,:) = [xr(i)-(objW-1),yr(j)-(objH-1),objW,objH];        
        ind = ind + 2;
    end
end

wP(wP(:,1) < x0 | wP(:,2) < x0 | (wP(:,1) + (objW-1)) > xf | (wP(:,2) + (objH-1)) > yf,:) = [];

