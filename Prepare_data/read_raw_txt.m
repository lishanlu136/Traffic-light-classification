%% ---------------------------------------------------------
% 从txt文件里读出目标的位置信息并增加背景信息随机裁剪
% by lishanlu136
%% ---------------------------------------------------------
function read_raw_txt(fileName)
prefix = 'nightClip5_';
count = 0;
fid = dlmread(fileName,'');
rows = size(fid,1);
for i = 1:rows
    imgFrame = fid(i,1);
    str = sprintf('%05d',imgFrame);
    imgName = strcat('E:\迅雷下载\VIVA_traffic_light_detection\nightTrain\nightClip5\frames\nightClip5--',str,'.png');
    I = imread(imgName);
    blubNum = fid(i,2);
    for j = 1:blubNum
        upLeft_x = fid(i,4*j-1);
        upLeft_y = fid(i,4*j);
        w = fid(i,4*j+1);
        h = fid(i,4*j+2);
        img  = crop_image(I,upLeft_x,upLeft_y,w,h);
        desImg = imresize(img,[256,256],'bilinear');
        savePath = sprintf('%s%s%05d%s','E:\迅雷下载\VIVA_traffic_light_detection\green_croped\',prefix,count,'.jpg');
        imwrite(desImg,savePath);
        count = count + 1;
    end
end
end

%% ------------------------------------------------

function img = crop_image(I,upLeft_x,upLeft_y,w,h)
[H,W,D] = size(I);
pad_x = floor(rand(1)*(40-10)+10);
pad_y = floor(rand(1)*(40-10)+10);
w0 = floor(rand(1)*(100-60)+60);
%h0 = floor(rand(1)*(100-60)+60);
upLeft_x = upLeft_x - pad_x;
upLeft_y = upLeft_y - pad_y;
w = w + w0;
h = h + w0;
if(upLeft_x<=0)
    upLeft_x = 1;
end
if(upLeft_y<=0)
    upLeft_y = 1;
end
if((upLeft_x + w) > W)
    w = W - upLeft_x;
end
if((upLeft_y + h) > H)
    h = H - upLeft_y;
end
img = I(upLeft_y:(upLeft_y+h),upLeft_x:(upLeft_x+w),:);
end


