%% -----------------------------------------------
% 自己制作的数据库，需要分训练集，验证集，测试集
% 该程序用于随机打乱数据集，然后分出它们
% by lishanlu136
%% -----------------------------------------------
clc;
clear all;
fid1 = importdata('F:\traffic_lights\trafficLightDataset\nolight\zpos.txt');  %需要分开的数据库
trainlabel = fopen('F:\traffic_lights\trafficLightDataset\traindata.txt','a');
vallabel =  fopen('F:\traffic_lights\trafficLightDataset\validationdata.txt','a');
testlabel =  fopen('F:\traffic_lights\trafficLightDataset\testdata.txt','a');
strBlank = '     ';
len = size(fid1.data,1);
a = randperm(len);
train_ratio = 0.7;
val_ratio = 0.1;
test_ratio = 0.2;
for i = 1:len
    if(i<=(len*train_ratio))
        fprintf(trainlabel,'%s%s%d\n',fid1.textdata{i},strBlank,fid1.data(i));
    elseif((len*train_ratio)<i&&i<=(len*(train_ratio+val_ratio)))
        fprintf(vallabel,'%s%s%d\n',fid1.textdata{i},strBlank,fid1.data(i));
    else
        fprintf(testlabel,'%s%s%d\n',fid1.textdata{i},strBlank,fid1.data(i));
    end
end
    %关闭文件
fclose(trainlabel);
fclose(vallabel);
fclose(testlabel);
    


