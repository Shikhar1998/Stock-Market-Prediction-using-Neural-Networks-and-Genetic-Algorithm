%% Final Output Plotting Function
fileID = fopen('C:\Users\Shikhar\Desktop\stock_market_train.csv');
fgetl(fileID); 
C=textscan(fileID,'%s %f %f %f %f','delimiter',',');
fclose(fileID);
Open = cell2mat(C(1,2));
Open = Open.';
High = cell2mat(C(1,3));
High = High.';
Low = cell2mat(C(1,4));
Low = Low.';
Close = cell2mat(C(1,5));
Close = Close.';
SMA_10 = tsmovavg(Open,'s',10);
SMA_50 = tsmovavg(Open,'s',50);
EMA_10 = tsmovavg(Open,'e',10);
EMA_50 = tsmovavg(Open,'e',50);
Input = {Open; High; Low; SMA_10; EMA_10; SMA_50; EMA_50};
Input = cell2mat(Input);
net = newff([minmax(Open);minmax(High);minmax(Low);minmax(SMA_10);minmax(EMA_10);minmax(SMA_50);minmax(EMA_50)],[abs(floor(7)),1],{'purelin', 'purelin', 'transIm'},'traingdx');
net.trainparam.epochs = 8000;
net.trainparam.goal = 1e-25;
net.trainparam.lr = 0.001;
net.divideFcn ='dividetrain';
net = train(net, Input, Close);
t = net(Input);
perf = perform(net, Close, t);
view(net);
x = 1:size(Close,2);
plot(x,Close, x, Open, x, High, x, Low);
fileID = fopen('C:\Users\Shikhar\Desktop\stock_market_test_final.csv');
fgetl(fileID); 
C2=textscan(fileID,'%s %f %f %f %f','delimiter',',');
fclose(fileID);
Open2 = cell2mat(C2(1,2));
Open2 = Open2.';
High2 = cell2mat(C2(1,3));
High2 = High2.';
Low2 = cell2mat(C2(1,4));
Low2 = Low2.';
Close2 = cell2mat(C2(1,5));
Close2 = Close2.';
SMA_10_2 = tsmovavg(Open2,'s',10);
SMA_50_2 = tsmovavg(Open2,'s',50);
EMA_10_2 = tsmovavg(Open2,'e',10);
EMA_50_2 = tsmovavg(Open2,'e',50);
Input2 = {Open2; High2; Low2; SMA_10_2; EMA_10_2; SMA_50_2; EMA_50_2};
Open2 = cell2mat(C2(1,2));
Input2 = cell2mat(Input2);
answer = ones(1,size(Close2,2));
answer2 = ones(1,size(Close2,2));
fori=1:size(Close2,2)
%answer(i) = net([Input2(1,i);Input2(2,i);Input2(3,i)]);
    answer(i) = net([Input2(1,i);Input2(2,i);Input2(3,i);Input2(4,i);Input2(5,i);Input2(6,i);Input2(7,i)]);
    answer2(i) = Close2(i);
end
x = 50:size(Close2,2);
plot(x,answer(50:428),x,answer2(50:428));
legend('Actual Value','PredictedValue','Location','southeast')
xlabel('Data Points');
ylabel('Closing Stock Market Value');
title('Stock Market Prediction using Neural Networks');
