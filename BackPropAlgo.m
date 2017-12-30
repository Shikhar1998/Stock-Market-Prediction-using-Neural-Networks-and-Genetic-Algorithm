function perf = BackPropAlgo(u)
%% Stock Market Prediction using Simple Neural Networks
fileID = fopen('...\stock_market_train.csv');
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
net = newff([minmax(Open);minmax(High);minmax(Low);minmax(SMA_10);minmax(EMA_10);minmax(SMA_50);minmax(EMA_50)],[abs(floor(u)),1],{'purelin', 'purelin', 'transIm'},'traingdx');
net.trainparam.epochs = 100;
net.trainparam.goal = 1e-25;
net.trainparam.lr = 0.001;
net.divideFcn ='dividetrain';
net = train(net, Input, Close);
t = net(Input);
perf = perform(net, Close, t);
if (perf<0)
    perf =  10^20;
end
