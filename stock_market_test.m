%% Stock Market Prediction Model proposed by - Shikhar Makhija
%% Final Output Plotting Function

% opening the training test data
fileID = fopen('... \stock_market_train.csv');
fgetl(fileID); 

% delimitting the various sub-fields
C=textscan(fileID,'%s %f %f %f %f','delimiter',',');
fclose(fileID);

% Opening stock value for the day
Open = cell2mat(C(1,2));
Open = Open.';

% Highest stock value for the day
High = cell2mat(C(1,3));
High = High.';

% Lowest stock value for the day
Low = cell2mat(C(1,4));
Low = Low.';

% Closing stock value for the day
Close = cell2mat(C(1,5));
Close = Close.';

% Simple Moving Average for 10 and 50 days
SMA_10 = tsmovavg(Open,'s',10);
SMA_50 = tsmovavg(Open,'s',50);

% Exponential Moving Average for 10 and 50 days
EMA_10 = tsmovavg(Open,'e',10);
EMA_50 = tsmovavg(Open,'e',50);

% Input vector of the input variables
Input = {Open; High; Low; SMA_10; EMA_10; SMA_50; EMA_50};
Input = cell2mat(Input);

% Construction of feed-forward neural network
net = newff([minmax(Open); minmax(High); minmax(Low); minmax(SMA_10); minmax(EMA_10); minmax(SMA_50); minmax(EMA_50)], [abs(floor(7)), 1],
{'purelin', 'purelin', 'transIm'},'traingdx');

% Maximum number of iterations
net.trainparam.epochs = 8000;

% Desired Tolerance value
net.trainparam.goal = 1e-5;

% learning rate initialisation
net.trainparam.lr = 0.001;

% using full data to train the neural network
net.divideFcn ='dividetrain';
net = train(net, Input, Close);
t = net(Input);

% eveluating the performance of the neural network - using mse as 
% the measuring standard

perf = perform(net, Close, t);
view(net);

% Plot generation of the market values
x = 1:size(Close,2);
plot(x, Close, x, Open, x, High, x, Low);

%% Testing the constructed neural network

% Opening sample test data
fileID = fopen('C:\Users\Shikhar\Desktop\stock_market_test_final.csv');
fgetl(fileID);
C_t = textscan(fileID,'%s %f %f %f %f','delimiter',',');
fclose(fileID);

Open_t = cell2mat(C_t(1, 2));
Open_t = Open2.';
High_t = cell2mat(C_t(1, 3));
High_t = High_t.';
Low_t = cell2mat(C_t(1, 4));
Low_t = Low_t.';
Close_t = cell2mat(C_t(1, 5));
Close_t = Close_t.';
SMA_10_t = tsmovavg(Open_t, 's', 10);
SMA_50_t = tsmovavg(Open_t, 's', 50);
EMA_10_t = tsmovavg(Open_t, 'e', 10);
EMA_50_t = tsmovavg(Open_t, 'e', 50);

Input_t = {Open_t; High_t; Low_t; SMA_10_t; EMA_10_t; SMA_50_t; EMA_50_t};

Open_t = cell2mat(C_t(1,2));
Input_t = cell2mat(Input_t);

% Plotting the final output graph 
answer = ones(1, size(Close_t, 2));
answer_t = ones(1, size(Close_t, 2));
for i=1:size(Close_t, 2)
    answer(i) = net([Input_t(1, i); Input_t(2,i); Input_t(3, i); Input_t(4, i); Input_t(5, i); Input_t(6, i); Input_t(7, i)]);
    answer_t(i) = Close_t(i);
end
x = 50:size(Close_t, 2);
plot(x, answer(50:428), x, answer_t(50:428));
legend('Actual Value','PredictedValue','Location','southeast')
xlabel('Data Points');
ylabel('Closing Stock Market Value');
title('Stock Market Prediction using Neural Networks');
