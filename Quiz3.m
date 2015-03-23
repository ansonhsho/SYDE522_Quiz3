close all;
clear all;

import = importdata('/GS.csv');

% Decision Trees
x = import.data(:,1);
y = import.data(:,2);

[trian, val, test] = dividerand(import.data, 0.7, 0, 0.3);

tree = fitrtree(x,y);

view(tree,'mode', 'graph');

tree2 = prune(tree,'level', 10);

view(tree2,'mode', 'graph');

% Neural networks
net = feedforwardnet(10);
net = train(net, x,y);
view(net);
