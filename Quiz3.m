close all;
clear all;

import = importdata('GS.csv');


% Decision Trees
x = import.data(:,1);
y = import.data(:,2);

% divide data so that 30% is traing dat and 70% is 
[train, val, test] = dividerand(length(import.data), 0.7, 0, 0.3);

% Training the deciion tree
tree = fitrtree(x(train),y(train));
view(tree,'mode', 'graph');
    
    label_train= predict(tree, x(train));

    figure();
    plot(x(train),label_train,'-',x(train),y(train),'-');
    title('Training set');
    legend('estimate','actual')

    rmse_train = sqrt(sum((y(train)-label_train).^2))

    label_test= predict(tree, x(test));

    figure();
    plot(x(test),label_test,'-',x(test),y(test),'-');
    title('Testing set');
    legend('estimate','actual')

    rmse_test = sqrt(sum((y(test)-label_test).^2))


% % Pruning the tree to 10 levels
prune_tree = prune(tree, 'level', max(tree.PruneList) - 10);
view(prune_tree,'mode', 'graph');

    label_train= predict(prune_tree, x(train));
    
    figure();
    plot(x(train),label_train,'-',x(train),y(train),'-');
    title('Training set on a pruned tree');
    legend('estimate','actual')

    rmse_train =sqrt(sum((y(train)-label_train).^2))

    label_test= predict(prune_tree, x(test));
    
    figure();
    plot(x(test),label_test,'-',x(test),y(test),'-');
    title('Testing set on a pruned tree');
    legend('estimate','actual')

    rmse_test = sqrt(sum((y(test)-label_test).^2))

% Neural networks
% net = feedforwardnet(10);
% net = train(net, x,y);
% view(net);
