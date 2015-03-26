
%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% Set up %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%
close all;
clear all;

% import data
import = importdata('GS.csv');

x = import.data(:,1);
y = import.data(:,2);

% divide data so that 30% is traing dat and 70% is 
[trainset, val, test] = dividerand(length(import.data), 0.7, 0, 0.3);

%%%%%%%%%%%%%%%%%%%%%%
%%% Decision Trees %%%
%%%%%%%%%%%%%%%%%%%%%%

% % Training the decision tree
tree = fitrtree(x(trainset),y(trainset));
view(tree,'mode', 'graph');
    
    label_train= predict(tree, x(trainset));

    figure();
    plot(x(trainset),label_train,'-',x(trainset),y(trainset),'o');
    title('Training set');
    legend('estimate','actual')
    xlabel('Input');
    ylabel('Output');
    print('train_tree','-dpng');
    
    rmse_train = sqrt(sum((y(trainset)-label_train).^2))

    label_test= predict(tree, x(test));

    figure();
    plot(x(test),label_test,'-',x(test),y(test),'o');
    title('Testing set');
    legend('estimate','actual');
    xlabel('Input');
    ylabel('Output');
    print('test_tree','-dpng');
    
    rmse_test = sqrt(sum((y(test)-label_test).^2))

% % Pruning the tree to 10 levels
prune_tree = prune(tree, 'level', max(tree.PruneList) - 10);
view(prune_tree,'mode', 'graph');

    label_train= predict(prune_tree, x(trainset));
    
    figure();
    plot(x(trainset),label_train,'-',x(trainset),y(trainset),'o');
    title('Training set on a pruned tree');
    legend('estimate','actual');
    xlabel('Input');
    ylabel('Output');
    print('train_treeprune','-dpng');
    
    rmse_train = sqrt(sum((y(trainset)-label_train).^2))

    label_test= predict(prune_tree, x(test));
    
    figure();
    plot(x(test),label_test,'-',x(test),y(test),'o');
    title('Testing set on a pruned tree');
    legend('estimate','actual')
    xlabel('Input');
    ylabel('Output');
    print('test_treeprune','-dpng');
    
    rmse_test = sqrt(sum((y(test)-label_test).^2))

%%%%%%%%%%%%%%%%%%%%%%%
%%% Neural networks %%%
%%%%%%%%%%%%%%%%%%%%%%%

% SINGLE HIDDEN UNIT
[output1,rmse1] = netcreation(1,x(trainset)',y(trainset)',x(test)',y(test)')
figure()
plot(x(test)',output1,'-', x(test)',y(test)','o')
title('Single Hidden Unit');
xlabel('Input');
ylabel('Output');
print('single_hidden_unit','-dpng');

% TEN HIDDEN UNIT
[output10,rmse10] = netcreation(10,x(trainset)',y(trainset)',x(test)',y(test)');
figure()
plot(x(test)',output10,'-',x(test)',y(test)','o')
title('Ten Hidden Unit');
xlabel('Input');
ylabel('Output');
print('ten_hidden_unit','-dpng');

varyhiddenlayers = [1,2,5,10,20,50,100,200];

for i=1:8
    [outputtemp,rmsetemp] = netcreation(varyhiddenlayers(i),x(trainset)',y(trainset)',x(test)',y(test)');
    storermse(i)=rmsetemp;
end

figure();
semilogx(varyhiddenlayers,storermse','-');
hold on;
semilogx(varyhiddenlayers,storermse','o');
title('Hidden Unit vs. RMSE');
xlabel('Hidden Units');
ylabel('RMSE');
print('rmse','-dpng');


