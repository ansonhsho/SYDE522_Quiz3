function [outputs,rmse_test,rmse_train] = netcreation(hiddenunits,xtrain,ytrain,xtest,ytest)

    net = feedforwardnet(hiddenunits);
    net = train(net,xtrain,ytrain);
    view(net)

    outputs_train = sim(net,xtrain);
    rmse_train= rms(outputs_train-ytrain)
    outputs = sim(net,xtest);
    rmse_test= rms(outputs-ytest)
    
end

