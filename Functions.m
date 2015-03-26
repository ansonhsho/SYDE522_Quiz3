function [outputs,rmse] = netcreation(hiddenunits,xtrain,ytrain,xtest,ytest)

    net = feedforwardnet(hiddenunits);
    net = train(net,xtrain,ytrain);
    view(net)

    outputs =sim(net,xtest);
    rmse= sqrt(sum((ytest-outputs).^2))
end

