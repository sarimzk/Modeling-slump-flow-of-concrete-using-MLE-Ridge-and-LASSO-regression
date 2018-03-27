

filename = 'slump_test.data';
delimiterIn = ',';
headerlinesIn = 1;
slumptest = importdata(filename, delimiterIn, headerlinesIn);
var = slumptest.textdata;

for s = 1:10
    i = randperm(103);
    k = 1;
    mtrain = zeros(85,11);
    mtest = zeros(18,11);
    for j = 1:103
        if j<=85
            mtrain(j,:) = slumptest.data(i(j),:);
        else
            mtest(k,:) = slumptest.data(i(j),:);
            k = k+1;
        end
    end
    ytrain = mtrain(:,10);
    xtrain = mtrain(:,2:8);
    ytest = mtest(:,10);
    xtest = mtest(:,2:8);

    mdl = fitlm(xtrain, ytrain);
    pred = predict(mdl, xtest);
    errLin(s) = immse(pred, ytest);
    r2lin(s) = mdl.Rsquared.Ordinary;

    options = glmnetSet;
    options.alpha = 0;
    cvfitRidge = cvglmnet(xtrain, ytrain, 'gaussian', options, 'mse', 5, [], [], [], []);
    predictRidge = cvglmnetPredict(cvfitRidge, xtest);    
    errRid(s) = immse(predictRidge, ytest);
    minCVMr(s) = min(cvfitRidge.cvm);
    minLambdaRidge(s) = cvfitRidge.lambda_min;

    options = glmnetSet;
    options.alpha = 1;
    cvfitLasso = cvglmnet(xtrain, ytrain, 'gaussian', options, 'mse', 5, [], [], [], []);
    predictLas = cvglmnetPredict(cvfitLasso, xtest);
    errLas(s) = immse(predictLas, ytest);
    minCVMl(s) = min(cvfitLasso.cvm);
    minLambdaLasso(s) = cvfitLasso.lambda_min;
end

minimumMSElin = min(errLin);
minimumMSERid = min(errRid);
minimumMSElas = min(errLas);
rsquaredLin = mean(r2lin);
errLinear = mean(errLin);
errRidge = mean(errRid);
errLasso = mean(errLas);

[M1, I1] = min(minCVMr);
minimumLambdaRidge = minLambdaRidge(I1);
%minLambdaRidgeMean = mean(minLambdaRidge);
%minLambdaLassoMean = mean(minLambdaLasso);

[M2, I2] = min(minCVMl);
minimumLambdaLasso = minLambdaLasso(I2);
