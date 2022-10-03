load cancer_dataset.mat

targets = cancerTargets;
hiddenLayerSize = 10;

columns = size(cancerInputs, 1);
combos = dec2bin(0:2^9-1) - '0';
indecies = {};

for i = 1:size(combos, 1)
    row = combos(i, :);
    index = [];
    for j = 1:size(row, 2)
        value = combos(i, j);
        if value
            index(end+1) = j;
        end
    end
    if size(index) ~= [0, 0]
        indecies{end+1} = index;
    end
end

progressBar = waitbar(0, "Training net using all possible combinations of input sets");
results = ones(511, 9);

for index = 1:size(indecies, 2)
    features = indecies(index);
    features = cell2mat(features);
    inputs = cancerInputs(features,:);
    net = patternnet(hiddenLayerSize);
    net.divideParam.trainRatio = 70/100;    
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 15/100;
    net.trainParam.showWindow = 0;
    [net, tr] = train(net, inputs, targets);
    tInd = tr.testInd;
    tstOutputs = net(inputs(:, tInd));
    tstPerform = perform(net, targets(:, tInd), tstOutputs);
    for feature = features
        results(index, feature) = tstPerform;
    end
    waitbar(index/size(indecies, 2), progressBar, sprintf('Progress: %d %%', floor(index/size(indecies, 2)*100)))
end
close(progressBar);