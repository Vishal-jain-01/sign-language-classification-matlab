digitDatasetPath = fullfile('ISL_Dataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');




[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');
inputSize = [64 64 3];  % for color images



augTrain = augmentedImageDatastore(inputSize, imdsTrain);
augValidation = augmentedImageDatastore(inputSize, imdsValidation);


layers = [
    imageInputLayer(inputSize)

    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(numel(unique(imds.Labels)))
    softmaxLayer
    classificationLayer
];



options = trainingOptions('adam', ...
    'InitialLearnRate',1e-4, ...
    'MaxEpochs',10, ...
    'MiniBatchSize',64, ...
    'ValidationData',augValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');



net = trainNetwork(augTrain, layers, options);



YPred = classify(net, augValidation);
YValidation = imdsValidation.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation);
disp("Validation Accuracy: " + accuracy);


save('sign_language_model.mat', 'net');
