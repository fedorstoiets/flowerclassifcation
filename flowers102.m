%download the network%
net = googlenet; 
load("imagelabels.mat");
load("training102flowers.mat");
%input size required for the googlenet%
inputSize = net.Layers(1).InputSize;  

%putting all the images in the datastore, look in each subfolder with the
%flowers and label those images based on the name of the subfolders
flowerds102 = imageDatastore("flowers102");  %when using the original images

%%%flowerds102 = imageDatastore("segmented");  %when using the segmented
%%%images

labels = labels';

%label all the flowernames based on the subfolders name%

categoricalLabels = categorical(labels);
flowerds102.Labels = categoricalLabels;



%my regular one %%%%
[Trainflower, Testflower, Validateflower] = splitEachLabel(flowerds102, 0.8,0.1,0.1,"randomized");

resizeTrainImgs = augmentedImageDatastore([224 224],Trainflower);
resizeTestImgs = augmentedImageDatastore([224 224],Testflower);
resizeValidateImgs = augmentedImageDatastore([224 224],Validateflower);


%training options for the system%
opts = trainingOptions(...
    "sgdm","InitialLearnRate",0.003,...
    "MaxEpochs",20,...
    "Verbose", 1,...
    "VerboseFrequency",2,...
    "ValidationData", resizeValidateImgs,...
    "ValidationFrequency", 7 ,...  
    "ValidationPatience",5, ...
    "Momentum",0.9, ...
    "Plots", "training-progress"); 




  %decrease validation frequency or increase validation patience
%training the network%
[flowernet,info] = trainNetwork(resizeTrainImgs,lgraph_1,opts);


                                
%classification, on how many images are correct
flwrPreds = classify(flowernet,resizeTestImgs);
[class, score] = classify(flowernet, resizeTestImgs);

flwrActual = Testflower.Labels;
numCorrect = nnz(flwrPreds == flwrActual);
fracCorrect = numCorrect/numel(flwrPreds);
%confusionchart(flwrActual,flwrPreds)

numImages = 10;


%----------------------------------------------------------------------------------------------------------------
