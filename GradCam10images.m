% Load and preprocess the images for GradCAM
%imageSubset = readall(resizeTestImgs); % Assuming resizeTrainImgs is an augmentedImageDatastore
num10 = 10;


%imageSubset5 = resizeTestImgs.Files(60:60+num10);



% Specify the target class index for GradCAM
%[classIndex, ~] = classify(flowernet, resizeTrainImgs);
figure('Name', 'GradCAM Heatmaps');
% Iterate over the selected images and compute GradCAM heatmaps
randnum = randi(808);
for i = randnum:randnum+9 
%for i = 1:10 
    % Preprocess the image
    image = imageSubset.input{i,1}; % Read the image at index i
        
    
    %image = imread(imageSubset5{i});
   
    
    % Convert the image to double data type
    image = imresize(image, [224 224]);
    % Compute GradCAM heatmap
    heatmap = gradCAM(flowernet, image, class(i));
    % Plot the GradCAM heatmap
    subplot(2, 5, i-randnum+1); % Adjust the subplot arrangement as per your preference
    
    
    %subplot(2, 5, i);
    
    
    imshow(image);
    hold on;
    imagesc(heatmap, 'AlphaData', 0.5);
    colormap jet;
    %colorbar;
    hold off;
    title(sprintf('Image %d', i));
    text(0, size(image, 1)+20, char(flwrActual(i)), 'Color', 'b', 'FontSize', 10, 'FontWeight', 'bold');
    text(0, (size(image, 1)+15)+30, char(flwrPreds(i)), 'Color', 'r', 'FontSize', 10, 'FontWeight', 'bold');
end
