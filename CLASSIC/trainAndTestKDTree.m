function [tree_model, neighbours, predictions, accuracy, confusion] = ...
    trainAndTestKDTree(path, train_size, k, distance)
%Read the training data and test the accuracy of the KDtree.

    % Create the datasets
    [train_data, train_labels, test_data, test_labels] = ...
        createAndSplitDataset(path, train_size);
    
    tic % Required for measuring time

    % Create KDTree to hold the train data
    disp('### Creating KDTreeSearcher...');
    tree_model = KDTreeSearcher(train_data, 'Distance', distance);
    
    % Query for the closest neighbours
    disp('### Querying for test data...');
    closests = knnsearch(tree_model, test_data, 'K', k);
    
    % Check what are the predictions
    disp('### Calculating predictions...');
    neighbours = train_labels(closests);
    predictions = mode(neighbours, 2);
    
    % Calculate accuracy
    succeed = predictions == test_labels;
    accuracy = sum(succeed) / size(test_labels, 1) * 1.0;
    
    disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    fprintf('Final accuracy on test set: %.6f\n', accuracy);
    confusion = confusionmat(test_labels, predictions);
    
    toc % Required for measuring time
end