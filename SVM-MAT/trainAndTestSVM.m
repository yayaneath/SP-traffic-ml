function [svm_model, predictions, accuracy, confusion] = ...
    trainAndTestSVM(path, train_size, kernel)
%Read the training data, train SVM and test it.

    % Create the datasets
    [train_data, train_labels, test_data, test_labels] = ...
        createAndSplitDataset(path, train_size);
    
    tic % Required for measuring time
    
    % Train SVM
    disp('### Training SVMs...');
    svm_template = templateSVM('KernelFunction', kernel);
    options = statset('UseParallel',true);
    svm_model = fitcecoc(train_data, train_labels, ...
        'Learners', svm_template, ...
        'Verbose', 1, 'Options', options);
    
    % Check what are the predictions
    disp('### Calculating predictions...');
    predictions = predict(svm_model, test_data);
    
    % Calculate accuracy
    succeed = predictions == test_labels;
    accuracy = sum(succeed) / size(test_labels, 1) * 1.0;
    
    disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    fprintf('Final accuracy on test set: %.6f\n', accuracy);
    confusion = confusionmat(test_labels, predictions);
    
    toc % Required for measuring time
end

