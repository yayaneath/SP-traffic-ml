function [train_data, train_labels, test_data, test_labels] = ...
    createAndSplitDataset(path, train_size)
%Reads matrices in 'path', creates a dataset and splits it in train/test.

    tic % Required for measuring time    

    % Read mat files, which are our datasets
    files = dir(path);
    mats = files(~cat(1, files.isdir));
    
    train_data = [];
    train_labels = [];
    test_data = [];
    test_labels = [];
    
    distribution = [];
    labels = [];
    
    % Read every dataset and split it into train and test sets
    for file = 1 : size(mats, 1)
        file_name = strcat(path, '/', mats(file).name);
        file_mat = matfile(file_name);
        file_data = file_mat.dataset;
        
        samples = size(file_data, 1);
        indexes = false(samples, 1);
        indexes(1:floor(samples * train_size)) = true;
        indexes = indexes(randperm(samples));
        
        train_data = [train_data; file_data(indexes, :)];
        train_labels = [train_labels; ones(sum(indexes), 1) * file];
        test_data = [test_data; file_data(~indexes, :)];
        test_labels = [test_labels; ones(samples - sum(indexes), 1) * file];
        
        disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
        disp(file_name);
        fprintf('Label number: %d\n', file);
        fprintf('Samples: %d\n', samples);
        fprintf('Train Samples: %d\n', sum(indexes));
        fprintf('Test Samples: %d\n', samples - sum(indexes));
        
        distribution = [distribution samples];
        labels = [labels file];
    end
    
    bar(labels, distribution);
    title('Classes distribution in the dataset')
    xlabel('Label')
    ylabel('Total Samples')
    
    toc % Required for measuring time
end