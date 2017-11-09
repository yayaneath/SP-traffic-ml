function [cnn_model, predictions, accuracy, confusion] = ...
    trainAndTestCNNnorm(path, image_size, channels, ...
    train_size, epochs, batch_size, learning_rate)
%Read the training data, train CNN and test it.

    % Create the datasets
    disp('### Reading dataset...');
    
    % Get the files' names
    labels_folders = dir(path);
    labels_folders = labels_folders(3:end); % Avoid '.' and '..'

    dataset = zeros([image_size image_size channels 0]);
    num_labels = zeros([0 1]);

    % For each label/class in the dataset
    for label = 1 : length(labels_folders)        
        files_folder = strcat(path, '/', labels_folders(label).name);
        files = dir(files_folder);
        files = files(3:end);

        disp('************************');
        disp(files_folder);

        % Read files data, resize them and store in the new folder
        for file = 1 : length(files)
            file_name = files(file).name;
            file_path = strcat(files_folder, '/', file_name);
            image_original = imread(file_path);
            dataset = cat(4, dataset, image_original);
            num_labels = cat(1, num_labels, ones(1, 1) * label);

            disp(strcat('#', file_path));
        end
    end

    labels = categorical(num_labels);
    label_count = size(unique(labels), 1);
    
    % Display random images
    disp_images = 20;
    dataset_samples = size(dataset, 4);
    perm = randperm(dataset_samples , disp_images);

    figure(1)
    for i = 1 : disp_images
        subplot(4, 5, i)
        imshow(dataset(:, :, :, perm(i)))
        title(char(labels(perm(i))))
    end
    drawnow
    
    % Labels distribution
    figure(2)
    hist(labels)
    drawnow
    
    % Image size
    images_dim = [image_size image_size channels];

    % Normalize images
    dataset_norm = double(dataset) / 255.0;
    
    % Split into train and test sets
    disp('### Spliting...');
    
    indexes = false(dataset_samples, 1);
    indexes(1:floor(dataset_samples * train_size)) = true;
    indexes = indexes(randperm(dataset_samples));

    train_data = dataset_norm(:, :, :, indexes);
    train_labels = labels(indexes);
    test_data = dataset_norm(:, :, :, ~indexes);
    test_labels = labels(~indexes);
    
    % Define network
    disp('### Defining network...');
    conv_normal_dist = makedist('Normal', 'mu', 0, 'sigma', 0.1);
    
    % Conv-1
    % Input: Batch_size, 40, 40, 3
    % Output: Batch_size, 40, 40, 32 (32 filters) 
    conv1_input_channels = 3; % RGB
    conv1_filter_dim = 3;
    conv1_num_filters = 32;
    
    conv_l1 = convolution2dLayer(conv1_filter_dim, conv1_num_filters, ...
        'Padding', 1);
    conv_l1.Weights = conv_normal_dist.random([conv1_filter_dim ...
        conv1_filter_dim conv1_input_channels conv1_num_filters]);
    conv_l1.Bias = zeros([1 1 conv1_num_filters]);
    
    % Conv-2
    % Input: Batch_size, 20, 20, 32
    % Output: Batch_size, 20, 20, 32 (32 filters) 
    conv2_input_channels = conv1_num_filters;
    conv2_filter_dim = 3;
    conv2_num_filters = 64;
    
    conv_l2 = convolution2dLayer(conv2_filter_dim, conv2_num_filters, ...
        'Padding', 1);
    conv_l2.Weights = conv_normal_dist.random([conv2_filter_dim ...
        conv2_filter_dim conv2_input_channels conv2_num_filters]);
    conv_l2.Bias = zeros([1 1 conv2_num_filters]);
    
    % Conv-3
    % Input: Batch_size, 10, 10, 64
    % Output: Batch_size, 10, 10, 128 (128 filters) 
    conv3_input_channels = conv2_num_filters;
    conv3_filter_dim = 3;
    conv3_num_filters = 128;
    
    conv_l3 = convolution2dLayer(conv3_filter_dim, conv3_num_filters, ...
        'Padding', 1);
    conv_l3.Weights = conv_normal_dist.random([conv3_filter_dim ...
        conv3_filter_dim conv3_input_channels conv3_num_filters]);
    conv_l3.Bias = zeros([1 1 conv3_num_filters]);
    
    % FC-1
    fc1_input_size = 5 * 5 * conv3_num_filters; % After maxpool on conv3
    fc1_units = 1024;
    fc1_normal_dist = makedist('Normal', 'mu', 0, ...
        'sigma', sqrt(2.0 / fc1_input_size));
    
    fc1_layer = fullyConnectedLayer(fc1_units);
    fc1_layer.Weights = fc1_normal_dist.random([fc1_units fc1_input_size]);
    fc1_layer.Bias = zeros([fc1_units 1]);
    
    % FC-2
    fc2_input_size = fc1_units;
    fc2_units = 512;
    fc2_normal_dist = makedist('Normal', 'mu', 0, ...
        'sigma', sqrt(2.0 / fc2_input_size));
    
    fc2_layer = fullyConnectedLayer(fc2_units);
    fc2_layer.Weights = fc2_normal_dist.random([fc2_units fc2_input_size]);
    fc2_layer.Bias = zeros([fc2_units 1]);
    
    % FC-3
    fc3_input_size = fc2_units;
    fc3_units = 256;
    fc3_normal_dist = makedist('Normal', 'mu', 0, ...
        'sigma', sqrt(2.0 / fc3_input_size));
    
    fc3_layer = fullyConnectedLayer(fc3_units);
    fc3_layer.Weights = fc3_normal_dist.random([fc3_units fc3_input_size]);
    fc3_layer.Bias = zeros([fc3_units 1]);
    
    % Output
    fcOut_input_size = fc3_units;
    fcOut_units = label_count;
    fcOut_normal_dist = makedist('Normal', 'mu', 0, ...
        'sigma', sqrt(2.0 / fcOut_input_size));
    
    fcOut_layer = fullyConnectedLayer(fcOut_units);
    fcOut_layer.Weights = fcOut_normal_dist.random([fcOut_units fcOut_input_size]);
    fcOut_layer.Bias = zeros([fcOut_units 1]);
    
    
    cnn_layers = [
        imageInputLayer(images_dim, 'Normalization', 'None') % Input layer
        
        % 1st Conv(kernel_size, filters)
        conv_l1
        reluLayer
        maxPooling2dLayer(2, 'Stride', 2)
        
        % 2nd Conv(kernel_size, filters)
        conv_l2
        reluLayer
        maxPooling2dLayer(2, 'Stride', 2)
        
        % 3rd Conv(kernel_size, filters)
        conv_l3
        reluLayer
        maxPooling2dLayer(2, 'Stride', 2)
        
        % FC1(1024)
        fc1_layer
        reluLayer
        
        % FC2(512)
        fc2_layer
        reluLayer
        
        % FC3(256)
        fc3_layer
        reluLayer
        
        % FCOUT
        fcOut_layer
        softmaxLayer
        classificationLayer
    ];

    % Train options
    train_options = trainingOptions('sgdm', ...
        'InitialLearnRate', learning_rate, ...
        'MiniBatchSize', batch_size, ...
        'MaxEpochs', epochs, ...
        'Verbose', true, ...
        'VerboseFrequency', 1, ...
        'OutputFcn',@plotTrainingAccuracy, ...
        'ExecutionEnvironment', 'parallel');
    
    % Train
    disp('### Training CNN...');
    figure(3); % So training plots are in a new figure
    cnn_model = trainNetwork(train_data, train_labels, cnn_layers, ...
        train_options);
    
    % Validation on test set
    disp('### Calculating predictions...');
    predictions = classify(cnn_model, test_data);
    accuracy = sum(predictions == test_labels) / numel(test_labels);
    
    disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    fprintf('Final accuracy on test set: %.6f\n', accuracy);
    confusion = confusionmat(test_labels, predictions);
    
    % Show some predicted labels
    perm = sort(randperm(size(test_data, 4), 20));

    figure(4)
    for i = 1 : 20
        subplot(4, 5, i)
        imshow(test_data(:, :, :, perm(i)))
        title(char(predictions(perm(i))))
    end
    drawnow
end