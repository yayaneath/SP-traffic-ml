function [] = readHogFiles(path)
%Creates a matrix with the HOG features for each label inside a 'path' folder.

    tic % Required for measuring time
    
    FORMAT_SPEC = '%f';
    
    % Get the files' names
    labels_folders = dir(path);
    labels_folders = labels_folders(3:end); % Avoid '.' and '..'
    
    % For each label/class in the dataset
    for label = 1 : length(labels_folders)        
        dataset = [];
        files_folder = strcat(path, '/', labels_folders(label).name);
        files = dir(files_folder);
        files = files(3:end);
       
        disp('************************');
        disp(files_folder);
            
        % Read files data and concatenate them into 'dataset'
        for file = 1 : length(files)
            file_name = strcat(files_folder, '/', files(file).name);
            file_id = fopen(file_name, 'r');
            features = fscanf(file_id, FORMAT_SPEC);
            fclose(file_id);

            dataset = [dataset; features'];
           
            disp(strcat('#', file_name));
        end

        % Save in a mat file that can be reloaded after
        save(strcat(files_folder, '-dataset.mat'), 'dataset');
    end
    
    toc % Required for measuring time
end