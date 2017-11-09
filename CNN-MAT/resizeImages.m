function [] = resizeImages(path, size)
%Creates a matrix with the images for each label inside a 'path' folder.
    
    tic % Required for measuring time
    
    % Create new destination folder with resized images
    new_labels_folder = strcat(path, int2str(size));
    mkdir(new_labels_folder);
    
    % Get the files' names
    labels_folders = dir(path);
    labels_folders = labels_folders(3:end); % Avoid '.' and '..'
    
    % For each label/class in the dataset
    for label = 1 : length(labels_folders)
        new_files_folder = strcat(new_labels_folder, '/', int2str(label));
        mkdir(new_files_folder);
        
        files_folder = strcat(path, '/', labels_folders(label).name);
        files = dir(files_folder);
        files = files(3:end-1); % Avoid the CSV file
       
        disp('************************');
        disp(files_folder);
            
        % Read files data, resize them and store in the new folder
        for file = 1 : length(files)
            file_name = files(file).name;
            file_path = strcat(files_folder, '/', file_name);
            image_original = imread(file_path);
            image_resized = imresize3(image_original, [size size 3]);
            imwrite(image_resized, strcat(new_files_folder, '/', file_name));
           
            disp(strcat('#', file_path));
        end
    end
    
    toc % Required for measuring time
end

