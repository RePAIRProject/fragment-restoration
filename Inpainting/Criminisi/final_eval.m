close all;
clear all;

path_1 = '/MATLAB Drive/Test/first_iter_results'; 
path_2 = '/MATLAB Drive/Test/second_iter_masks';
filelist1 = dir(fullfile(path_1, '*.png')); % List all files with .png extension
filelist2 = dir(fullfile(path_2, '*.png')); % List all files with .png extension
num_images = length(filelist1);

% Sort filenames
[~,idx] = sort_nat({filelist1.name});
filelist1 = filelist1(idx);

[~,idx] = sort_nat({filelist2.name});
filelist2 = filelist2(idx);

for i = 1:num_images
    filename = fullfile(path_1, filelist1(i).name);
    maskname = fullfile(path_2, filelist2(i).name);
    rgb_im = imread(filename);
    BW = imread(maskname);
    % Process the image here
    BW = logical(BW);


    J = inpaintExemplar(rgb_im,BW,'PatchSize',[11 11], 'FillOrder', 'tensor');

    imwrite(J,['/MATLAB Drive/Test/second_iter_results/image_',num2str(i-1),'_inpainted.png']);
end
