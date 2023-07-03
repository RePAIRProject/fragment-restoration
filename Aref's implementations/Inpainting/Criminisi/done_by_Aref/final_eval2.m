close all;
clear all;

colorTone = [172, 172, 170]; %we will paint the final errorenous regions into this color (approximated fragment region color)


path_input_imgs = '/home/sinem/PycharmProjects/fragment-restoration/Dataset/images_WhiteBG';
path_bm_yolo = '/home/sinem/PycharmProjects/fragment-restoration/Dataset/BOFF/black_mark_region_masked_with_fg';
path_fg = '/home/sinem/PycharmProjects/fragment-restoration/Dataset/fg/';
path_out = '/home/sinem/PycharmProjects/fragment-restoration/Inpainting/Criminisi/Inpainted_Results/Original/Inpainted_first_iter_bm_masked_ps11';

mkdir(path_out);

filelist1 = dir(fullfile(path_input_imgs, '*.png')); % List all files with .png extension
filelist2 = dir(fullfile(path_bm_yolo, '*.png')); % List all files with .png extension
filelist_fg = dir(fullfile(path_fg, '*.png')); % List all files with .png extension
num_images = length(filelist1);

% Sort filenames
[~,idx] = sort_nat({filelist1.name});
filelist1 = filelist1(idx);

[~,idx] = sort_nat({filelist2.name});
filelist2 = filelist2(idx);

[~,idx] = sort_nat({filelist_fg.name});
filelist_fg = filelist_fg(idx);

for i = 1:num_images

    filename = fullfile(path_input_imgs, filelist1(i).name);
    bm_mask = fullfile(path_bm_yolo, filelist2(i).name);

    rgb_im = imread(filename);
    fg = ~logical(rgb2gray(imread(fullfile(path_fg, filelist_fg(i).name))));
    BW = logical(rgb2gray(imread(bm_mask))); 


    % Get the linear indices of the pixels where the mask is true
    indices = find(fg);
    maskedImage = rgb_im;
    % Assign the color tone to the corresponding pixels
    maskedImage(indices) = colorTone(1);
    maskedImage(indices + numel(rgb_im(:,:,1))) = colorTone(2);
    maskedImage(indices + 2*numel(rgb_im(:,:,1))) = colorTone(3);

    rgb_im = maskedImage;

    J = inpaintExemplar(rgb_im,BW,'PatchSize',[9 9], 'FillOrder', 'tensor');

    maskedImage = J;
    maskedImage(repmat(fg, [1, 1, 3])) = 0;


    imwrite(maskedImage,fullfile(path_out, filelist2(i).name));
end
