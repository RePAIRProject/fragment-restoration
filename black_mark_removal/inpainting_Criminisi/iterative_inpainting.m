close all;
clear all;

colorTone = [172, 172, 170]; %we will paint the final errorenous regions into this color (approximated fragment region color)
ps = 11; %patch size for the inpainting algorithm
threshold = 254/255;  % threshold to detect synthetically added white regions in a mask



path_input_imgs = '/home/sinem/PycharmProjects/fragment-restoration/Dataset/images_WhiteBG';
path_bm_yolo = '/home/sinem/PycharmProjects/fragment-restoration/Dataset/BOFF/black_mark_region_masked_with_fg';
path_fg = '/home/sinem/PycharmProjects/fragment-restoration/Dataset/fg/';
path_out = ['/home/sinem/PycharmProjects/fragment-restoration/Inpainting/Criminisi/Inpainted_Results/Original/images_bm_masked_inpainted_ps', num2str(ps)];

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

    for j=1:2 % we are doing inpainting two times
        J = inpaintExemplar(rgb_im, BW, 'PatchSize', [ps ps], 'FillOrder', 'tensor');

        % after getting the output of inpainting, create a new inpainting
        % mask by detecting its white-colored regions
        maskedImage = J;
        maskedImage(repmat(fg, [1, 1, 3])) = 0;
        BW = imbinarize(rgb2gray(maskedImage), threshold);
        rgb_im =J;
        %sum_ones = sum(BW(:) == 1)
    end

    % Assign approximated fragment foreground color to regions which still
    % has white colors after two-step inpainting
    inpainted_img = rgb_im;
    inpainted_img(repmat(fg, [1, 1, 3])) = 0;
    BW = imbinarize(rgb2gray(inpainted_img), threshold);


    % Get the linear indices of the pixels where the mask is true
    indices = find(BW);

    % Assign the color tone to the corresponding pixels
    inpainted_img(indices) = colorTone(1);
    inpainted_img(indices + numel(rgb_im(:,:,1))) = colorTone(2);
    inpainted_img(indices + 2*numel(rgb_im(:,:,1))) = colorTone(3);

    imwrite(inpainted_img,fullfile(path_out, filelist2(i).name));
end
