

clear all;

expansionAmount = 0.02;
thr_edgeArea = 50;
thr_perimeter = 35;

set = [12,13,15,4,8];
for m=1:length(set)
    i = set(m);
    close all;
    
    rgb_im = imread(['grbimg_',num2str(i),'.png']);
    im = imread(['timg',num2str(i),'_7_8.png']);
    
    fim=logical(im);
    
    figure
    subplot(1,3,1),    imshow(fim);
    title('Thresholded')
    
    
    SE = strel('square',3);
    BW2 = imdilate(fim,SE);
    % figure; imshow(BW2);
    %BW2 = imdilate(BW2,SE);
    %figure; imshow(BW2);
    BW2 = imdilate(BW2,SE);
    %figure; imshow(BW2);

    BW2=imdilate(BW2,SE);
    BW2=imdilate(BW2,SE);
    BW2=imdilate(BW2,SE);
    BW2=imdilate(BW2,SE);
    BW2=imdilate(BW2,SE);
    
    subplot(1,3,2), imshow(BW2);
    title('Dilation-3')
    %medim = medfilt2(BW2,[10 10]);
    %subplot(1,3,3), imshow(medim);
    %title('Dilation & Median filter')
    
    %     fim2 = bwareaopen(medim,100);
    %     figure; subplot(1,2,1), imshow(medim);
    %     subplot(1,2,2), imshow(fim2);
    %     title('Remove small regions')
    %
    
    BW = BW2;
    cc = bwconncomp(BW);
    % Compute region properties MajorAxisLength and MinorAxisLength
    regionStatistics = regionprops(BW, 'MajorAxisLength', 'MinorAxisLength','Area','Perimeter');
    
    % To discard some edge candidates
    % minAspectRatio = 5;
    % candidateRegions = find(([regionStatistics.MajorAxisLength]./[regionStatistics.MinorAxisLength]) > minAspectRatio);
    
    % Discard edge partititons that are quite small
    candidateRegions = find(([regionStatistics.Area] > thr_edgeArea) & ([regionStatistics.Perimeter] > thr_perimeter));
    
    % Binary image to store the filtered components.
    BW = false(size(BW));
    
    for j = 1:length(candidateRegions)
        BW(cc.PixelIdxList{candidateRegions(j)}) = true;
    end
    figure; imshow(BW)
    

    
    J = inpaintExemplar(rgb_im,BW,'PatchSize',[10 10], 'FillOrder', 'Tensor');
    
    
    imwrite(J,[num2str(i),'_inpainted.png']);
%     imwrite(maskedimg,[imname,'_masked_marks2.png']);
%     imwrite(uint8(BW.*255),[imname,'_mask_marks2.png']);
%     
    
    
    imwrite(uint8(BW)*255,['mask',num2str(i),'_7_8.png']);
    
    %     %% Compute bounding boxes
    %     [expandedBBoxes,coord] = func_computeBBs(BW,expansionAmount);
    %
    %
    %     hold on
    %     %%Plot Bounding Box
    %     for n=1:size(expandedBBoxes,1)
    %         rectangle('Position',expandedBBoxes(n,:),'EdgeColor','r','LineWidth',1)
    %     end
    %     hold off
    %     pause (1)
    %
    %     [xmin, ymin, xmax, ymax] = coord;
    %
    %     textBBoxes = func_mergeBoxes(expandedBBoxes,xmin,ymin,xmax,ymax);
    %
    %
    %
    
    
    
end

