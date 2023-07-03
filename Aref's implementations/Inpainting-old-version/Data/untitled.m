set = [1,2,12,13,15,4,8];
    close all;
for m=1:length(set)
    
    i = set(m);

    
    J = imread([num2str(i),'_inpainted.png']);
    RGB = imread(['grbimg_',num2str(i),'.png']);

%         
%     B(:,:,1) = medfilt2(J(:,:,1),[20,20]);
%     B(:,:,2) = medfilt2(J(:,:,2),[20,20]);
%     B(:,:,3) = medfilt2(J(:,:,3),[20,20]);
%     
%     
    LAB = rgb2lab(J);
    L = LAB (:,:,1)/100;
    L = adapthisteq(L,'NumTiles',[8 8],'ClipLimit',0.005);
    LAB(:,:,1) = L*100;
    J2 = lab2rgb(LAB);
    
    %imshowpair(J,J2,'montage')
    
    
%     % Visual enhancement
%     % 1.CLAHE
%     LAB = rgb2lab(J);
%     L = LAB (:,:,1)/100;
%     L = adapthisteq(L,'NumTiles',[8 8],'ClipLimit',0.005);
%     LAB(:,:,1) = L*100;
%     J2 = lab2rgb(LAB);
%     
%     figure
%     imshowpair(RGB,J2,'montage')
%     
%     figure
%     imshowpair(RGB,RGB.* uint8(BW),'montage')
%     
%     


J_lab = rgb2lab(J2);

numColors = 4;
[lout,cmap2] = imsegkmeans(im2single(J_lab),numColors,'NumAttempts',5);

[loutor,cmap2or] = imsegkmeans(im2single(rgb2lab(RGB)),numColors,'NumAttempts',5);

cmap = lab2rgb(cmap2); %imshow(label2rgb(lout))

figure; imshowpair(label2rgb(loutor),label2rgb(lout),'montage')


end
