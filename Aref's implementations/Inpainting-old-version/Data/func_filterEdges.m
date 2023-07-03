function BW = func_filterEdges(BW,area_, perim_)
% BW = bwmorph(BW, 'thin', Inf);

cc = bwconncomp(BW);
% Compute region properties MajorAxisLength and MinorAxisLength
regionStatistics = regionprops(BW, 'MajorAxisLength', 'MinorAxisLength','Area','Perimeter');

% To discard some edge candidates
% minAspectRatio = 5;
% candidateRegions = find(([regionStatistics.MajorAxisLength]./[regionStatistics.MinorAxisLength]) > minAspectRatio);

% Discard edge partititons that are quite small
candidateRegions = find(([regionStatistics.Area] > area_) & ([regionStatistics.Perimeter] > perim_));

% Binary image to store the filtered components.
BW = false(size(BW));

for i = 1:length(candidateRegions)
    BW(cc.PixelIdxList{candidateRegions(i)}) = true;
end


end
