photoDir = "/media/shared_storage/datasets/my_photos/Sep21/Sep21_pruned";
segmentationDir = "/media/shared_storage/datasets/my_photos/Sep21/texture_defects_line_segmentation";
outputDir = "/media/shared_storage/datasets/my_photos/Sep21/texture_defects-windows_test";

listing = dir(segmentationDir);

for i=1:length(listing)
   if endsWith(listing(i).name, ".png") && ~startsWith(listing(i).name,".")
       inputImg = fullfile(photoDir, strrep(listing(i).name, ".png", ".tiff"));
       segmentationPath = fullfile(segmentationDir, listing(i).name);
       saveImagewindows(inputImg, segmentationPath, outputDir);
   end
end

function saveImagewindows(inputImg, segmentationPath, outputPath)

%%%
gaussRatio = 0.2; % Proportion of the texture window size to define the Gauss filter size
%%%

if ~exist(outputPath, 'dir')
   mkdir(outputPath)
end
idx = 1;
[filepath, name, ext] = fileparts(inputImg);

textureWindowWidthRatio = 0.15625; 
interstitialR = 5;

photo = imread(inputImg);
f = figure('visible','off');
imshow(photo, 'border', 'tight' ); %//show your image
hold on;
photoGray = rgb2gray(photo);

rgbImage = imread(segmentationPath); % Read the predictions from Python
grayImage = rgbImage(:,:,1); % Be sure that the predictions are single channel
[h,w] = size(grayImage); % Get the input image dimensions
grayImage = imopen(grayImage,strel('disk',max(1, interstitialR - 2))); % Open the image to get rid of small noise
grayImage = imclose(grayImage,strel('disk',interstitialR + 2)); % Close the image to deal with small line discontinuities

skeleton = bwmorph(grayImage > 128,'thin', inf); % Threshold image (to have binary values) and get the morphological skeleton
img = im2double(imdilate(skeleton,strel('disk',interstitialR))); % Dilate the skeleton to have fixed thick interstitial lines

% Get ROI (based on predicted inter-layer regions)
lineBorders = filter2([-1;1], img);
roi = zeros(h,w, 'logical');
for j=1:w
    column = lineBorders(:, j);
    borderPoints = find(column ~= 0);
    pairs = [];
    k=1;
    while k <= length(borderPoints)
        if column(borderPoints(k)) == -1
            for l=k+1:length(borderPoints)
                if column(borderPoints(l)) == 1
                    pairs = [pairs; borderPoints(k), borderPoints(l)];
                    k = l;
                    break
                end
            end
        end
        k = k+1;
    end
    [n_pairs, null] = size(pairs);
    for m=1:n_pairs
        roi(pairs(m, 1):pairs(m,2), j) = 1;
    end   
end
roi = imopen(roi, strel('disk',interstitialR)); % Separate layers, specially at borders

eroImg = imerode(img, strel('disk', 1));

% Calculate layers thicknesses
realDistances = bwdist(skeleton); % Get distance function
distances1 = filter2([0;-1;1],realDistances, 'valid'); % Vertical 1D derivative
disSkeleton = filter2([1;-1;0],sign(distances1), 'valid') >= 1;  % Get skeleton of the distance function
disSkeleton = padarray(disSkeleton,[2,0],0,'both');

filteredSkeleton = disSkeleton.*roi;
filteredSkeleton = bwareafilt(logical(filteredSkeleton), [interstitialR^2, h*w]);

SE = strel('disk', interstitialR);
disLines = imdilate(filteredSkeleton, SE);

[roi_lab,MM] = bwlabel(roi);

windowWidth = round(textureWindowWidthRatio * w);
leveledImage = double(photoGray)-imgaussfilt(double(photoGray),round(windowWidth*gaussRatio));
leveledImage = uint8(leveledImage-min(leveledImage(:)));
for ii = 1:MM
    layerMask = uint8(roi_lab == ii);
    layerSkeleton = uint8(filteredSkeleton).*layerMask;  
    layer = photoGray.*layerMask;
    layerDouble = im2double(leveledImage);
    layerDouble(layerMask == 0) = NaN;
    bb = regionprops(layerMask,'BoundingBox').BoundingBox;
    left = bb(1); 
    top = bb(2);
    regionW = bb(3);
    regionH = bb(4);
    x_lims = [ceil(left), floor(left + regionW)];
    y_lims = [ceil(top), floor(top + regionH)];
    layerRoi = layer(y_lims(1):y_lims(2), x_lims(1):x_lims(2));
    roiLayerMask = layerMask(y_lims(1):y_lims(2), x_lims(1):x_lims(2));
    layerMask = double(layerMask);
    [sk_lab,NN] = bwlabel(layerSkeleton); % Find and label the lines between interstitial lines
    for jj = 1:NN
        indx=find(bwmorph(sk_lab==jj,'endpoints'));
        M = zeros(size(img),'logical');
        M(indx(1))=true;
        dd_ii = bwdistgeodesic(sk_lab==jj,M);
        maxLen = max(dd_ii(:));
        origin = 0; % Begin sliding window at pixel 0
        while true
            [yy,xx] = find(dd_ii==origin); % Find current pixel in line
            windowHeight= double(round(3*realDistances(yy(1), xx(1)))); % Determine the height of the window with respect to the local thickness of the layer. Make it larger to compensate for non-horizontal layers
            % Create current window
            startY = max(1, yy(1) - floor(windowHeight / 2));
            endY = min(h, yy(1) + floor(windowHeight / 2));
            startX = xx(1);
            endX = min(w, xx(1) + (windowWidth - 1));
            smallWindow = layerDouble(startY:endY, startX:endX);
            nonNanPixels = sum(sum(smallWindow >= 0));
            if nonNanPixels > 0.5*windowHeight*windowWidth
                bb = regionprops(smallWindow >= 0,'BoundingBox').BoundingBox;
                left = bb(1); 
                top = bb(2);
                regionW = bb(3);
                regionH = bb(4);
                x_lims = [ceil(left), floor(left + regionW)];
                y_lims = [ceil(top), floor(top + regionH)];
                finalWindow = smallWindow(y_lims(1):y_lims(2), x_lims(1):x_lims(2));
                imwrite(finalWindow, fullfile(outputPath, name + "-" + idx + ext), 'Compression', 'none');
                
                rectangle('Position', [startX+left startY+top  regionW regionH], 'EdgeColor', 'white'); %// draw rectangle on photo
                text(startX+left+0.5*regionW, startY+top+0.5*regionH, num2str(idx), 'Color', 'white');
                idx = idx + 1;
            end
            origin = origin + windowWidth; % Move the beggining of the sliding window
            if origin > maxLen
                break
            end
        end
    end
end

frm = getframe(f); %// get the image+rectangles
imwrite(frm.cdata, fullfile(outputPath, name + ext), 'Compression', 'none'); %// save to file
imwrite(leveledImage, fullfile(outputPath, name + "-0_leveled" + ext))
clf(f,'reset')
close all
end
