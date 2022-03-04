function analyze(inputImg, segmentationPath, f)

% Define the user thresholds to raise anomaly alarm
orientationThresholds = [-10, 10];
curvatureThresholds = [-1.5e-3, 1.5e-3];
thicknessThresholds = [180, 300];

dRes = 5; % The step resolution to calculate angles
splineStep = 0.10; % The step resolution to calculate splines from interstitial lines. It is the ration between the spline length and the image width
vertexStep = 0.25;
lineStep = 0.20; % The line segment lenght used to analyze orientation/curvature. It is the ration between the spline length and the image width
grainStep = 10;
textureWindowWidthRatio = 0.15625; 

energyDefectRatio = 0.75;
rLimits = [-3e-3,3e-3];
grainSizeLimit = 80;
thicknessRatio = 0.125;%0.33;
interstitialR = 5;  % Recommended value <= 0.25*interstitial_segmentation_diameter

gaussRatio = 0.2; % Proportion of the texture window size to define the Gauss filter size

rgbImage = imread(segmentationPath); % Read the predictions from Python
grayImage = rgbImage(:,:,1); % Be sure that the predictions are single channel
[h,w] = size(grayImage); % Get the input image dimensions
grayImage = bwareafilt(imbinarize(grayImage),[round(w/20)*(4*interstitialR - 1), h*w]); % Get rid of small noise
grayImage = imclose(grayImage,strel('disk',2*interstitialR + 1)); % Close the image to deal with small line discontinuities

skeleton = bwmorph(grayImage > 0.5,'thin', inf); % Threshold image (to have binary values) and get the morphological skeleton
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
roi = imopen(roi, strel('disk',2*interstitialR)); % Separate layers, specially at borders

% % tic
SE = padarray(ones(1,round(lineStep*w)),[round(lineStep*w/2),1],0,'both'); % Create structuring element for the orientation calculation
im_filt = zeros(size(img));
theta = zeros(size(img)); % This variable will save the predicted local orientations
e_d = [;]; % This variable saves the calculated energy e over the whole image using a filter with orientation d
angleAnalysisRange = 90;
for d = -angleAnalysisRange+dRes:dRes:angleAnalysisRange
	ii = imfilter(1*skeleton,imrotate(SE,d,'crop')); % Apply thhe filter with orientation d
    indx = ii>im_filt; % Find the pixels where the filter response is bigger than the previous filter responses
	theta(indx) = d; % Save the current orientation to the pixels found in the previous line
	im_filt = max(ii,im_filt); % Update the maximum response values
    
    e = mean(ii(skeleton == 1)); % Calculate the global energy as a measure of the filter response on the pixels with detected interstitial lines
    e_d = [e_d, [e; d]];
end


[out,idx] = sort(-e_d(1,:));
e_d = e_d(:, idx);
globalOrientation = e_d(2,1); % Get the global orientation as the direction with the highest energy e

% Clean the theta values (edelete borders that may be noisy)
theta = skeleton .* theta;

% Create an HSV image for UI displaying purposes (local orientations)
thetaLines = max(min(theta, orientationThresholds(2)), orientationThresholds(1)); % Crop values to thresholds
thetaLines = (thetaLines - orientationThresholds(1)) / (orientationThresholds(2) - orientationThresholds(1));   % Normalize hsv values to the range [min, max]
SE = strel('line', 2*interstitialR, 90);
thetaLines = imdilate(skeleton.*thetaLines, SE); %  Thicken the lines for visualization
mask = imdilate(skeleton, SE);
theta_hsv = cat(3, thetaLines.*mask, mask, mask);  % Normalize hsv values to the range [min, max]

% Find defects as line segments with orientations having a low value of e
% (75 percent of maximum e)
maxEnergyMask = e_d(1,:) > energyDefectRatio*e_d(1,1);
minD = min(e_d(2,maxEnergyMask));
maxD = max(e_d(2,maxEnergyMask));
defects = imdilate(or(theta < minD, theta > maxD), SE) .* img;
% Clean the detection by connecting very close defects and discarding very
% small ones
defects = imclose(defects, strel('disk', interstitialR));
defects = bwareafilt(imbinarize(defects),[40*(2*interstitialR - 1), h*w]); % From N pixels times a square with side equal to interstitial line width
% Clean the image by making the defects as wide as the interstitial
% segmentation
defects = imdilate(defects, strel('disk',2*interstitialR)) .* img;
% toc

% Create images for UI displaying purposes (interstitial lines and defects)
photo = imread(inputImg);
photoGray = rgb2gray(photo);
photo2 = photo;
photo2(:,:,1) = max(photo2(:,:,1),uint8(255*defects)); % Overlay defects over input image
nonDefectiveLinesMask = uint8(255*uint8(img.*1-defects));
photo2 = max(photo2, cat(3, nonDefectiveLinesMask, nonDefectiveLinesMask, nonDefectiveLinesMask)); % Overlay interstitial lines over input image

% tic
% Calculate layers thicknesses
realDistances = bwdist(skeleton); % Get distance function
distances1 = filter2([0;-1;1],realDistances, 'valid'); % Vertical 1D derivative
disSkeleton = filter2([1;-1;0],sign(distances1), 'valid') >= 1;  % Get skeleton of the distance function
disSkeleton = padarray(disSkeleton,[2,0],0,'both');

filteredSkeleton = disSkeleton.*roi;  % Get only lines inside layers
filteredSkeleton = bwareafilt(logical(filteredSkeleton), [interstitialR^2, h*w]);  % Remove small lines which have low likelihood of being layer centers

% Create image to display
SE = strel('line', 2*interstitialR, 90);
mask = imdilate(filteredSkeleton, SE); % Mask to ignore background
disLines = imdilate(2*filteredSkeleton.*realDistances, SE); % Multiply by 2 to get diameter instead of radius; dilate to have thick line to display
disLines = max(min(disLines, thicknessThresholds(2)), thicknessThresholds(1)); % Crop values to thresholds
% The min is used because, if any value exceeds the thicknessRatio * h
% value, the display shows black regions
dHsv = cat(3,mask.*(disLines - thicknessThresholds(1)) / (thicknessThresholds(2) - thicknessThresholds(1)), mask, mask);  % Normalize hsv values to the range [min, max]
% toc

% tic
% Calculate local curvatures
fSkeleton = filter2([1;-1], skeleton) .* skeleton; % Delete joint pixels between independent lines
fSkeleton = ~imbinarize(filter2(ones(5,1), fSkeleton), 1) .* fSkeleton;
[sk_lab,NN] = bwlabel(fSkeleton); % Separate and label each interstitial line
splineStep = round(splineStep * w);
vertexStep = round(vertexStep * w);
dd = zeros(size(img));
rMap = zeros(size(img), 'double'); % Local curvatures will be stored here
% there are NN interface lines found in the image
for ii = 1:NN
    try
        indx=find(bwmorph(sk_lab==ii,'endpoints')); % Find pixels of the current line
%         if ~length(indx) > splineStep % Skip line if it is very short
%             continue
%         end
        [endy,endx]=find(bwmorph(sk_lab==ii,'endpoints'));
        % Measure the distance of pixels to the extreme points of the line
        M = zeros(size(img),'logical');
        M(indx(1))=true;
        dd_ii = bwdistgeodesic(sk_lab==ii,M);
        dd = max(dd,dd_ii);
        pts = [];
        % Get points to calculate the splines
        maxlen = max(max(dd_ii));
        points = 0:splineStep:maxlen;
        if ~(points(end) == maxlen)
            points = [points maxlen];
        end
        for jj = points
            [yy,xx] = find(dd_ii==jj);
            pts = [pts, [yy(1);xx(1)]];
        end
        [e, n] = size(pts);
        
        % At this place, the variable 'pts' contains the coordinates of equidistant 
        % nodes of an interface line. The nodes are spaced by 'step' pixels. 
        % The nodes can now be used to approximate the lines (by splines) and
        % compute the curvature on.
        % pp=spline(pts(2,:),pts(1,:),endx(1):(endx(end)+1));
        %pp = pp(1:length(pp)-1); % version
        % with replicated last point
        pp=spline(pts(2,:),pts(1,:),endx(1):endx(end));
        if length(pp) < 3 % If the spline is too short, ignore it
            s = 'Element ignored.';
            continue
        end
        
        pp1 = diff(pp);
        pp2 = diff(pp1);
        
        % Use calculus to get the inflection points from the polynomical
        % approximations
        iXs = find(diff(pp2./abs(pp2)) ~= 0) + 1;

        iXs = [1, iXs, length(pp)];
        % Clean consecutive points since they are not very informative
        iXs = [1, iXs(find(diff(iXs) > 1)), length(pp)];
        
        if iXs(end) == iXs(end-1) || iXs(end)-iXs(end-1) == 1
            iXs = iXs(1:end-1);
        end
        
        % Get values between each pair of inflection points to calculate
        % the curvature there
        centers = zeros(1, length(iXs) - 1);
        for i=2:length(iXs)
            centers(i-1) = round(mean(iXs(i-1:i)));
        end
        cXs = sort([iXs centers]);
            
        % In case of short lines with not enough inflection points, delete
        % duplicates
        cXs = cXs([1 , find(diff(cXs) ~= 0) + 1]);
        cYs = pp(cXs + 1 - cXs(1)); 
        
         
        vertices = [cYs' cXs'];
        lines = [(1:size(vertices,1)-1)' (2:size(vertices,1))'];
        k = LineCurvature2D(vertices, lines); % Calculate curvatures
        
        % Save curvatures in an image
        curvatures = zeros(1,cXs(end));
        for v=2:2:length(cXs)-1
            curvatures(cXs(v-1):cXs(v+1)) = k(v);
        end
        z = sub2ind(size(img), round(pp), endx(1):endx(end));
        rMap(z) = curvatures;
    catch
        s = 'Element ignored.';
    end
end

% Create image for UI displaying purposes (local radii)
mask = imdilate(double(rMap ~= 0), strel('line', 2*interstitialR, 90));

% Dilate lines with negative curvature
negativeR = rMap;
negativeR(find(rMap) > 0) = 0;
negativeR = -imdilate(-negativeR, strel('line', 2*interstitialR, 90));

% Dilate lines with positive curvature
positiveR = rMap;
positiveR(find(rMap) < 0) = 0;
positiveR = imdilate(positiveR, strel('line', 2*interstitialR, 90));

curLines = positiveR + negativeR;  % Combine lines with positive and negative curvature
curLines = max(min(curLines, curvatureThresholds(2)), curvatureThresholds(1)); % Crop values to thresholds

rHsv = cat(3,(curLines - curvatureThresholds(1))/(curvatureThresholds(2) - curvatureThresholds(1)), mask, mask);
% toc

%%%
% Texture features
[roi_lab,MM] = bwlabel(roi);
outputPath = fullfile("texture", "images");
mkdir(outputPath);
[filepath, name, ext] = fileparts(inputImg);
idx = 1;

windowWidth = round(textureWindowWidthRatio * w);
leveledImage = double(photoGray)-imgaussfilt(double(photoGray),round(windowWidth*gaussRatio));
leveledImage = uint8(leveledImage-min(leveledImage(:)));
windowNames = [];
textCoordinates = [];
borderMap = zeros(size(img),'logical');
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
        [y0,x0] = find(dd_ii==0); x = x0(1); % Find first pixel in layer center line
        [yF,xF] = find(dd_ii==(maxLen)); xF = xF(1); % Find first pixel in layer center line
      
        while true
            % Create current window
            startX = x;  % Left limit of the window
            endX = min(w, x + (windowWidth - 1)); % Right limit; don't surpass image width
            
            bb = regionprops(layerDouble(:, startX:endX) >= 0,'BoundingBox').BoundingBox; % Get bounding box of the layer within the startX and endX
            startY = floor(bb(2));
            endY = startY + bb(4);
            
            smallWindow = layerDouble(startY:endY, startX:endX); % Window to analyze
            smallWindowSize = size(smallWindow);
            nonNanPixels = sum(sum(smallWindow >= 0));

            if nonNanPixels > 0.50*smallWindowSize(1)*windowWidth % Ignore windows with very few texture pixels according to the expected window size
                
                % Save image to disk so it can be read by Python
                finalWindow = smallWindow;
                finalWindow(isnan(finalWindow)) = 0.0;
                windowName = name + "-" + sprintf( '%03d', idx ) + ext;
                imwrite(finalWindow, fullfile(outputPath, windowName), 'Compression', 'none');
                
                % Save image info to show the predictions in the UI
                windowNames = [windowNames; windowName];
                textCoordinates = [textCoordinates; startX+left+interstitialR, startY+0.5*smallWindowSize(1)];   
                % To illustrate the paper only
                %rectangle('Position',[startX,startY,endX-startX,endY-startY],'LineWidth',10, 'EdgeColor', 'g')
                %
                % Find analyzed texture borders
                windowMask = zeros(size(img), 'logical');
                windowMask(startY:endY, startX:endX) = 1;
                windowMask = windowMask &  layerMask;
                borderMap = max(borderMap, windowMask - imerode(windowMask, strel('disk', interstitialR)));
                
                idx = idx + 1;
            end
            x = x + windowWidth; % Move the beggining of the sliding window
            if x > xF
                break
            end
        end
    end
end
fileID = fopen(fullfile(outputPath, "matlab_flag"),'w'); % Tell python that texture predictions are needed
fprintf(fileID,'%s', '');
fclose(fileID);

while true % Wait until texture predictions are finished by Python
    if isfile(fullfile(outputPath, "python_flag"))
        break
    end
end
textureTable = readtable(fullfile(outputPath, "outputs.csv"), 'ReadVariableNames', true, 'ReadRowNames', true); % Read texture predictions
[status, message, messageid] = rmdir(outputPath, 's');  % Delete temporal files
borderMap = uint8(255*borderMap);  % Change from boolean to uint8

%%%


% Display all the results
% tic
clf(f,'reset')
set(gcf,'name',inputImg,'numbertitle','off')

subplot(2,3,3)
imagesc(photo2), hold on
title({[strcat('Global orientation:', {' '}, string(globalOrientation), '°', {' '}, 'Regularity: [', string(minD), ',', string(maxD), ']°')], ['Normal regions in white'], ['Abnormal regions in red']});

% Display texture predictions
h0 = subplot(2,3,6);
imagesc(max(photo, cat(3, borderMap, borderMap, borderMap)))
title({["Texture classification"], ['(per region)']})
labels = {'fluid', 'good','shark_skin','tearing'};
colors = {'red', 'cyan', 'magenta', 'red'};
colorDict = containers.Map(labels, colors);
for i=1:length(windowNames)
   window = windowNames(i);
   probs = table2array(textureTable(window, :));
   prob = max(probs);
   labelIdx = find(probs == prob);
   label = textureTable.Properties.VariableNames{labelIdx(1)};
   color = colorDict(label);
   label = strrep(label, '_', '-\n');
   text(textCoordinates(i, 1), textCoordinates(i, 2), sprintf("%s:\n%0.2f", sprintf(label), prob), 'Color', color);
end

h1 = subplot(2,3,2);
imagesc(hsv2rgb(theta_hsv)); caxis(orientationThresholds); colormap hsv;
title("Local orientations (degrees)")
originalSize = get(gca, 'Position');
colorbar;
set(h1, 'Position', originalSize);


subplot(2,3,1)
imagesc(photo), hold on
title("Input image")

h2 = subplot(2,3,5);
dRgb = hsv2rgb(dHsv);
lines = im2double(img);
dRgb = max(dRgb, cat(3, lines, lines, lines));
ax = gca;imagesc(dRgb); caxis(ax, thicknessThresholds); colormap hsv;
title("Local width (pixels)")
originalSize = get(gca, 'Position');
colorbar;
set(h2, 'Position', originalSize);

h3 = subplot(2,3,4);
ax = gca;imagesc(hsv2rgb(rHsv)); caxis(ax, curvatureThresholds); colormap hsv;
title("Local curvatures (pixels^{-1})")
originalSize = get(gca, 'Position');
colorbar;
set(h3, 'Position', originalSize);

% set(h0, 'Colormap', jet);
% toc

end

function weights = normalEquation(X,Y)
    weights = (X'*X)\X'*Y;
end

function thicknesses = getLayersThickness(x, distances, skeleton, disSkeleton)

centersOfLayers = find(disSkeleton(:, x) > 0);
interlayerYs = find(skeleton(:, x) > 0);

thicknesses = [;];

for i=1:length(interlayerYs) - 1
    for j=1:length(centersOfLayers)
        if centersOfLayers(j) > interlayerYs(i) && centersOfLayers(j) < interlayerYs(i+1)
            thicknesses = [thicknesses [centersOfLayers(j); 2*distances(centersOfLayers(j), x); interlayerYs(i); interlayerYs(i+1)]];
            break
        end
    end
end

end