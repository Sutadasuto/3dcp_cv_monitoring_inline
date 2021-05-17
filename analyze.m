function analyze(inputImg, segmentationPath, f)

dRes = 2; % The step resolution to calculate angles
splineStep = 20; % The step resolution to calculate splines from interstitial lines

tic
rgbImage = imread(segmentationPath); % Read the predictions from Python
grayImage = rgbImage(:,:,1); % Be sure that the predictions are single channel

skeleton = bwmorph(grayImage > 128,'thin', inf); % Threshold image (to have binary values) and get the morphological skeleton

img = im2double(imdilate(skeleton,strel('disk',2))); % Dilate the skeleton to have fixed thick interstitial lines



SE = padarray(ones(5,20),[12,5],0,'both'); % Create structuring element for the orientation calculation
im_filt = zeros(size(img));
theta = zeros(size(img)); % This variable will save the predicted local orientations
e_d = [;]; % This variable saves the calculated energy e over the whole image using a filter with orientation d
for d = -90+dRes:dRes:90
	ii = imfilter(img,imrotate(SE,d,'crop')); % Apply thhe filter with orientation d
    indx = ii>im_filt; % Find the pixels where the filter response is bigger than the previous filter responses
	theta(indx) = d; % Save the current orientation to the pixels found in the previous line
	im_filt = max(ii,im_filt); % Update the maximum response values
    
    e = mean(ii(img > 0.0)); % Calculate the global energy as a measure of the filter response on the pixels with detected interstitial lines
    e_d = [e_d, [e; d]];
end


[out,idx] = sort(-e_d(1,:));
e_d = e_d(:, idx);
globalOrientation = e_d(2,1); % Get the global orientation as the direction with the highest energy e

eroImg = imerode(img, strel('disk', 1));

% Create an HSV image for UI displaying purposes (local orientations)
theta_hsv = cat(3,(theta + 90)/180, eroImg, eroImg);


theta = eroImg .* theta; % Clean the theta values

[h,w] = size(grayImage);

% Find defects as line segments with orientations having a low value of e
% (75 percent of maximum e)
maxEnergyMask = e_d(1,:) > 0.75*e_d(1,1);
minD = min(e_d(2,maxEnergyMask));
maxD = max(e_d(2,maxEnergyMask));
defects = or(theta < minD, theta > maxD) .* eroImg;
% Clean the detection by connecting very close defects and discarding very
% small ones
r = 4;
defects = imclose(defects, strel('disk', r));
defects = bwareafilt(imbinarize(defects),[(2*r + 1)^2, h*w]); 


% Create images for UI displaying purposes (interstitial lines and defects)
photo = imread(inputImg);
photoGray = rgb2gray(photo);
photo2 = photo;
photo2(:,:,1) = max(photo2(:,:,1),uint8(255*defects)); % Overlay defects over input image
photo2(:,:,2) = max(photo2(:,:,2),uint8(255*grayImage.*uint8(1-defects))); % Overlay interstitial lines over input image

% Calculate layers thicknesses
realDistances = bwdist(skeleton); % Get distance function
distances = bwdist(grayImage > 128);
disSkeleton = bwmorph(distances,'skel',Inf); % Get skeleton of the distance function
% thicknesses = getLayersThickness(round(w/2), realDistances, skeleton, disSkeleton);


% Calculate local curvatures
[sk_lab,NN] = bwlabel(skeleton); % Separate and label each interstitial line
dd = zeros(size(img));
rMap = zeros(size(img), 'double'); % Local curvatures will be stored here
% there are NN interface lines found in the image
for ii = 1:NN
    try
        indx=find(bwmorph(sk_lab==ii,'endpoints')); % Find pixels of the current line
        if ~length(indx) > splineStep % Skip line if it is very short
            continue
        end
        [endy,endx]=find(bwmorph(sk_lab==ii,'endpoints'));
        % Measure the distance of pixels to the extreme points of the line
        M = zeros(size(img),'logical');
        M(indx(1))=true;
        dd_ii = bwdistgeodesic(sk_lab==ii,M);
        dd = max(dd,dd_ii);
        pts = [];
        % Get points to calculate the splines
        maxlen = max(max(dd_ii));
        for jj = 0:splineStep:maxlen
            [yy,xx] = find(dd_ii==jj);
            pts = [pts, [yy(1);xx(1)]];
            % Get points inside a spline to calculate the curvature
            if jj == 0
                cXs = [xx(1)];
            else
                cXs = [cXs round(mean([cXs(end), xx(1)])) xx(1)];
            end
        end
        [e, n] = size(pts);
        try
            pts = [pts, [pts(1, n-1);pts(2, n)+splineStep]]; % Replicate the last point to have a cleaner ending of line in the spline calculation
        catch
            continue
        end
        % At this place, the variable 'pts' contains the coordinates of equidistant 
        % nodes of an interface line. The nodes are spaced by 'step' pixels. 
        % The nodes can now be used to approximate the lines (by splines) and
        % compute the curvature on.
        pp=spline(pts(2,:),pts(1,:),endx(1):(endx(end)+1));
        pp = pp(1:length(pp)-1);
        if length(pp) < 3 % If the spline is too short, ignore it
            s = 'Element ignored.';
            continue
        end
        
        % Calculate the vertices and edges of the chosen points
        cYs = pp(cXs + 1 - cXs(1));  
        vertices = [cYs' cXs'];
        lines = [(1:size(vertices,1)-1)' (2:size(vertices,1))'];
        k = LineCurvature2D(vertices, lines); % Calculate curvatures
        % Save curvatures in an image
        curvatures = zeros(1,w);
        lastPixel = cXs(1)-1;
        for v=1:length(k)
            prevPixel = lastPixel + 1;         
            if v == length(k)
                lastPixel = endx(end);
            else
                lastPixel = ceil(mean([cXs(v), cXs(v+1)]));
            end            
            curvatures(prevPixel:lastPixel) = k(v);  
        end
        z = sub2ind(size(img), round(pp), endx(1):endx(end));
        rMap(z) = curvatures(endx(1):endx(end));
    catch
        s = 'Element ignored.';
    end
end

% Create image for UI displaying purposes (local radii)
mask = imdilate(double(rMap ~= 0), strel('line', 5, 90));
maxR = max(abs(min(rMap(:))), abs(min(rMap(:))));
rHsv = cat(3,imdilate((rMap + maxR)/(2*maxR), strel('line', 5, 90)), mask, mask);

% Create image for UI displaying purposes (local layer thicknesses)
filteredSkeleton = disSkeleton.*imclose(imbinarize(imfilter(im2double(disSkeleton),imrotate(SE,globalOrientation,'crop')),10), strel('disk', 5));
SE = strel('line', 3, 90);
disLines = imdilate(filteredSkeleton, SE);
mask = max(im2double(grayImage), disLines);
dHsv = cat(3,disLines.*imdilate(2*(disSkeleton.*realDistances)/(round(h/4)), SE), mask, mask);
toc

%%%
% Texture features
grainSize = zeros(size(grayImage)); % Local grain size modes per region will be saved here
[sk_lab,NN] = bwlabel(filteredSkeleton); % Find and label the lines between interstitial lines
WtoHractor = 2;

% Analyze the image with sliding windows to avoid interstitial lines (thus the only pixels to analyze are actually a texture)
for ii = 1:NN
    try
        indx=find(bwmorph(sk_lab==ii,'endpoints'));
        if ~length(indx) > splineStep
            continue
        end
        M = zeros(size(img),'logical');
        M(indx(1))=true;
        dd_ii = bwdistgeodesic(sk_lab==ii,M);
        maxLen = max(dd_ii(:));
        origin = 0; % Begin sliding window at pixel 0
        while true
            [yy,xx] = find(dd_ii==origin); % Find current pixel in line
            windowSize = double(round(realDistances(yy(1), xx(1)))); % Determine the size of the window with respect to the local thickness of the layer
            % Create current window
            startY = yy(1) - floor(windowSize / 2);
            endY = yy(1) + floor(windowSize / 2);
            startX = xx(1);
            endX = xx(1) + WtoHractor*(endY - startY + 1) - 1;
            if ~(endX > w) && ~(startY < 1) && ~(endY > h) && windowSize > 1
%                 rectangle('Position',[startX,startY,endX-startX,endY-startY],'LineWidth',1, 'EdgeColor', 'g')
%                 pause(0.1)
                roi = photoGray(startY:endY, startX:endX); % Get region of interest using the calculated window
                radiusRange = 0:floor(windowSize / 2); % Get grain radius from one to half the layer thickness
                intensity_area = zeros(size(radiusRange));
                % Obtain granulometry of the region of interest
                for counter = radiusRange
                    %imclose behaves weird. The cardinality of imclose(img,
                    %se_size=k+1) is not always greater than the cardinality
                    %of imclose(img, se_size=k)
                    % remain = imclose(roi, strel('disk', counter));
                    remain = imerode(imdilate(roi, strel('disk', counter)), strel('disk', counter));
                    difference = remain - roi;
                    intensity_area(counter + 1) = sum(difference(:));  
                end
                intensity_area_prime = diff(intensity_area);
                m = find(intensity_area_prime == max(intensity_area_prime)); % Mode of grain size
                grainSize(startY:endY, startX:endX) = m(1); % Save grain size in image
            end
            origin = origin + max(1, WtoHractor*(endY - startY)-1); % Move the beggining of the sliding window
            if origin > maxLen
                break
            end
        end
    catch e
        disp("Line ignored")
        disp(e.message)
    end
end
toc

% Local binary pattern transform
% radiusRange = 1:5;
% tic
% lbps = zeros(h, w, length(radiusRange));
% for r = radiusRange
%    lbps(:,:,r) = LBP(photoGray,1);
% end
% toc

% tic
% %co-occurrence matrix
% glcm = graycomatrix(photoGray, 'Offset', [0,1]);
% toc

%%%


% Display all the results
tic
% clf(f);
set(gcf,'name',inputImg,'numbertitle','off')

subplot(2,3,1)
imagesc(photo2), hold on
title({[strcat('Global orientation:', {' '}, string(globalOrientation), '°', {' '}, '[', string(minD), ',', string(maxD), ']°')], ['Layer thickness in green'], ['Anomalies in red']});
% [e, n] = size(thicknesses);
% for i =1:n
%     text(round(w/2), double(thicknesses(1, i)), strcat(string(round(thicknesses(2, i), 1)), {' ' }, 'pixels'),'Color','green','FontWeight','bold','HorizontalAlignment','center')
%     plot([round(w/2), round(w/2)],thicknesses(3:4,i), 'r+', 'MarkerSize', 3, 'LineWidth', 1, 'Color','green');
% end


h0 = subplot(2,3,2);
imagesc(grainSize), hold on; colormap hot;
title({["Grain size"], ['(mode per region)']})
originalSize = get(gca, 'Position');
colorbar;
set(h0, 'Position', originalSize);


h1 = subplot(2,3,3);
imagesc(hsv2rgb(theta_hsv)); caxis([-90+dRes,90]); colormap hsv;
title("Local orientations")
originalSize = get(gca, 'Position');
colorbar;
set(h1, 'Position', originalSize);


subplot(2,3,4)
imagesc(photo), hold on
title("Input image")

h2 = subplot(2,3,5);
ax = gca;imagesc(hsv2rgb(dHsv)); caxis(ax,[0,round(h/4)]); colormap hsv;
title("Local thickness")
originalSize = get(gca, 'Position');
colorbar;
set(h2, 'Position', originalSize);

h3 = subplot(2,3,6);
ax = gca;imagesc(hsv2rgb(rHsv)); caxis(ax,[-maxR,maxR]); colormap hsv;
title("Local curvatures")
originalSize = get(gca, 'Position');
colorbar;
set(h3, 'Position', originalSize);

set(h0, 'Colormap', hot);
toc

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