# Tool for 3D printed concrete analysis
Tool for real-time monitoring of concrete 3D printing. It combines a fully-convolutional neural network coded in Python (3.7.8) using Tensorflow (2.1.0) for feature extraction (detection of interstitial lines), and image processing techniques coded in Matlab (R2020b) for measurement and visualization.

![alt text](https://github.com/Sutadasuto/concrete_analysis/blob/main/output_example.png?raw=true)

## Requirements

### Python
The current tool was developed and tested using Python 3.7.8, opencv 4.4.0 and Tensorflow 2.1.0. For commodity, a conda environment.yml file is provided to replicate the environment.

### Matlab
The current tool was developed and tested using Matlab R2020b with the Image Processing Version 11.2.

## How to run

Matlab and Python communicate using TCP/IP sockets through local host. You can change the port at the "config" file (the number after 'localhost, ').

Similarly, in the same file you can choose the video source. You can choose a video input device (an integer number; usually 0 if using an integrated webcam) or a saved video file (a string; e.g. /path/to/video.MOV). Finally, you can and must specify the path to a file containing the pre-trained weights of the fully-convolutional neural network.

Next we share links to both a sample input video and the current neural network weights.

* Link video: https://drive.google.com/file/d/1EfY37ipjub7DKIei8upSQ5h3AwislnjK/view?usp=sharing
* Link weights: https://drive.google.com/file/d/1qHm2XbXgazFARGi8-UbHfI19mBFNlv1q/view?usp=sharing

To run the tool, you must first begin the python process by running:
```
python main.py
```

This script has two optional arguments: 
* 'max_res' allows you to resize input images so that the image width is reduced to this (integer) number. The default value is 1280 (high-resolution).
* 'gpu' allows you to choose if you want to use a GPU for feature extraction (specially useful for machines without an integrated GPU). The default value is True (use False to disable GPU use).

To run the tool, for example, with a maximum horizontal resolution of 512 pixels using CPU instead of GPU you should run:
```
python main.py --max_res 512 --gpu False
```

Let the program run until the console prints "Waiting for a connection". This means Python is set up and ready to communicate with Matlab to begin the analysis.

In Matlab, you just need to run "main.m". Once Matlab has communicated with Python, the Matlab console will show a tcpclient variable and the Python console will print "Connection ready. Start!".

The expected output should like the above image. The tool will keep running until one of the next finishing conditions is met:

* (If analyzing a stored video) Python has read the last frame from the video
* An acquisition error occurred
* The user closes the analysis user interface

Notice that each time that Python is required to process an image, both the image and the output of the neural network are saved at "tmp". Below, there is an example of input image and its corresponding output:

![alt text](https://github.com/Sutadasuto/concrete_analysis/blob/main/tmp/img.jpg?raw=true)
![alt text](https://github.com/Sutadasuto/concrete_analysis/blob/main/tmp/img_gt.png?raw=true)

## User interface

The Matlab figure contains 6 subplots distributed in 2 rows and 3 columns. Each subplot shows measurements derived from the detected interstitial lines (i.e. lines separating two printed layers).

These calculations are done by the function "analyze.m"; any variable mentioned in this section is accessible from within that function.

### Subplot 1,1
This plot shows the input image along with the interstitial lines detected by the neural network in green, and detected anomalies in red. The title shows the estimated global orientation of the interstitial lines.
* Interstitial lines are stored in the variable 'grayImage'
* Errors are detected as interstitial line segments with a local orientation far from the estimated global orientation (see 'Subplot 1,3' for more info on global and local orientation). These defects are stored in the variable 'defects' 
* Global orientation is stored in the variable 'globalOrientation'

### Subplot 1,2
This plot shows the local granulometry (https://fr.mathworks.com/help/images/granulometry-of-snowflakes.html) in subregions contained between interstitial lines. Per region, only the most frequent grain size (the mode) is displayed.
* The mode of grain size per region is stored in the variable 'grainSize' (consider that any region that is not part of an interstitial line has a default value of 0)

### Subplot 1,3
This plot shows the local orientation of the different interstitial line segments. The orientation is measured as an angle using a conventional XY coordinate system with origin at the bottom left corner of the image.

The local orientation is determined by comparing the filter responses of a given segment to a set of oriented filters. The orientation of the oriented filter with the greatest response in a given region is chosen as the local orientation. With this reasoning, the global orientation corresponds to the orientation of the filter with the greatest response over the whole image.
* The local orientations are stored in the variable 'theta' (consider that any region that is not part of an interstitial line has a default value of 0)

### Subplot 2,1
This plot shows the raw input image.
* This image is stored at the variable 'photo'

### Subplot 2,2
This subplot shows the estimated thicknesses of the layers contained between two interstitial lines.

This value is calculated with the help of the distance function (https://fr.mathworks.com/help/images/ref/bwdist.html) calculated over the morphological skeleton of the interstitial lines in 'grayImage'. Since the value calculated by the distance function is a radius, the value shown in the plot is twice the value in the distance function (i.e. diameter instead of radius).

For clarity purposes, the lines shown in the plot correspond to the skeleton of the distance function output (the skeleton must preserve the pixels with the highest values). 
Additionally, interstitial lines are shown with a value of 0 (for localization purposes).

* The morphological skeleton of the interstitial lines is stored in the variable 'skeleton'
* The output of the distance function is stored in the variable 'realDistances'
* The skeleton of the distance function output is stored in the variable 'filteredSkeleton'

### Subplot 2,3
This plot shows the approximate estimated local curvatures of the interstitial lines. The sign of the curvature determines of the line is concave (positive) or convex (negative) with respect to the horizontal axis.

Interstitial lines are first approximated as continuous (i.e. non-discrete) cubic splines and then local curvatures are calculated in terms of general parametrization (https://en.wikipedia.org/wiki/Curvature#In_terms_of_a_general_parametrization).
* The local curvatures are stored in the variable 'rMap' (consider that any region that is not part of an interstitial line has a default value of 0)