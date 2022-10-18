# Inline monitoring of 3D concrete printing
This repository provides an extension of the code presented at: https://github.com/Sutadasuto/3dcp_cv_monitoring

In this version, you can perform the analysis in a continuous loop. If you use this tool in your work, kindly cite the following paper:
```
@article{RILLGARCIA2022103175,
title = {Inline monitoring of 3D concrete printing using computer vision},
journal = {Additive Manufacturing},
volume = {60},
pages = {103175},
year = {2022},
issn = {2214-8604},
doi = {https://doi.org/10.1016/j.addma.2022.103175},
url = {https://www.sciencedirect.com/science/article/pii/S2214860422005644},
author = {Rodrigo Rill-García and Eva Dokladalova and Petr Dokládal and Jean-François Caron and Romain Mesnil and Pierre Margerit and Malo Charrier},
}
```

## Software requirements
The current repository uses Python and Matlab code.
Regarding the Python setup, an 'environment.yml' file is provided to replicate the conda environment used for testing.
The Python environment is used to build the interlayer segmentation and the texture classification networks on Tensorflow 2.
Regarding the Matlab code, it was tested on the 2020b version; it uses the image and the curve fitting toolboxes. The Matlab script performs the geometrical characterization.

## How to run

Matlab and Python communicate using TCP/IP sockets through local host. You can change the port at the "config_files/cam_and_com_config" file (the number after 'localhost, ').

Similarly, in the same file you can choose the video source. You can choose a video input device (an integer number; usually 0 if using an integrated webcam) or a folder containing only images (a string). By default, this argument is set to "I3DCP_demo_mini", a folder containing some sample images from the I3DCP dataset (https://github.com/Sutadasuto/I3DCP). Notice that the image names are numbered; you should follow this patter when using your own data.

Finally, you must specify the path to a file containing the pre-trained weights of the fully-convolutional neural network. By default, this path is set to "uvgg19_interlayer_segmentation.hdf5", meaning that the file is at the root of the project. We provide this file in zip parts.


To run the tool, you must first begin the python process by running:
```
python main.py
```

This script has an optional arguments: 'gpu' allows you to choose if you want to use a GPU (specially useful for machines without an integrated GPU). The default value is True (use False to disable GPU use).

Let the program run until the console prints "Waiting for a connection". This means Python is set up and ready to communicate with Matlab to begin the analysis.

In Matlab, you just need to run "main.m". Once Matlab has communicated with Python, the Matlab console will show a tcpclient variable and the Python console will print "Connection ready. Start!".

We include a tutorial video, which also shows the expected outputs ("quick_demo.mkv").

## User interface

Per analyzed frame/image, the present tool will produce local measurements about the geometry and texture of the observed piece. The geometry measures correspond to the orientation and curvature of the interlayer lines, the width of the printed layers, and the distance of the printing nozzle with respect to the last printed layer (for furhter information about these measurements and the properties of the expected input image, please refer to the article). The texture measurements are used to provide a region-wise classification of the observed material, either good or one of three anomalous classes: fluid, dry or tearing (for further information about these classes, please refer to the article). Once the measures are obtained, one histogram per each one of them is calculated.

Two windows are shown during execution: one for the geometrical measurement and texture classification, and one for their corresponding histograms:

![alt text](https://github.com/Sutadasuto/3dcp_cv_monitoring_inline/blob/main/results/plots.png?raw=true)
![alt text](https://github.com/Sutadasuto/3dcp_cv_monitoring_inline/blob/main/results/histograms.png?raw=true)