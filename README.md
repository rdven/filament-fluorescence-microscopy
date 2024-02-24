# Fluorescence Microscopy - Filament Extraction
A python package for image processing to identify and extract filament like structures from fluorescence microscopy images using a CNN approach. This can be useful for analysing cytoskeletal protein networks in cells like microtubuli networks.

_Author: David Ventzke_

### Introduction
When analysing confocal fluorescence microscope images of cytoskeletons or other biological protein networks one may be confronted with the task to extract all the filaments from such a noisy image for further analysis of the network. This should include approximating curves for each of the detected filaments.

CyCNN solves this problem and idenfies all filament like structures in an image as well as close approximation of the filament shapes (as much as the image resolution permits). While this method was developed to analyze in-vivo networks of microtubules but generalizes well to other types of filament networks. A demo can be found in the following [Jupyter-Notebook](CNN%20Demo.ipynb)

Our algorithm makes use of a Convolutional Neural Network trained on filament network structures to invert the artefacts introduced by the flourescence microscopy imaging process (i.e. different types of imaging noise, Point-Spread-Functions, etc.). The network reliably identifies filaments and also allows for differentiation of close by or intersecting filaments. In a second step identfied filaments are smoothly interpolated on the pixel grid, based on the intensity distribution of the image. For further details see [Methods](#methods)

### Example

Consider the following cell image of Microtubules in a MEF (image on the right is a subsection). We would like to detect all the Microtubules in the dense network and retrieve their coordinates from the image. 

![Microtubules in Fibroblast](./example_img/example.png)

Applying the method from this repository yields the following result on the given image: The filaments detected are highlighted in red.

![Demo](./example_img/processed.png)

The package also provides automatic local orientation and curvature estimates of with the coordinates of each extracted filament. 

![Single Microtubule](./example_img/single.png)

### Methods

Suppose we have an approximately planar network of protein filaments. Then we can describe each filament by some curve in the plane $f_i: [0,1] \mapsto \mathbb{R^2}$. These filaments $f_1,..,f_n$ are then imaged and we obtain an image like in the example above. During the imaging process the filament intensities are convolved with a point-spread function and the final intensity at a pixel is furthermore subject to different noise effects (possion noise, detector noise,..). We can thus write the imaging process as some operator $\mathcal F$ where $$\mathcal{F}:(f_i)_{i=1,..,n} \mapsto I$$ where $I$ is the resulting image of pixel-wise intensities. We would now like to learn an inverse to this map $\mathcal F$ given the knowledge that the initial configuration that was images was some element in a manifold of possible filament configurations.

The image processing method for filament extraction works as follows:
- 1) Simulate random filament network structures
- 2) Apply the known imaging model to these images to simulate fluorescence microscopy images of the filament networks. This requires some parameter tuning (optical resolution, SNR, background) to mimic the real world imaging process. The parameters can be identified by comparing the simulated images to the real images of interest. See [Forward-Operator-Demo.ipynb](./Forward-Operator-Demo.ipynb) for an example.
- 3) Train a convolutional neural network (CNN) to reconstruct the original network structure given the fluorescence microscopy image on the simulate. The network structure is encoded in a sparse 3D matrix, where 2D represent the original image and a 3rd (periodic) dimension represents local filament orientation, thus representing filaments as curves on $\mathbb R^2 \times S^1$ for better filament separability even in dense network regions. 
- 4) To identify and extract filaments from a real images, apply the trained CNN to the image, then use a skeletonization based algorithm to detect all filaments in the output matrix and extract their approximate pixel coordinates
- 5) For each filament, create a refinement of the pixelated coordinates by fitting a smoothe curve onto the previous pixel coordinates such that the resulting curve optimally matches the intensity pattern in the image, while preserving smootheness.
- 6) A list of all extracted filaments, their coordinates, orientation and local curvatures is provided as output.
