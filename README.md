# Fluorescence Microscopy - Filament Extraction
A python package to extract filament like structures from fluorescence microscopy images using a CNN approach.

_Author: David Ventzke_

### Introduction
When analysing fluorescence microscope images of cytoskeletons or other biological protein networks one may be confronted with the task to extract all the filaments from such a noisy image for further analysis of the network. This should include approximating curves for each of the detected filaments.

CyCNN solves this problem and idenfies all filament like structures in an image as well as close approximation of the filament shapes (as much as the image resolution permits). While this method was developed to analyze in-vivo networks of microtubules but generalizes well to other types of filament networks.

Our algorithm makes use of a Convolutional Neural Network trained on filament network structures to invert the artefacts introduced by the flourescence microscopy imaging process (i.e. different types of imaging noise, Point-Spread-Functions, etc.). The network reliably identifies filaments and also allows for differentiation of close by or intersecting filaments. In a second step identfied filaments are smoothly interpolated on the pixel grid, based on the intensity distribution of the image. For further details see [Methods](#methods)

### Example

Consider the following cell image of Microtubules in a MEF (image on the right is a subsection). We would like to detect all the Microtubules in the dense network and retrieve their coordinates from the image. 

![Microtubules in Fibroblast](./example_img/example.png)

Applying the method from this repository yields the following result on the given image: The filaments detected are highlighted in red.

![Demo](./example_img/processed.png)

The package also provides automatic local orientation and curvature estimates of with the coordinates of each extracted filament. 

TODO

### Methods

Suppose we have an approximately planar network of protein filaments. Then we can describe each filament by some curve $f_i: [0,1] \mapsto \mathbb{R^2}$