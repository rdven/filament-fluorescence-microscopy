# Fluorescence Microscopy - Filament Extraction
A python package to extract filament like structures from fluorescence microscopy images using a CNN approach.

### Setting
Given a noisy fluorescence microscopy image of some filament like structures one would like to retrieve these structures from the image, including a coordinat based parameterisation of the fiaments.
This can be formulated as an inverse problem, as follows:

### Example

Cosinder the following cell image
![Microtubules in Fibroblast](./example_img/example.png)

We now want to extract all the microtubules we can detect, (displayed in red) and also get local curvature estimates es seen in the rightmost image:

![Demo](./example_img/processed.png)
