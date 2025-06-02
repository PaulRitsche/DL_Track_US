# DL_Track_US

[![DOI](https://joss.theoj.org/papers/10.21105/joss.05206/status.svg)](https://doi.org/10.21105/joss.05206)

The DL_Track_US package provides an easy to use graphical user interface (GUI) for deep learning based analysis of muscle architectural parameters from longitudinal ultrasonography images of human lower limb muscles. Please take a look at our [documentation](https://paulritsche.github.io/DL_Track_US/) for more information (note that aggressive ad-blockers might break the visualization of the repository description as well as the online documentation).
This code is based on a previously published [algorithm](https://github.com/njcronin/DL_Track) and replaces it. We have extended the functionalities of the previously proposed code. The previous code will not be updated and future updates will be included in this repository. 

## Getting started

For detailled information about installaion of the DL_Track_US python package we refer you to our [documentation](https://paulritsche.github.io/DL_Track_US/). There you will finde guidelines not only for the installation procedure of DL_Track_US, but also concerding conda and GPU setup.

## Quickstart

Once installed, DL_Track_US can be started from the command prompt with the respective environment activated:

``(DL_Track_US0.3.0) C:/User/Desktop/ python -m DL_Track_US`` 

In case you have downloaded the executable, simply double-click the DL_Track_US icon.

Regardless of the used method, the GUI should open. For detailed the desciption of our GUI as well as usage examples, please take a look at the [user instruction](https://paulritsche.github.io/DL_Track_US/). An illustration of out GUI start window is presented below. It is here where users must specify input directories, choose the preferred analysis type, specify the analysis parameters or train thrain their own neural networks based on their own training data. 

![GUI](./DL_Track_US/docs/md_graphics/DLTrack_mainUI.png)

## Testing

We have not yet integrated unit testing for DL_Track_US. Nonetheless, we have provided instructions to objectively test whether DL_Track_US, once installed, is functionable. To perform the testing procedures yourself, check out the [test instructions](https://paulritsche.github.io/DL_Track_US/).

## Code documentation 

In order to see the detailled scope and description of the modules and functions included in the DL_Track_US package, you can do so either directly in the code, or in the [Documentation](https://paulritsche.github.io/DL_Track_US/) section of our online documentation.

## Community guidelines

Wheter you want to contribute, report a bug or have troubles with the DL_Track_US package, take a look at the provided [instructions](https://paulritsche.github.io/DL_Track_US/) how to best do so.
