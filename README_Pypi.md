# DL_Track_US

[![Documentation Status](https://readthedocs.org/projects/dltrack/badge/?version=latest)](https://dltrack.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7866598.svg)](https://doi.org/10.5281/zenodo.7866598)

The DL_Track_US package provides an easy to use graphical user interface (GUI) for deep learning based analysis of muscle architectural parameters from longitudinal ultrasonography images of human lower limb muscles. Please take a look at our [documentation](https://dltrack.readthedocs.io/en/latest/index.html) for more information.
This code is based on a previously published [algorithm](https://github.com/njcronin/DL_Track) and replaces it. We have extended the functionalities of the previously proposed code. The previous code will not be updated and future updates will be included in this repository.

## Getting started

For detailled information about installaion of the DL_Track_US python package we refer you to our [documentation](https://dltrack.readthedocs.io/en/latest/installation.html). There you will finde guidelines not only for the installation procedure of DL_Track_US, but also concerning conda and GPU setup.

## Quickstart

Once installed, DL_Track_US can be started from the command prompt with the respective environment activated:

``(DL_Track_US) C:/User/Desktop/ python -m DL_Track_US`` 

In case you have downloaded the executable, simply double-click the DL_Track_US icon.

Regardless of the used method, the GUI should open. For detailed the desciption of our GUI as well as usage examples, please take a look at the [user instruction](https://github.com/PaulRitsche/DL_Track_US/tree/main/docs/usage).

## Testing

We have not yet integrated unit testing for DL_Track_US. Nonetheless, we have provided instructions to objectively test whether DL_Track_US, once installed, is functionable. To perform the testing procedures yourself, check out the [test instructions](https://github.com/PaulRitsche/DL_Track_US/tree/main/tests).

## Code documentation 

In order to see the detailled scope and description of the modules and functions included in the DL_Track_US package, you can do so either directly in the code, or in the [Documentation](https://dltrack.readthedocs.io/en/latest/modules.html#documentation) section of our online documentation.

## Previous research

The previously published [algorithm](https://github.com/njcronin/DL_Track) was developed with the aim to compare the performance of the trained deep learning models with manual analysis of muscle fascicle length, muscle fascicle pennation angle and muscle thickness. The results were presented in a published [preprint](https://arxiv.org/pdf/2009.04790.pdf). The results demonstrated in the article described the DL_Track_US algorithm to be comparable with manual analysis of muscle fascicle length, muscle fascicle pennation angle and muscle thickness in ultrasonography images as well as videos.

## Community guidelines

Wheter you want to contribute, report a bug or have troubles with the DL_Track_US package, take a look at the provided [instructions](https://dltrack.readthedocs.io/en/latest/contribute.html) how to best do so. You can also contact us via email at paul.ritsche@unibas.ch, but we would prefer you to open a discussion as described in the instructions.
