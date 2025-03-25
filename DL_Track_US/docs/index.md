# Welcome to DL_Track_US

## Overview

***Automated analysis of human lower limb ultrasonography images***

So, what is DL_Track_US all about? The DL_Track_US algorithm was first presented by Neil Cronin,
Olivier Seynnes and Taija Finni in [2020](https://arxiv.org/pdf/2009.04790.pdf). The algorithm makes extensive use of fully convolutional neural networks trained on a fair amount of ultrasonography images of the human lower limb. Specifically, the dataset included longitudinal ultrasonography images from the human gastrocnemius medialis, tibialis anterior, soleus and vastus lateralis. The algorithm is able to analyse muscle architectural parameters (muscle thickness, fasciclelength and pennation angle) in both, single image files as well as videos. By employing deep learning models, the DL_Track_US algorithm is one of the first fully automated algorithms, requiring no user input during the analysis. Then in 2022, we (Paul Ritsche, Olivier Synnes, Neil Cronin) have updated the code and deep learning models substantially, added a graphical user interface, manual analysis and an extensive documentation. Moreover we turned everything into an openly available Pypi package.

## Why use DL_Track_US?

Using the DL_Track_US python package to analyze muscle architectural parameters in human lower limb muscle ultrasonography images hase two main advantages. The analysis is objectified when using the automated analysis types for images and videos because no user input is required during the analysis process. Secondly, the required analysis time for image or video analysis is drastically reduced compared to manual analysis. Whereas an image or video frame manual analysis takes about one minute, DL_Track_US analyzes images and video frames in less than one second. This allows users to analyze large amounts of images without supervision during the analysis process in relatively short amounts of time.

## Good to know

Before you get started, here are some important tips:

- Test the algorithm first and train your own models if necessary, especially if 
you plan to analyze images taken from different muscles.
- Be cautious about the generalizability of the models, even though 
extensive data augmentation was used during the model training process. 
Different device types, muscle regions, and settings during image 
acquisition may impact model performance.
- Image quality is crucial. The images should have good contrast, appropriate 
brightness, clearly visible fascicles and aponeuroses, and clear alignment of 
the probe with the fascicle plane.
- If model performance is poor, visually inspect the output of the models and 
compare them to manual analysis results. Adjust analysis parameters or 
train a separate model if necessary.
- Follow the provided testing procedures in the DL_Track_US/tests folder to 
ensure proper functionality on your computer.

## Limitations

Currently, we have not provided unit testing for the functions and modules included in the DL_Track_US package. Moreover, the muscles included in the training data set are limited to the lower extremities. Although we included images from as many ultrasonography devices as possible, we were only able to include images from four different devices. Therefore, users aiming to analyze images from different muscles or different ultrasonography devices might be required to train their own models because the provided pre-trained models result in bad segmentations. The time required for image analysis compared to manual analysis is tremendously reduced. However, employing the networks for analysis of long videos containing many frames (>2000) may still require a few hours. Lastly, even though  DL_Track_US objectifies the analysis of ultrasonography images when using the automated analysis types, we labeled the images manually. Therefore, we introduced some subjectivity into the datasets.