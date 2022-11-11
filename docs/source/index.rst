.. DL_Track documentation master file, created by
   sphinx-quickstart on Mon Nov  7 13:31:24 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DL_Track
========
*Automated analysis of human lower limb ultrasonography images*

So, what is DL_Track all about? The DL_Track algorithm was first presented by Neil Cronin,
Olivier Seynnes and Taija Finni in 2020. The algorithm makes extensive use of fully
convolutional neural networks trained on a fair amount of ultrasonography images of the
human lower limb. Specifically, the dataset included longitudinal ultrasonography images
from the human gastrocnemius medialis, tibialis anterior, soleus and vastus lateralis. The
algorithm is able to analyse muscle architectural parameters (muscle thickness, fascicle
length and pennation angle) in both, single image files as well as videos. By employing the
deep learning models, the DL_Track algorithm is one of the first fully automated algorithms,
requiring no user input during the analysis. Then in 2022, we (Paul Ritsche, Olivier Synnes,
Neil Cronin) have updated the code and deep learning models substantially, added a
graphical user interface, manual analysis and an extensive documentation. Moreover we
turned everything into an openly available Pypi package.


Why use DL_Track?
=================

Using the DL_Track python package to analyze muscle architectural parameters in human lower limb muscle ultrasonography images hase two main advantages. The analysis is objectified when using the automated analysis types for images and videos because no user input is required during the analysis process. Secondly, the required analysis time for image or video analysis is drastically reduced compared to manual analysis. Whereas an image or video frame manual analysis takes about one minute, DL_Track analyzes images and video frames in less than one second.


Limitations
===========

Currently, we have not provided unit testing for the functions and modules included in the DL_Track package. Moreover, the muscles included in the training data set are limited to the lower extremities. Although we included images from as many ultrasonography images as possible, we were only able to include images from four different devices.Therefore, users aiming to analyze images from different muscles or different ultrasonography devices might be required to train their own models because the provided pre-trained model result in bad segmentations. The time required for image analysis compared to manual analysis is tremendously reduced. However, employing the networks on many images requires time and therefore long videos containing many frames may still require a few hours to be analyzed. Lastly, even though  DL_Track objectifies the analysis of ultrasonography image analysis when using the automated analysis types, we labeled the images manually. Therefore, we introduced some subjectivity into the datasets.


.. toctree::
   :caption: Contents
   :hidden:

   modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
