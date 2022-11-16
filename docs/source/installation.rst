Installation
============

We offer two possible installation approaches for our DL_Track software. The first option is to download the DL_Track executable file. The second option we describe is DL_Track package installation via Github and pythons package manager pip. We want to inform you that there are more ways to install the package. However, we do not aim to be complete and rather demonstrate an (in our opinion) user friendly way for the installation of DL_Track. Moreover, we advise users with less programming experience to make use of the first option and download the executable file.

Download the DL\_Track executable
---------------------------------

1. Got to the Zenodo webpage containing the DL_Track executable, the pre-trained models and the example file using this link (LINK).
2. Download the DL_Track.exe file. (Note that this is only an executable, not an installer.)
3. Download both pre-trained models (model-apo-VGG-BCE-512.h5 & model-VGG16-fasc-BCE-512.h5).
4. Download the DL_Track_example.zip file.
5. Create a specified DL_Track directory and put the DL_Track.exe, the model files and the example file in seperate subfolders (for example "Executable", "Models" and "Example"). Moreover, unpack the DL_Track_example.zip file.
6. Open the DL_Track GUI by double clicking the DL_Track.exe file and start with the testing procedure to check that everything works properly (see `Examples <https://dltrack.readthedocs.io/en/latest/usage.html>`_ and `Testing <https://dltrack.readthedocs.io/en/latest/tests.html>`_).

Install DL_Track via Github, pip and Pypi.org
---------------------------------------------

In case you want to use this way to install and run DL_Track, we advise you to setup conda (see step 1) and download the environment.yml file from the repo (see steps 5-8). If you want to actively contribute to the project or customize the code, it might be usefull to you to do all of the following steps (for more information see `Contributing Guidelines <https://dltrack.readthedocs.io/en/latest/contribute.html>`_).

1. Anaconda setup (only before first usage and if Anaconda/minicoda is not already installed).

Install `Anaconda <https://www.anaconda.com/distribution/>`_ (click ‘Download’ and be sure to choose ‘Python 3.X Version’ (where the X represents the latest version being offered. IMPORTANT: Make sure you tick the ‘Add Anaconda to my PATH environment variable’ box).

2. Git setup (only before first usage and if Git is not already installed). This is optional and only required when you want to clone the whole DL_Track Github repository.

In case you have never used Git before on you computer, please install it using the instructions provided `here <https://git-scm.com/download>`_.

3. Create a directory for DL_Track.

On your computer create a specific directory for DL_Track (for example "DL_Track") and navigate there. Once there open a git bash with right click and then "Git Bash Here"). In the bash terminal, type the following:

``git init``

This will initialize a git repository and allows you to continue. If run into problems, check this `website <https://git-scm.com/book/en/v2/Git-Basics-Getting-a-Git-Repository>`_.

4. Clone the DL_Track Github repository into a pre-specified folder (for example "DL_Track) by typing the following code in your bash window:

``git clone https://github.com/PaulRitsche/DL_Track.git``

This will clone the entire repository to your local computer. To make sure that everything worked, see if the files in your local directory match the ones you can find in the Github DL_Track repository. If you run into problem, check this `website <https://git-scm.com/book/en/v2/Git-Basics-Getting-a-Git-Repository>`_.

Alternatively, you can only download the environment.yml file from the `DL_Track repo <https://github.com/PaulRitsche/DLTrack/>`_ and continue to the next step.

5. Create the virtual environment required for DL_Track.

DL_Track is bound to several external depencies. To make your life easy, we provided a environment.yml file. Using the file, the entire environment containing all dependencies will be created for you. To do this, you need to navigate to the folder where the environment.yml file is placed. Then type the following command in your bash terminal:

``conda env create -f environment.yml``

The environment should be successfully created. We will see how to verify this in the next steps. If you run into problems with the .yml please file an issue in the issue section of the DL_Track repository (see docs directory on how to correctly file an issue for DL_Track).

**Attention MacOS users:** 
We advise you to use the provided requirements.txt file for environment creation. You need to slightly modify it. Change *tensorflow==2.10.0* to *tensorflow-macos=2.10.0*. You need to create and activate the environment first:

``conda create -n DL_Track python=3.10``

Activate the environment as described in the next section. Then you can install the requirements of DL_Track with: 

``pip install -r requirements.txt``

There are some more steps necessary for DL_Track usage, you'll finde the instructions in the `usage <https://dltrack.readthedocs.io/en/latest/usage.html>`_ section. 

6. Activate and verify the environment for usage of DL_Track.

You can now activate the virtual environment by typing:

``conda activate DL_Track``

An active conda environment is visible in () brackets befor your current path in the bash terminal. In this case, this should look something like (DL_Track) C:/user/.../DL_Track.

You can verify whether the environment was correctly created by typen the following command in your bash terminal:

``conda list``

Now, all packages included in the DL_Track environment will be listed and you can check if all packages listed in the environment.yml file under the section "- pip" are included in the DL_Track environment.

7. The First option of running DL_Track is installing the DL_Track package from Pypi.org. You do not need the whole cloned repository for this, only the active DL_Track environment. You do moreover not need be any specific directory.

You can install the DL_Track package by typing the following command in the bash terminal:

``pip install dl-track``

Once everythin is sucessfully installed, type in your bash terminal:

``python -m DL_Track``

The main GUI should now open. If you run into problems, open a discussion in the Q&A section of `DL_Track discussions <https://github.com/PaulRitsche/DLTrack_US/discussions/categories/q-a>`_ and assign the label "Problem".  For usage of DL_Track please take a look at the `docs <https://github.com/PaulRitsche/DLTrack/tree/main/docs/usage>`_ directory in the Github repository.

8. The second option of running DL_Track is using the DLTrack_GUI python script. This requires you to clone the whole directory and navigate to the directory where the DLTrack_GUI.py file is located. Moreover, you need the active DL_Track environment.

The DLTrack_GUI.py file is located at the `DL_Track <https://github.com/PaulRitsche/DLTrack/DL_Track>`_ folder. To execute the module type the following command in your bash terminal.

``python DLTrack_GUI.py``

The main GUI should now open. If you run into problems, open a discussion in the Q&A section of `DL_Track discussions <https://github.com/PaulRitsche/DLTrack_US/discussions/categories/q-a>`_ and assign the label "Problem". You can find an example discussion there. For usage of DL_Track please take a look at the `docs <https://github.com/PaulRitsche/DLTrack/tree/main/docs/usage>`_ directory in the Github repository.


GPU setup
---------

**Attention: The next section is only relevant for windows users!**

The processing speed of a single image or video frame analyzed with DL_Track is highly dependent on computing power. While possible, model inference and model training using a CPU only will decrese processing speed and prolong the model training process. Therefore, we advise to use a GPU whenever possible. Prior to using a GPU it needs to be set up. Firstly the GPU drivers must be locally installed on your computer. You can find out which drivers are right for your GPU `here <https://www.nvidia.com/Download/index.aspx?lang=en-us>`_. Subsequent to installing the drivers, you need to install the interdependant CUDA and cuDNN software packages. To use DL_Track with tensorflow version 2.10 you need to install CUDA version 11.2 from `here <https://developer.nvidia.com/cuda-11.2.0-download-archive>`_ and cuDNN version 8.5 for CUDA version 11.x from `here <https://developer.nvidia.com/rdp/cudnn-archive>`_ (you may need to create an nvidia account). As a next step, you need to be your own installation wizard. We refer to this `video <https://www.youtube.com/watch?v=OEFKlRSd8Ic>`_ (up to date, minute 9 to minute 13) or this `video <https://www.youtube.com/watch?v=IubEtS2JAiY&list=PLZbbT5o_s2xrwRnXk_yCPtnqqo4_u2YGL&index=2>`_ (older, entire video but replace CUDA and cuDNN versions). There are procedures at the end of each video testing whether a GPU is detected by tensorflow or not. If you run into problems with the GPU/CUDA setup, please open a discussion in the Q&A section of `DL_Track discussions <https://github.com/PaulRitsche/DLTrack_US/discussions/categories/q-a>`_ and assign the label "Problem".

**Attention : The next section is only relevant for MacOS users!**

In case you want to make use of you M1 / M2 chips for model training and / or inference, we refer you to this `tutorial <https://caffeinedev.medium.com/how-to-install-tensorflow-on-m1-mac-8e9b91d93706>`_. There you will find a detailed description of how to install tensorflow on you Mac.
