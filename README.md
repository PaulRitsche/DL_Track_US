# DL_Track

The DL_Track package provides an easy to use graphical user interface (GUI) for deep learning based analysing muscle architecture from longitudinal ultrasonography images of human lower limb muscles.

## Installation

We offer two possible installation approaches for our DL_Track software. The first option is to download the DL_Track executable file. The second option we describe is DL_Track package installation via Github and pythons package manager pip. We want to inform you that there are more ways to install the package. However, we do not aim to be complete and rather demonstrate an (in our opinion) user friendly way for the installation of DL_Track. Moreover, we advise users with less programming experience to make use of the first option and download the executable file.

### Download the DL_Track executable

1. Got to the Zenodo webpage containing the DL_Track executable, the pre-trained models and the example file using this link (LINK).
2. Download the DL_Track.exe file. (Note that this is only an executable, not an installer.)
3. Download both pre-trained models (model-apo-VGG-BCE-512.h5 & model-VGG16-fasc-BCE-512.h5).
4. Download the DL_Track_example.zip file.
5. Create a specified DL_Track directory and put the DL_Track.exe, the model files and the example file in seperate subfolders (for example "Executable", "Models" and "Example"). Moreover, unpack the DL_Track_example.zip file.
6. Open the DL_Track GUI by double clicking the DL_Track.exe file and start with the testing procedure to check that everything works properly (we will get to that down below, see section Examples and Testing).

### Install DL_Track via Github, pip and Pypi.org

In case you want to use this way to install and run DL_Track, we advise you to setup conda (see step 1) and download the environment.yml file from the repo (see steps 5-8). If you want to actively contribute to the project or customize the code, it might be usefull to you to do all of the following steps (for more information on contributing see Usage directory DL_Track_contribute.md).

1. Anaconda setup (only before first usage and if Anaconda/minicoda is not already installed).

Install [Anaconda](https://www.anaconda.com/distribution/) (click ‘Download’ and be sure to choose ‘Python 3.X Version’ (where the X represents the latest version being offered. IMPORTANT: Make sure you tick the ‘Add Anaconda to my PATH environment variable’ box).

2. Git setup (only before first usage and if Git is not already installed). This is optional and only required when you want to clone the whole DL_Track Github repository.

In case you have never used Git before on you computer, please install it using the instructions provided [here](https://git-scm.com/download).

3. Create a directory for DL_Track.

On your computer create a specific directory for DL_Track (for example "DL_Track") and navigate there. Once there open a git bash with right click and then "Git Bash Here"). In the bash terminal, type the following:
```sh
git init
```
This will initialize a git repository and allows you to continue. If run into problems, check this [website](https://git-scm.com/book/en/v2/Git-Basics-Getting-a-Git-Repository).

4. Clone the DL_Track Github repository into a pre-specified folder (for example "DL_Track) by typing the following code in your bash window:

```sh
git clone https://github.com/PaulRitsche/DL_Track.git
```
This will clone the entire repository to your local computer. To make sure that everything worked, see if the files in your local directory match the ones you can find in the Github DL_Track repository. If you run into problem, check this [website](https://git-scm.com/book/en/v2/Git-Basics-Getting-a-Git-Repository).

Alternatively, you can only download the environment.yml file and continue to the next step.

5. Create the virtual environment required for DL_Track.

DL_Track is bound to several external depencies. To make your life easy, we provided a environment.yml file. Using the file, the entire environment containing all dependencies will be created for you. To do this, you need to navigate to the folder where the environment.yml file is placed. Then type the following command in your bash terminal:

```sh
conda env create -f environment.yml
```
The environment should be successfully created. We will see how to verify this in the next steps. If you run into problems with the .yml please file an issue in the issue section of the DL_Track repository (see docs directory on how to correctly file an issue for DL_Track).

6. Activate and verifyinf the environment for usage of DL_Track.

You can now activate the virtual environment by typing:

```sh
conda activate DL_Track
```
An active conda environment is visible in () brackets befor your current path in the bash terminal. In this case, this should look something like (DL_Track) C:/user/.../DL_Track.

You can verify whether the environment was correctly created by typen the following command in your bash terminal:
```sh
conda list
```
Now, all packages included in the DL_Track environment will be listed and you can check if all packages listed in the environment.yml file under the section "- pip" are included in the DL_Track environmen.

7. The First option of running DL_Track is installing the DL_Track package from Pypi.org. You do not need the whole cloned repository for this, only the active DL_Track environment. You do moreover not need be any specific directory.

You can install the DL_Track package by typing the following command in the bash terminal:
```sh
pip install dl-track
```
Once everythin is sucessfully installed, type in your bash terminal:
```sh
python -m DL_Track
```
The main GUI should now open. If you run into problems, open a discussion in the Q&A section of [DL_Track discussions](https://github.com/PaulRitsche/DLTrack_US/discussions/categories/q-a) and assign the label "Problem". You can find an example discussion there. For usage of DL_Track please take a look at the [inference]() folder in the [docs]() directory and the respective [DL_Track_usage_inference.pdf]() file.

8. The second option of running DL_Track is using the DLTrack_GUI python script. This requires you to clone the whole directory and navigate to the directory where the DLTrack_GUI.py file is located. Moreover, you need the active DL_Track environment.

The DLTrack_GUI.py file is located at DL_Track/src/DL_Track. To execute the module type the following command in your bash terminal.
```sh
python DLTrack_GUI.py
```
The main GUI should now open. If you run into problems, open a discussion in the Q&A section of [DL_Track discussions](https://github.com/PaulRitsche/DLTrack_US/discussions/categories/q-a) and assign the label "Problem". You can find an example discussion there. For usage of DL_Track please take a look at the [inference]() folder in the [docs]() directory and the respective [DL_Track_usage_inference.pdf]() file.


### GPU/CUDA setup

The processing speed of a single or video frame analyzed with DL_Track is highly dependent on computing power. While possible, model inference and model training using a CPU only will decrese processing speed and prolong the model training process. Therefore, we advise to use a GPU whenever possible. Prior to using a GPU it needs to be set up. Firstly the GPU drivers must be locally installed on your computer. You can find out which drivers are right for your GPU here: [Drivers](https://www.nvidia.com/Download/index.aspx?lang=en-us). Subsequent to installing the drivers, you need to install the interdependant CUDA and cuDNN software packages. To use DL_Track with tensorflow version 2.10 you need to install CUDA version 11.2 from here: [CUDA](https://developer.nvidia.com/cuda-11.2.0-download-archive) and cuDNN version 8.5 for CUDA version 11.x from here: [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive) (you may need to create an nvidia account). As a next step, you need to be your own installation wizard. We refer to this [video](https://www.youtube.com/watch?v=OEFKlRSd8Ic) (up to date, minute 9 to minute 13) or this [video](https://www.youtube.com/watch?v=IubEtS2JAiY&list=PLZbbT5o_s2xrwRnXk_yCPtnqqo4_u2YGL&index=2) (older, entire video but replace CUDA and cuDNN versions). There are procedures at the end of each video testing whether a GPU is detected by tensorflow or not. If you run into problems with the GPU/CUDA setup, please open a discussion in the Q&A section of [DL_Track discussions](https://github.com/PaulRitsche/DLTrack_US/discussions/categories/q-a) and assign the label "Problem". You can find an example discussion there.

## Examples

The use of DL_Track is GUI based. Therefore, all functionalities are selected and executed from the GUI. The GUI can be started as described in the Installation section. Because DL_Track does not require you to interact with any python code, we refer you to the [DL_Track_usage_inference.pdf]() file in the [inference]() folder in the [docs]() directoy where usage examples are illustrated. For an example on model training we refer you to the [DL_Track_usage_training.pdf"]() file in the [training]() folder in the [docs]() directory. There are moreover specific instructions on data preparation prior to model training.


## Testing

Due to large file size of the example videos, all example videos and images can be found in the [DL_Track_examples.zip]() file on Zenodo. We do not provide unit tests for DL_Track yet, which is why the functionality of the package must be tested manually. For this, we refer you to the [DL_Track_test_image.pdf]() and [DL_Track_test_video.pdf]() file in the [tests]() directory. Please make sure to exactly follow all described steps and keep all parameter setting similiar to the ones described. The analysis results provided at the end of the pdf files will match the results of your analysis should the DL_Track package function properly. If you run into problems during testing, please file an issue in the issue section of the DL_Track repository (see docs directory on how to correctly file an issue for DL_Track).

## Community guidelines

We highly encourage you to contribute to the DL_Track package by training and providing you own models or improving the code based on your own needs. Please take a look at the [DL_Track_contribute.md]() file in the [community]() folder in the [docs]() directory on how to properly contribute to the DL_Track package. If you notice anything wrong with the package, don't hesitate to file a bug report in the [issue]() section. See how this is properly done in the [DL_Track_bugreport.md] file in the [community]() folder in the [docs]() directory. In case there is anything you would like to add to the package but do not know how to implement it yourself, feel free to open an issue in the [issue]() section. A guide on how to so is provided in the [DL_Track_issue.md]() file in the [community]() folder in the [docs]() directory.

## Support
In cases where installation and usage don't go smoothly and you run into problems there are two possible ways to seek support. We advise you to open a discussion in the Q&A section of [DL_Track discussions](https://github.com/PaulRitsche/DLTrack_US/discussions/categories/q-a) and assign the label "Problem". You can find an example discussion there. You can also contact us via email at paul.ritsche@unibas.ch, but we would prefer you to open a discussion.
