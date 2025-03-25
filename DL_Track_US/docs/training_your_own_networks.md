The DL_Track_US package GUI includes the possibility to train your own
neural networks. We will demonstrate how to do this, with a few notes at
the beginning:

- It is advantageous to have a working GPU setup, otherwise model training
will take much longer. Take a look at our [Github repository](https://github.com/PaulRitsche/DL_Track_US) for further
instructions.
- If you don’t have any experience with training deep neural networks,
please refer to this [course](https://deeplizard.com/learn/video/gZmobeGL0Yg). We advise you to start with the pre-defined
settings. However, DL_Track_US does not allow to change the
architecture of the trained neural networks.

The paired original images and labeled masks required for the network
training are located in the “DL_Track_US_example/model_training” folder. If
you haven’t downloaded this folder, please do so now (link: [DL_Track_US - Examples & Models](https://osf.io/7mjsc/?view_only=)). 
Unzip the folder and put it somewhere accessible. We
will demonstrate how to train a model that segments the muscle aponeuroses.

Please keep in mind that the model training process will be illustrated by
training a model for aponeurosis segmentation. The process is exactly the
same for training a fascicle segmentation model. Solely the images and masks
should then contain fascicles and fascicle labels.

## 1. Data Preparation and Image Labeling

- The «DL_Track_US_example/model_training” folder contains to subfolders,
<span style="color: #2eaf66;">**apo_img_example**</span> and 
<span style="color: #a34ba1;">**apo_mask_example**</span>.
- The original images are located in the “apo_img_example” folder.
- The corresponding masks are located in the “apo_maks_example”
folder.
- We advise you to keep a similar folder structure when you train
your own models outside of this tutorial.

![model training folder](md_graphics\training_your_own_networks\model_training_folder.png)

- Below you can see that the <span style="color: #2eaf66;">**original image**</span> and the 
<span style="color: #a34ba1;">**corresponding masks**</span>
have exactly the same name. This is **SUPER MEGA** important. Otherwise,
the model is trained using the wrong masks for the images.

![image mask](md_graphics\training_your_own_networks\image_mask.png)

## 2. Specifying Relevant Directories

- As a next step, you can start the GUI.
- Once you started the GUI and the main GUI window opened, click on the
<span style="color: #a34ba1;">**Advanced Methods**</span> Button.
- A small window will pop up, where you can select the method.
- Click Train Model from the <span style="color: #2eaf66;">**dropdown-menu**</span>.
- The window then folds out as shown in the picture below.

![advanced methods](md_graphics\training_your_own_networks\advanced_methods.png)
![select method](md_graphics\training_your_own_networks\select_method.png)
<img src="\md_graphics\training_your_own_networks\train_model.png" width="400">

Firstly, select the “Image Directory”.

- Click the button <span style="color: #a34ba1;">**Images**</span>.
- A selection window will appear and you can select the folder
containing the original images.
- Select the “DL_Track_US_example/model_training/apo_img_example”
folder.

<img src="\md_graphics\training_your_own_networks\images_button.png" width="400">

Your next step is to select the “Mask Directory”.

- Click on the button <span style="color: #a34ba1;">**Masks**</span>.
- A selection window will appear to select the folder containing the
mask images.
- Select the “DL_Track_US_example/model_training/apo_mask_example”
folder.

<img src="\md_graphics\training_your_own_networks\masks_button.png" width="400">

The last directory you need to select for training your own network
is the “Output Directory”.

- Click the button <span style="color: #a34ba1;">**Output**</span>.
- In the Output directory, the trained model, the corresponding loss
calculation results and a graphic displaying plotting the training
epochs against the loss values will be saved.
- A selection window will appear and you can select any folder you
like.

<img src="\md_graphics\training_your_own_networks\output_button.png" width="400">

## 3. Image Augmentation

Image augmentation is a method to artifically increase the size of your training
data. In this case, this means multiplying your images and masks based on a
generator that changes certain properties of the images. You can find the
details of this generator in the code documentation.

**Image augmentation optional but advisable if image number is low, i.e. <1500.**

Given you have specified the relevant directories priorly, simply click the
<span style="color: #a34ba1;">**Augment Images**</span> button and see your images being multiplied. A
Messagebox will indicate when the augmentation process is finished.

<img src="\md_graphics\training_your_own_networks\augment_images_button.png" width="400">

## 4. Specifying Training Parameters

Now to specifying the <span style="color: #a34ba1;">**training parameters**</span>.

- For the tutorial leave the pre-specified selections as they are.
- If you do not know what these training parameters mean, take a look at
this [course](https://deeplizard.com/learn/video/gZmobeGL0Yg).
- The only thing we have to say is that you must **NEVER** use only three
<span style="color: #299ed9;">**Epochs**</span> for actual model training.
- Such a small number of training Epochs is only acceptable for
demonstration and testing purposes.
- For actual training of your own neural networks, go with at least 60
Epochs.

<img src="\md_graphics\training_your_own_networks\hyperparameters.png" width="400">

The only thing you have left to do for the training process to start is to click the
<span style="color: #a34ba1;">**Start Training**</span> button.

<img src="\md_graphics\training_your_own_networks\start_training_button.png" width="400">

- During the training process, three messageboxes will pop up.
- The first one will tell you that the images and masks were
successfully loaded for further processing.
- The second one will tell you that the model was successfully
compiled and can now be trained.
- The last one will tell you that the training process was completed.
- You do have a choice in each messagebox of clicking “OK” or “Cancel”.
- Clicking “OK” will continue the training process, whereas clicking
“Cancel” will be cancelling the ongoing training process.

Once the training process in finished, three new files will be placed in your 
output directory.

- The trained model as Test_Apo.h5 file.
- The corresponding loss values for each epoch as Test_apo.csv file
- The graphical representation of the training process as Training_Results.tif
file.

## 5. Using Your Own Networks

How do you use you previously trained neural network?

- Simply select the path to your model by clicking the <span style="color: #a34ba1;">**Apo Model**</span> or 
<span style="color: #a34ba1;">**Fasc Model**</span> buttons in the GUI, depending on which model you want to
import.
- Subsequently to specifying all other relevant parameters for your
analysis in the GUI (as you have learned a couple pages ago).
- DL_Track_US will now analyse your data using your own model.

![apo model fasc model](md_graphics\training_your_own_networks\apo_model_fasc_model.png)

Lastly, a short disclaimer when training your own model.

- It is bad practice using the same images for model training and inference.
- The model should not be used for analysing images it was trained on
because it already knows the characteristics of these images.
- **ALWAYS** compare the results of your model to a manual evaluation on a
few of your own images. Use different images (best from different
individuals) for model training and comparison to manual analysis.
- If this seems strange to you, don’t hesitate to ask for further clarification in
the [DL_Track_US discussion forum](https://github.com/PaulRitsche/DL_Track_US/discussions/categories/q-a).

## 6. Error Handling

Whenever an error occurs during the analysis process, the DL_Track_US GUI
will open a <span style="color: #299ed9;">**messagebox**</span>. This looks always similar to this:

![error handling](md_graphics\training_your_own_networks\error_handling.PNG)

We tried to formulate these messageboxes as concise as possible. Just follow
their instructions to fix the error and run the analysis anew. In case an error
occurs that is not caught by an error messagebox, don’t hesitate to report this
in the Q&A section in the [DL_Track_US discussion forum](https://github.com/PaulRitsche/DL_Track_US/discussions/categories/q-a). 
lease take a look [here](https://dltrack.readthedocs.io/en/latest/contribute.html) how do best do this.

## 7. Image Labels

When you train your own networks, you need to label your original
ultrasonography images.

- We provide an <span style="color: #a34ba1;">**automated script**</span> for image labellig.
- This script does not automatically label the images, but automates the
selection processes and image / mask saving.
- The software you will perform the labelling in is called <span style="color: #299ed9;">**ImageJ / Fiji**</span>. You
can download it [here](https://imagej.net/software/fiji/downloads).
- The automated script “Image_Labeling_DL_Track_US.ijm” is located in
the folder “DL_Track_US/docs/labeling/” in our [Github repository](https://github.com/PaulRitsche/DL_Track_US).
- The easiest way to run the “Image_Labeling_DL_Track_US.ijm” script
is by simply drag and drop it in the running Fiji / ImageJ window.

<img src="\md_graphics\training_your_own_networks\fiji.png">
<img src="\md_graphics\training_your_own_networks\labelling_file.png">

Before you can start the labelling process:

- Create four folders in an easily accessible place.
- One folder containing the <span style="color: #a34ba1;">**original images**</span> you want to label.
- Then create three more folders, one named <span style="color: #299ed9;">**“output_images”**</span>, the
second called <span style="color: #f97e25;">**“fascicle_masks”**</span> and the third called
<span style="color: #2eaf66;">**“aponeurosis_masks”**</span>.
- In the “output_images” the original images are saved with an
adapted name.
- In the “fascicle_masks” and “aponeurosis_masks” folder the
respective masks are saved with the same name as the corresponding
image in “output_images”

<img src="\md_graphics\training_your_own_networks\labelling_folder.png">

When you have created all folders, press the <span style="color: #a34ba1;">**Run button**</span> in the Fiji /
ImageJ API to start the “Image_labelling_DL_Track_US.ijm” script.

<img src="\md_graphics\training_your_own_networks\run_button.png">

Follow the instructions appearing in the messageboxes.

- To begin with, you need to specify the four directories.
- The first directory you need to select is the original image folder (called
input dir).
- The second folder is the “aponeurosis_masks” folder (called apo
mask dir).
- The third is the “fascicle_masks” folder (called fasc mask dir).
- The last folder you need to specify is the “output_images” folder
(called image dir).

Subsequent to specifying the directories, you are required to create
the masks.

- First the aponeurosis mask, then the fascicle mask.
- Firstly, draw the superficial aponeurosis using the selected polygon tool
by following the instructions in the messagebox.
- Draw around the superficial aponeurosis (double click to start drawing,
click to add a segment, double click do stop drawing).
- Once you are finished, click the OK button in the messagebox to
proceed to the selection of the lower aponeurosis.
- Please be careful to only include aponeurosis tissue in your selection
and no surrounding tissue.
- The result should look like this for the upper and lower aponeurosis:

<img src="\md_graphics\training_your_own_networks\upper_aponeurosis.png">
<img src="\md_graphics\training_your_own_networks\lower_aponeurosis.png">

Once you have selected the lower aponeurosis, click the OK button in the
messagebox to proceed to the fascicle labelling. Follow the instructions in the messagebox.

- It is of utmost importance that you draw only over the actually visible
parts of the fascicle segment.
- Make sure that you only label bright fascicle tissue that is clearly visible.
- Once you drew one fascicle with segmented line tool (double click to
start drawing, click to add a segment, double click do stop drawing)
click the OK button in the messagebox to proceed to the next fascicle
segment.
- Draw as many segments as are clearly visible on the image.
- When you press the OK button in the messagebox without making a
further selection, you will proceed to the next image in the original image
folder and start again with the aponeurosis labelling.
- The result of you labelling should look something like this:

<img src="\md_graphics\training_your_own_networks\fascicles.png">