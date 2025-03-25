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

<img src="\md_graphics\image_labels\fiji.png">
<img src="\md_graphics\image_labels\labelling_file.png">

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

<img src="\md_graphics\image_labels\labelling_folder.png">

When you have created all folders, press the <span style="color: #a34ba1;">**Run button**</span> in the Fiji /
ImageJ API to start the “Image_labelling_DL_Track_US.ijm” script.

<img src="\md_graphics\image_labels\run_button.png">

Follow the instructions appearing in the messageboxes.

- To begin with, you need to specify the four directories.
- The first directory you need to select is the original image folder (called
input dir).
- The second folder is the “aponeurosis_masks” folder (called apo
mask dir).
- The third is the “fascicle_masks” folder (called fasc mask dir).
- The last folder you need to specify is the “output_images” folder
(called image dir).
- Subsequent to specifying the directories, you are required to create
the masks.
- First the aponeurosis mask, then the fascicle mask.
- How to do this is demonstrated on the next page.
- Firstly, draw the superficial aponeurosis using the selected polygon tool
by following the instructions in the messagebox.
- Draw around the superficial aponeurosis (double click to start drawing,
click to add a segment, double click do stop drawing).
- Once you are finished, click the OK button in the messagebox to
proceed to the selection of the lower aponeurosis.
- Please be careful to only include aponeurosis tissue in your selection
and no surrounding tissue.
- The result should look like this for the upper and lower aponeurosis: