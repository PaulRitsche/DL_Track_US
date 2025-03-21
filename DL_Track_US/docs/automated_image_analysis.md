On this page you get to know the automated image analysis.
The images are evaluated without user input and may be scaled. 
Scaling the images will ensure estimated muscle architectural parameters are converted to centimetre units.
For this type of analysis, single images (not videos) are a prerequisite.
These images should be contained in a single folder, like in the “DL_Track_US_example/images” folder.

If you haven’t downloaded this folder yet,
please do so now (link: [DL_Track_US - Examples & Models](https://osf.io/7mjsc/?view_only=)).
Unzip the folder and put it somewhere accessible.

## 1. Creating Image Directory & FlipFlag.txt File

- All images you want to analyze should be placed in one folder.
- The “DL_Track_US_example/images“ folder contains <span style="color: #2eaf66;">**4 images**</span> and 
a <span style="color: #f97e25;">**flip_flag.txt**</span> file.
- It is not required to have the flip_flag.txt file in the same folder as the images to be analysed, but it is convenient.

![image directory](md_graphics\aia\image_directory.PNG)

- Lets take a closer look at the flip_flag.txt file.
- For every image there must be a <span style="color: #a34ba1;">**flip-flag**</span>. 
- The flip-flag determines if an image is flipped during analysis or not. A “0” stands for no flipping, whereas “1” means flip the image.
- If the number of flip-flags and images doesn’t match, an error is raised.

![flip flag](md_graphics\aia\flip_flag.PNG)

- Another possible way to specify is displayed below. This is relevant when
multiple subfolders are included, as each line then represents a subfolder.

![flip flag 2](md_graphics\aia\flip_flag_2.PNG)

- None of the example images must be flipped. Their fascicle orientation is
correct, with fascicles originating at the bottom left and inserting on the
top right.
- Below is a visual representation of a **correct** fascicle orientation. If the
fascicles in your image are orientated differently, please specify a “1” as
a flip-flag for those images.

<img src="\md_graphics\aia\fascial_orientation.PNG" width="600">

## 2. Specifying Input Directories in the GUI

Once the GUI is openend, the first step of every analysis type in DL_Track_US
is to specify the input directories in the graphical user interface (GUI).

- First, specify the path to the folder containing the images to be analysed.
Remember this was the folder “DL_Track_US_example/**images**”.
    - By clicking on the <span style="color: #a34ba1;">**Inputs**</span> button a selection window opens were you
need to select the images folder.  
    - Click <span style="color: #299ed9;">**Select folder**</span> to specify the path in the GUI.

![input folder](md_graphics\aia\input_folder.PNG)
![input folder 2](md_graphics\aia\input_folder_2.PNG)

- Secondly, specify the absolute path to the **aponeurosis neural network**
which is located in the “DL_Track_US_example/**DL_Track_US_models**” Folder.
    - By clicking on the <span style="color: #a34ba1;">**Apo Model**</span> button, a selection window opens were
you need to select the **aponeurosis neural network**.
    - Click <span style="color: #299ed9;">**Open**</span> to specify the path to the **aponeurosis neural network** in the GUI.

![apo model path](md_graphics\aia\apo_model_path.PNG)
![apo model path 2](md_graphics\aia\apo_model_path_2.PNG)

- Thirdly, specify the absolute path to the **fascicle neural network** which is also located in
the “DL_Track_US_example/**DL_Track_US_models**” folder.
    - By clicking on the <span style="color: #a34ba1;">**Fasc Model**</span> button, a selection window opens were
you need to select the **fascicle neural network**.
    - Click <span style="color: #299ed9;">**Open**</span> to specify the path to the **fascicle neural network** in the GUI.

![fasc model path](md_graphics\aia\fasc_model_path.PNG)
![fasc model path 2](md_graphics\aia\fasc_model_path_2.PNG)

In the next section you will specify all relevant analysis parameters,
including the analysis type. We will also explain what each parameter is used
for.

## 3. Specifying Analysis Parameters

As a first step, you will select the right analysis type in the GUI.

- Select <span style="color: #a34ba1;">**image**</span> in the dropdown-menu.

![analysis type image](md_graphics\aia\analysis_type_image.PNG)

Next, you need to specify the **Image Type**.

- The ending of the Image Type must match the ending of your images,
otherwise no files are found by DL_Track_US.
- You can either select a pre-specified ending from the dropdown list or
type in your own ending.
- Please keep the formatting similar to the Image Types provided in
the dropdown list.
- All the images in the “DL_Track_US_example/images” folder
are of the Image Type **“.tif”**. Thus, you should select the <span style="color: #a34ba1;">**“/*.tif”**</span> Image Type.

![image type](md_graphics\aia\image_type.PNG)

Subsequently, you need to specify the image **Scaling Type**.

- Scaling in general has the huge advantage that the resulting estimated
muscle architectural features are in centimetre units rather than pixel
units.
- There are three Scyling Types in the DL_Track_US package.
- For this tutorial however, you will select the <span style="color: #a34ba1;">**"None"**</span> option as
displayed below.

![scaling type](md_graphics\aia\scaling_type.PNG)

- Another Scaling Type is <span style="color: #a34ba1;">**“Bar”**</span>. This Scaling Type is
only applicable if there are scaling bars in the right side of the
ultrasonography image:
- The <span style="color: #00bfba;">**scaling bars**</span> do not need to look exactly like the ones in the image below.
They just need to be next to the image and clearly separated from each other.
- We advise you to try this **Scaling type** on a few of your images and find
out for yourself if it works.
- Files that cannot be analysed with this Scaling type will be recorded in
an failed_images.txt file in the image input folder.

![scaling type bar](md_graphics\aia\scaling_type_bar.PNG)

The last of the three Scaling Types is  <span style="color: #a34ba1;">**“Manual”**</span>. This **Scaling Type** requires input from the user.

- Whenever you use “**Bar**” or “**Manual**” as your Scaling Type, make sure
that the minimal distance between the scaling bars or the known
distance between the manually specified points is represented in
the <span style="color: #2eaf66;">**Spacing**</span> parameter.
- Select the Spacing parameter from the dropdown list as 5, 10, 15 or 20
millimetre. For this tutorial it is not necessary to select anything, as the
Spacing parameter is not used during an analysis with Scaling Type “**None**”.

<img src="\md_graphics\aia\spacing_3.png" width="600">

- When you choose “Manual” as your Scaling type, you need to manually place **two points** on
the image using the left mouse button.
- In order to do this, you need to click <span style="color: #a34ba1;">**Calibrate**</span>.

<img src="\md_graphics\aia\calibrate_button.png" width="600">

- Then, just click one time with your left mouse button to record the first point
(a red dot will apear).
- Place the second point at a known distance of either 5, 10, 15 or 20
millimetre.
- Afterwards, click <span style="color: #a34ba1;">**Confirm**</span>.

<img src="\md_graphics\aia\calibrate.png" width="600">

After confirming a <span style="color: #299ed9;">**messagebox**</span> should appear with the distance of the spacing parameter in pixels.

<img src="\md_graphics\aia\calibration_result.png" width="600">

- In version 0.2.1 we introduced a new feature to DL_Track_US, called the **Filter Fascicle** option.
- Here, you have two options, <span style="color: #a34ba1;">**“YES”**</span> or <span style="color: #a34ba1;">**“NO”**</span>.
- Using **“YES”** all fascicles that overlap will be removed.

![filter fascicles](md_graphics\aia\filter_fascicles.PNG)

Here are some results demonstrating the difference.

![filter fascicles 2](md_graphics\aia\filter_fascicles_2.PNG)

As a next Stept you need to specify the absolute path to the **flip_flag.txt** file.

- By clicking the <span style="color: #a34ba1;">**Flip Flags**</span> button, a dialogue will pop up and you can
select the <span style="color: #f97e25;">**flip_flag.txt**</span> file.
- In this example, the flip_flag.txt file is located at “DL_Track_US_example/images”.
- Remember, the amount of flip-flags in the flip_flag.txt file must equal the
amount of images in the images folder.

![flip flag button](md_graphics\aia\flip_flag_button.PNG)
![flip flag location](md_graphics\aia\flip_flag_location.PNG)

## 4. Adjusting Settings

As a last step, you need to adjust the settings for the
aponeurosis and fascicle neural networks. If you click on the <span style="color: #a34ba1;">**settings wheel**</span> a 
python script with the name "settings.py" opens up in your default text editor. On this page, all parameters used by the aponeurosis and
fascicles neural networks during inference are specified. The default values are always listed on the right hand side of the parameters. The settings are explained in detail at the top of the settings.py file.

![analysis parameters](md_graphics\aia\analysis_parameters.PNG)
![settings.py](md_graphics\aia\settings_py.PNG)

- The **aponeurosis detection threshold** determines the threshold of the minimal
acceptable probability by which a pixel is predicted as aponeurosis. The lower,
the more pixels will be classified as aponeurosis.

- Changing the **aponeurosis length threshold** will result in longer or shorter structures
detected as aponeurosis.

- The **fascicle detection threshold** and the fascicle lenght threshold are the same thing, just for the fasicles.

- The **minimal muscle width** determines the minimal acceptable distance between superficial and deep aponeurosis.

- **Minimal and Maximal Pennation** describe the respective minimal and maximal
pennation angle that is physiologically possible in the analysed image/muscle.

- The **fascile calculation method** determines the approach by which the fascile length is calculated. This can either be linear_extrapolation, curve_polyfitting, curve_connect_linear, curve_connect_poly or orientation_map.

- The lower the **fascile contour tolerance**, the shorter the minimal acceptable length of
detected fascicle segments to be included in the results.

- The lower the **aponeurosis distance tolerance**, the nearer a fascicle fragment must be to the aponeurosis. This increases certainty of pennation angle calculation and extrapolation.

For this tutorial, you can leave all parameters the way they are. You can set the parameters by saving the python file. Adapt
these parameters according to your images in analyses. For future analyses, it’s best you test the ideal parameter configuration in
a small sample of your images prior to the actual analysis. If you should somehow distruct the settings.py file there is a backup called _backup_settings.py.

## 5. Running / Breaking DL_Track_US

- By clicking the <span style="color: #a34ba1;">**Run**</span> button in the main GUI window, you can start the
analysis.
- Moreover, you can see that there is a <span style="color: #299ed9;">**Break**</span> button placed in the GUI as
well.
- Clicking the <span style="color: #299ed9;">**Break**</span> button allows you to stop the analysis at any point.
The currently evaluated image will be processed and then the
analysis isterminated.

![running breaking](md_graphics\aia\running_breaking.PNG)

After running the analyis the three lines are displayed in the line graph:

- Median Fascicle Length
- Median Filtered Fascicle Length
- Filtered Median Fascicle Length

<!-- Beschreibung Paul -->

<img src="\md_graphics\aia\plotted_results.png">

- In the “DL_Track_US_example/images” folder, you will see that two files
will be / have been created, <span style="color: #f97e25;">**ResultImages.pdf**</span> and <span style="color: #2eaf66;">**Results.xlsx**</span>.
- The <span style="color: #f97e25;">**ResultImages.pdf**</span> file contains each original input image and
concomitant prediction results with fascicles and aponeurosis displayed.
- The <span style="color: #2eaf66;">**Results.xlsx**</span> file contains the actual architectural parameter estimates
for each input image. There, the median value of all detected muscle
fascicle length and pennation angles as well a the calculated muscle
thickness will be displayed. Each input image is displayed in a
separate row.
- Note that the <span style="color: #f97e25;">**ResultImages.pdf**</span> file can be opened only after the
<span style="color: #2eaf66;">**Results.xlsx**</span> was created.

![results](md_graphics\aia\results.PNG)

You have now completed the DL_Track_US tutorial for automated image
analysis! There is one more thing though, error handling. Take a look at the
next section to get more information.

## 6. Error Handling

Whenever an error occurs during the analysis process, the DL_Track_US GUI
will open a <span style="color: #299ed9;">**messagebox**</span>. This looks always similar to this:

![error](md_graphics\aia\error.PNG)

We tried to formulate these messageboxes as concise as possible. Just
follow their instructions to fix the error and run the analysis anew. In case
an error occurs that is not caught by an error messagebox, don’t hesitate to
report this in the Q&A section in the [DL_Track_US discussion forum](https://github.com/PaulRitsche/DLTrack/discussions/categories/q-a).
Please take a look here how do best do this.