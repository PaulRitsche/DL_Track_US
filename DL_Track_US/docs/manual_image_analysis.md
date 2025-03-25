The next analysis type this tutorial covers is the manual image analysis. The
images are evaluated manually by drawing the muscle thickness, fascicle length
and pennation angles directly on the Image. For this type of analysis, single
images (not videos) are a prerequisite. These images should be
contained in a single folder, like in the
“DL_Track_US_example/images_manual” folder. 

If you haven’t downloaded
this folder, please do so now (link: [DL_Track_US - Examples & Models](https://osf.io/7mjsc/?view_only=)). Unzip
the folder and put it somewhere accessible. We will make use of the
included example file included in the DL_Track_US_examples folder
extensively during this tutorial. In the next few pages, we will look at every
required step to successfully perform manual image analysis with DL_Track_US.

## 1. Creating Image Directory

All **images** to be analyzed should be in a single folder.

![image folder](md_graphics\mia\image_folder.PNG)

- The “DL_Track_US/image_manual“ folder contains <span style="color:#2eaf66;">**2 images**</span>.
- In contrast to automated image analysis, you do not need a flip_flag.txt file nor do you need neural networks that do predictions. 
- In manual image analysis, you are the neural network.
- The next step is to specify the input directory in the GUI.

## 2. Specifying Input Directories in the GUI

- You will begin with specifying the path to the folder containing the images to be analysed, the “DL_Track_US_example/images_manual” folder.
- By clicking on the <span style="color: #a34ba1;">**Inputs**</span> button in the GUI a selection window opens were you need to select the images folder.
- Click **Select folder** to specify the path in the GUI.

![input button](md_graphics\mia\input_button.PNG)

- Once that is done, the selected folder will be displayed below the **Inputs** field and you can start to specify the relevant parameters for the analysis.

## 3. Specifying Analysis Parameters

Please select <span style="color: #a34ba1;">**image_manual**</span> from the dropdown-menu.

![select image manual](md_graphics\mia\select_image_manual.PNG)

Next, you need to specify the **Image Type**.

- The ending of the Image Type must match the ending of your images,
otherwise no files are found by DL_Track_US.
- You can either select a pre-specified ending from the dropdown list or
type your own ending.
- Please keep the formatting similar to those Image Types provided in the dropdown list.
- All the images in the “DL_Track_US_example/images_manual” folder
are of the Image Type **“.tif”**.
- Thus, you should select the <span style="color: #a34ba1;">**“/*.tif”**</span> Image Type as shown below.

![image type](md_graphics\mia\image_type.PNG)

- Once you have specified the Image Type, you can start with the
analysis of the images contained in the
“DL_Track_US_example/images_manual” folder.
- You can start the analysis by clicking the <span style="color: #a34ba1;">**Run**</span> button in the main GUI.

![run button](md_graphics\mia\run_button.PNG)

- Take a look at the next chapter to see how to continue in the “Manual Analysis window” that pops up.

## 4. Manual Analysis of Image

After clicking the Run button in the main GUI, the “Manual Analysis
window” opens. Here is how it looks like:

![manual analysis window](md_graphics\mia\manual_analysis_window.PNG)

Important to note:

- The actual lines you draw are not used during the computation of the
architectural parameters.
- The start- and endpoints of each line are relevant.
- The start point is defined as the point where you clicked the left mouse
button to start drawing the line.
- The endpoint is defined as the point where you released the left mouse
button to stop drawing the line.
- The line follows the cursor as long as the left mouse button is pressed.
- The calculations of the scaling line length, muscle thickness, fascicle length
and pennation angle are dependent on the number of specified
lines/segments.
- **Do NOT click somewhere random** on the image during the analysis of
a parameter and exactly follow the instructions. If additional clicks
happened, start the analysis new by selecting the radiobutton
representing the parameter again.
- If you do not follow the instructions presented in this tutorial, we
cannot guarantee the correctness of the analysis results.

First of all, you will scale the images manually so that the calculated
architectural parameters are returned in centimetre rather than pixel units.

- Draw a one centimetre long straight line in the image.
- The distance of one centimetre is usually recognizable in the scaling
bars in the image.
- You can initiate the scaling process by selecting the <span style="color: #a34ba1;">**Scale Image**</span>
radiobutton in the “Manual Analysis window”.
- A <span style="color: #299ed9;">**messagebox**</span> will appear advising you what to do.

![scaling messagebox](md_graphics\mia\scaling_messagebox.PNG)

The <span style="color: #f97e25;">**drawn line**</span> should look like this.

![drawn line](md_graphics\mia\drawn_line.PNG)

As a next step you have the option to extend the muscle aponeuroses to ease
the extrapolation of fascicles extending outside of the image.

- Select the <span style="color: #a34ba1;">**Draw Aponeurosis**</span> button in the “Manual Analysis window”
and draw the <span style="color: #f97e25;">**aponeurosis lines**</span> on the image as shown below.
- A <span style="color: #299ed9;">**messagebox**</span> will appear advising you what to do.

![draw aponeuroses](md_graphics\mia\draw_aponeuroses.PNG)
![draw aponeuroses 2](md_graphics\mia\draw_aponeuroses_2.PNG)

Now you can start with the muscle thickness assessment.

- Select the <span style="color: #a34ba1;">**Muscle Thickness**</span> radiobutton in the “Manual Analysis
window”.
- A <span style="color: #299ed9;">**messagebox**</span> will appear advising you what to do.
- Draw <span style="color: #f97e25;">**three straight lines**</span> reaching from the superficial to the deep
aponeurosis in the middle right and left portion of the muscle image.

![muscle thickness](md_graphics\mia\muscle_thickness.PNG)
![muscle thickness 2](md_graphics\mia\muscle_thickness_2.PNG)

Next you can mark single fascicles on the image.

- Select the <span style="color: #a34ba1;">**Muscle Fascicles**</span> radiobutton in the “Manual Analysis
window”.
- A <span style="color: #299ed9;">**messagebox**</span> will appear advising you what to do.
- Draw at least three <span style="color: #f97e25;">**fascicles**</span> per image in different regions of the image.
- It is possible to extrapolate the <span style="color: #f97e25;">**fascicles**</span> outside of the image region.
- Each <span style="color: #f97e25;">**fascicles**</span> MUST consist of three segments.
- Do not draw more or less segments per <span style="color: #f97e25;">**fascicles**</span> and pay attention to avoid
any extra unwanted mouse clicks.
- One segment **MUST** start where the previous segment ended.
- Take a look at the image sequence below to see how it is done:

![muscle fasciles](md_graphics\mia\muscle_fasciles.PNG)
![fascile segments](md_graphics\mia\fascile_segments.PNG)

Next you can manually analyse the pennation angle.

- Select the radiobutton <span style="color: #a34ba1;">**Pennation Angle**</span>.
- A <span style="color:  #299ed9;">**messagebox**</span> will appear advising you what to do.
- Draw at least three <span style="color: #f97e25;">**pennation angles**</span> per image at different regions of
the image.
- Each drawn pennation angles MUST consist of two segments. The first
segment should follow the orientation of the fascicle, the second
segment should follow the orientation of the deep aponeurosis. The
segments should both originate at the insertion of the fascicle in the
deep aponeurosis.
- Please pay attention to avoid unwanted clicks on the image.

![pennation angle](md_graphics\mia\pennation_angle.PNG)
![pennation angle 3](md_graphics\mia\pennation_angle_3.PNG)

## 5. Saving / Breaking / Next Image

There are three buttons in the “Manual Analysis window” left to explain. 
The first button is the <span style="color: #a34ba1;">**Save Results**</span> button.

- The Save Results button is a very important button!
- Press the Save Results button once you have analyzed all parameters
that you wanted to analyze and before continuing with the next
image.
- An excel file with the name Manual_Results.xlsx is saved in the
directory of the input images upon pressing the Save Results button.
Therein, all analysis results are stored. Moreover, by pressing the
Save Results, a screenshot of your current analysis is captured and
stored. (Note: The image may look strange, as we can only approximate
the coordinates and size of the manual analysis on your screen.)
- In your case all files are saved in the
“DL_Track_US_example/images_manual” folder.

![save results](md_graphics\mia\save_results.PNG)

The second button we haven’t explained yet is the <span style="color: #299ed9;">**Next Image**</span>
button.

- By clicking this button, you can proceed to the next image in the input
folder (in your case the “DL_Track_US_example/images_manual”
folder).
- Please remember to press the **Save Results** button prior to
proceeding to the next images, otherwise you analysis results for this
image will be lost.
- When the <span style="color: #299ed9;">**Next Image**</span> button is pressed, the displayed image is
updated.

![next image](md_graphics\mia\next_image.PNG)
![next image 2](md_graphics\mia\next_image_2.PNG)

The last button we need to explain is the <span style="color: #2eaf66;">**Break Analysis**</span> button.

- Pressing this button allows you to terminate the analysis and
return to the main GUI window.
- A <span style="color: #299ed9;">**messagebox**</span> will appear asking you if you really want to stop the
analysis.
- Once the <span style="color: #2eaf66;">**Break Analysis**</span> button is pressed and you answered the
messagebox with “YES”, the “Manual Analysis window” will be
automatically closed.

![break analysis](md_graphics\mia\break_analysis.PNG)

When you have saved your results clicking the very important button and
followed our instructions during this tutorial, your input directory 
“DL_Track_US_example/images_manual” should look like this. It should contain
<span style="color: #2eaf66;">**the images**</span>, saved <span style="color: #299ed9;">**screenshots**</span>, as well as the <span style="color: #f97e25;">**Manual_Results.xlsx**</span> file.

![images folder after](md_graphics\mia\images_folder_after.PNG)

## 6. Error Handling

Whenever an error occurs during the manual image analysis process, the
DL_Track_US GUI will open a <span style="color: #299ed9;">**messagebox**</span>. This looks always similar to this:

![error](md_graphics\mia\error.PNG)

We tried to formulate these messageboxes as concise as possible. Just follow
their instructions to fix the error and run the analysis anew. In case an error
occurs that is not caught by an error messagebox, don’t hesitate to report this
in the Q&A section in the [DL_Track_US discussion forum](https://github.com/PaulRitsche/DLTrack/discussions/categories/q-a). Please take a look
[here](https://dltrack.readthedocs.io/en/latest/contribute.html) how do best do this.