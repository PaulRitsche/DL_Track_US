This section of the tutorial covers the automated video analysis. The videos
are evaluated without user input and may be scaled. The videos should be
contained in a single folder, like in the “DL_Track_US_example/videos” folder.
If you haven’t downloaded this folder, please do so now (link:
DL_Track_US - Examples & Models). Unzip the folder and put it
somewhere accessible. We will make use of the included example files
extensively during this tutorial. The automated video analysis is very
similar to the automated image analysis. In fact, the inputted video is
analysed frame by frame and each frame is therefore treated like
an independent image. Moreover, only few analysis parameters are
different between both analysis types. Once the analysis of the video file is
finished, a „proc.avi“ file will be created at the directoy of the input video.
The „proc.avi“ file can be openend with, i.e., VLC-Player on windows and
Omni-Player on macOS. In the next few pages, we will look at every
required step to successfully perform automated video analysis
with DL_Track_US.

## 1. Creating Video and Network Directories

- In order for DL_Track_US to recognize your videos, they should best be
in a single folder.

![video folder](md_graphics\ava\video_folder.PNG)

- The “DL_Track_US_example/videos“ folder contains <span style="color: #2eaf66;">**one video**</span>.
- The pre-trained aponeurosis and fascicle neural networks are located in the “DL_Track_US_example/DL_Track_US_models” folder. 
- You can make use of these neural networks later as well, when you analyse your own videos outside of this tutorial.

## 2. Specifying Input Directories in the GUI

Once the GUI is openend, the first step of every analysis type in DL_Track_US
is to specify the input directories in the graphical user interface (GUI).

- Start the analysis with specifying the path to the folder containing the video to be analysed.
- Remember this was the folder "DL_Track_US_example/video". By clicking on the <span style="color: #a34ba1;">**Inputs**</span> button in the GUI a selection window opens were you need to select the images folder.
- Click **select folder** to specify the path in the GUI.

![inputs button](md_graphics\ava\inputs_button.PNG)

Now, you will specify the absolute path to the aponeurosis neural
network.

- Remember that the model is in the “DL_Track_US_example/models”
folder.
- By clicking on the <span style="color: #a34ba1;">**Apo Model button**</span> in the GUI a selection window
opens were you need to select the aponeurosis neural network in the
models folder.
- Click <span style="color: #299ed9;">**open**</span> to specify the path to the aponeurosis neural network in the
GUI

![apo model button](md_graphics\ava\apo_model_button.png)
![apo model](md_graphics\ava\apo_model.png)

Next, you will specify the absolute path to the fascicle neural
network.

- The model is in the “DL_Track_US_example/models” folder.
- By clicking on the <span style="color: #a34ba1;">**Fasc Model button**</span> in the GUI a selection window
opens were you need to select the fascicle neural network in the
models folder.
- Click <span style="color: #299ed9;">**open**</span> to specify the path to the fascicle neural network in the GUI.

![fasc model button](md_graphics\ava\fasc_model_button.png)
![fasc model](md_graphics\ava\fasc_model.png)

In the next section you will specify all relevant analysis parameters,
including the analysis type. We will also explain what each parameter is used
for.

## 3. Specifying Relevant Parameters

As a first step, you will select the right analysis type in the GUI.

- Please select <span style="color: #a34ba1;">**Video**</span> from the dropdown-menu.

![analysis type video](md_graphics\ava\analysis_type_video.png)

You now need to specify the **Video Type**.

- The ending of the Video Type must match the ending of your videos,
otherwise no files are found by DL_Track_US.
- You can either select a pre-specified ending from the dropdown list or
type your own ending.
- Please keep the formatting similar to those Video Type provided in the
dropdown list.
- The video in the “DL_Track_US_example/video” folder are of the Video
Type “.mp4”. Thus, you should select the <span style="color: #a34ba1;">**“/*.mp4”**</span> Video Type.

![video type](md_graphics\ava\video_type.png)

Subsequently, you need to specify the video **Scaling Type**.

- Scaling in general has the advantage that the resulting estimated
muscle architectural features are in centimetre units rather than pixel
units.
- There are two Scaling Types in the DL_Track_US package.
- For this tutorial however, you will select the <span style="color: #a34ba1;">**“None”**</span> option as
displayed below. We will explain the other Scaling Type on the next.

![scaling type](md_graphics\ava\scaling_type.png)

The other Scaling Types is “Manual”.

- This Scaling Type requires input from the user.
- When you choose <span style="color: #a34ba1;">**“Manual”**</span> as your Scaling type, you need to
manually place <span style="color: #01dcd6;">**two points**</span> on the first video frame using the left mouse
button.
- This step is similar to the “Manual” scaling option for automated
and manual image analysis.

![scaling type 2](md_graphics\ava\scaling_type_2.png)

- Just click one time with your left mouse button to record the <span style="color: #01dcd6;">**first point**</span>
 (nothing will be displayed on the video frames during actual
analysis).
- Place the <span style="color: #01dcd6;">**second point**</span> at a known distance of either 5, 10, 15 or 20
millimetre.
- The distance you chose must be represented in the Scaling (see next
page) parameter in the GUI
- Whenever you use “Manual” as your Scaling Type, please make sure that
the minimum distance between the scaling bars or the known distance
between your manually specified points is represented in the <span style="color: #a34ba1;">**Spacing**</span>
parameter.

![spacing](md_graphics\ava\spacing.png)

- You can select the <span style="color: #a34ba1;">**Spacing**</span> parameter only from the dropdown list as 5,
10, 15 or 20 millimetre. For this tutorial it is not necessary to select
anything, as the Spacing parameter is not used during an analysis
with Scaling Type “None”.
- The minimal <span style="color: #01dcd6;">**distance**</span> is simply the distance in millimeter between the
two nearest scaling bars in the frame. If you do not know this distance,
please use “Manual” or “None” Scaling Type. For example in the
frame from before, the distance between the nearest bars is 5 millimetre.

![spacing 2](md_graphics\ava\spacing_2.png)

- In version 0.2.1 we introduced a new feature to DL_Track_US, called the **Filter Fascicle** option.
- Here, you have two options, <span style="color: #a34ba1;">**“YES”**</span> or <span style="color: #a34ba1;">**“NO”**</span>.
- Using “YES” all fascicles that overlap will be removed

![filter fasciles](md_graphics\ava\filter_fascicles.png)

Here are some results demonstrating the difference in an image, for video
frames the effect would be similar.

![difference filter fasciles](md_graphics\ava\difference_filter_fascicles.png)

Another parameter that you need to specify is the <span style="color: #a34ba1;">**Flip Options**</span> parameters.

- The Flip Options parameter determines if the whole video is flipped
along the vertical axis. “Flip” stands for flipping the video, whereas
“Don’t Flip” means please do not flip the video.
- The example video must be flipped.
- Its fascicle orientation is incorrect, with fascicles originating at the
bottom right and inserting on the top left.
- Below is a visual representation of a correct fascicle orientation.
- The fascicles are originating at the bottom left and are inserting on the top
right.
- Note that all videos in the specified input folder, in this case the
DL_Track_US_example/video” folder, MUST have the same fascicle
orientation, since the Flip Option is applied to all of them.

![flip option](md_graphics\ava\flip_option.png)
![flip option 2](md_graphics\ava\flip_option_2.png)

The next step is to specify the <span style="color: #a34ba1;">**Frame Steps**</span>.

- You can either select a pre-specified Frame Step from the dropdown list
or type your Frame Step.
- The Frame Step is used during the analysis as a step size while iterating
through all the frames in a video.
- In this tutorial you should specify a Frame Step of 1. This means that
every video frame is analysed. With a Frame Step of 3, every 3rd
frame is analysed. With a Frame Step of 10, every 10th frame an so
on.
- Although information is lost when you skip frames during the analysis, it
also reducesthe overall analysis time.

![step size](md_graphics\ava\step_size.png)

## 4. Specifying Analysis Parameters

<!-- Noch einfügen von Automated Image Analysis -->

## 5. Running / Breaking DL_Track_US

- By clicking the <span style="color: #a34ba1;">**Run**</span> button in the main GUI window, you can start the
analysis.

![running breaking](md_graphics\ava\running_breaking.png)

- Moreover, you can see that there is a <span style="color: #299ed9;">**Break**</span> button placed in the GUI as
well.
- Clicking the <span style="color: #299ed9;">**Break**</span> button allows you to stop the analysis at any point.
The currently evaluated image will be processed and then the
analysis isterminated.

Subsequently to clicking the Run button in the main GUI, navigate again to the
“DL_Track_US_example/video”.

- You will see that two files will be / have been created, <span style="color: #299ed9;">**calf_raise_proc.avi**</span> 
and <span style="color: #2eaf66;">**calf_raise.xlsx**</span>.
- The calf_raise_proc.avi file contains each the input video with overlaid
segmented fascicles and aponeurosis. This file allows you to visually
inspect the model outputs.
- The calf_raise.xlsx file contains the actual architectural parameter
estimates for each video frame. There, all detected muscle fascicle
lengths and pennation angles as well a the calculated muscle thickness will
be displayed. Each video frame is displayed in a separate row.
- Note that the calf_raise_proc.avi file can be opened only after the
calf_raise.xlsx. was created.

![video folder after](md_graphics\ava\video_folder_after.png)

## 6. Error Handling

Whenever an error occurs during the analysis process, the DL_Track_US GUI
will open a <span style="color: #299ed9;">**messagebox**</span>. This looks always similar to this:

![error](md_graphics\aia\error.PNG)

We tried to formulate these messageboxes as concise as possible. Just
follow their instructions to fix the error and run the analysis anew. In case
an error occurs that is not caught by an error messagebox, don’t hesitate to
report this in the Q&A section in the [DL_Track_US discussion forum](https://github.com/PaulRitsche/DLTrack/discussions/categories/q-a).
Please take a look here how do best do this.