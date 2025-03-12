The next and last analysis type this tutorial covers is the manual video analysis.
The video frames are evaluated manually by drawing the muscle thickness,
fascicle length and pennation angles directly on the Image. For this type of
analysis, single videos are a prerequisite. These videos should be
contained in a single folder, like in the
“DL_Track_US_example/videos_manual” folder. If you haven’t downloaded this
folder, please do so now (link: [DL_Track_US - Examples & Models](https://osf.io/7mjsc/?view_only=)). Unzip
the folder and put it somewhere accessible. We will make use of the
included example files extensively during this tutorial. The manual video
analysis type is identical to the manual image analysis type. The only
difference is that the absolute video path must be specified instead of the
File Type. The video is first converted and all the contained frames are
separately stored as single images. Then, each frame image is analysed
separately. In the next few pages, we will look at every required step to
successfully perform manual video analysis with DL_Track_US.

## 1. Creating a Video Directory

All videos to be analyzed should be in a single folder.

- The “DL_Track_US_example/video_manual“ folder contains one <span style="color: #2eaf66;">**video file**</span>.

![video folder](md_graphics\mva\video_folder.PNG)

## 2. Specifying Input Directory in the GUI

- Please select  <span style="color: #a34ba1;">**video_manual**</span> from the dropdown-menu.

![analysis type video manual](md_graphics\mva\analysis_type_video_manual.png)

Next, you need to specify the absolute File Path of the video file to be
analysed.

- The example video file is placed in the
“DL_Track_US_example/video_manual” folder.
- By clicking on the <span style="color: #a34ba1;">**Video Path**</span> button in the GUI, a selection window
opens were you need to select the example video file in the
video_manual.
- Click open to specify the path to the video file in the GUI.

![video path](md_graphics\mva\video_path.png)

You can start the analysis by clicking the <span style="color: #a34ba1;">**Run**</span> button in the main GUI

![run button](md_graphics\mva\run_button.png)

- Once you clicked the Run button, the “Manual Analysis window” will pop
up.
- From here, all further steps are identical with the manual image analysis.
- The only difference though is that in the folder of the inputted video, a
new folder is created containing all the single image frames.
- The scaling of the image, extending of the aponeuroses, single segment
muscle thickness measurements, three segment muscle fascicle
measurement and two segment pennation angle measurement are
identical.
- Saving the results (with the very important button), continuing to
the next image frame, terminating the analysis process and error
handling is identical.
- Therefore, we kindly refer you to the [Manual Image Analysis](manual_image_analysis.md) to see how all the architectural parameters are analysed.