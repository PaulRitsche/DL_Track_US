# 🎥 Automated Video Analysis

This page introduces the **automated video analysis** in DL_Track_US.

- Videos are evaluated **without user input**.
- Videos must be contained in a single folder, e.g., `DL_Track_US_example/videos`.

If you have not downloaded the example folder yet, please do so:  
[DL_Track_US - Examples & Models](https://osf.io/7mjsc/?view_only=).

> 📦 Unzip the folder and save it somewhere easily accessible.

---

The automated video analysis is very similar to automated image analysis:  
Only a few analysis parameters differ between the two types.

After the analysis, a `proc.avi` file will be created in the input video directory.  
It can be opened with VLC Player (Windows) or OmniPlayer (macOS).

---

## 1. Creating Video and Network Directories

- Videos should be stored in a **single folder**.
- The `DL_Track_US_example/videos` folder contains <span style="color: #2eaf66;">**one video**</span>.

![video folder](md_graphics/ava/video_folder.PNG)

---

## 2. Specifying Input Directories in the GUI

Once the GUI is open:

- Click the <span style="color: #a34ba1;">**Inputs**</span> button to specify the folder containing your video.
- Select the `DL_Track_US_example/videos` folder and click **Select folder**.

![inputs button](md_graphics/ava/inputs_button.PNG)

---

Next, specify the aponeurosis model:

- Click the <span style="color: #a34ba1;">**Apo Model**</span> button.
- Select the aponeurosis neural network file from `DL_Track_US_example/models`.
- Click <span style="color: #299ed9;">**Open**</span>.

![apo model button](md_graphics/ava/apo_model_button.png)
![apo model](md_graphics/ava/apo_model.png)

---

Then, specify the fascicle model:

- Click the <span style="color: #a34ba1;">**Fasc Model**</span> button.
- Select the fascicle neural network file from `DL_Track_US_example/models`.
- Click <span style="color: #299ed9;">**Open**</span>.

![fasc model button](md_graphics/ava/fasc_model_button.png)
![fasc model](md_graphics/ava/fasc_model.png)

---

## 3. Specifying Analysis Parameters

### 3.1 Selecting the Analysis Type

- Choose <span style="color: #a34ba1;">**Video**</span> from the dropdown menu.

![analysis type video](md_graphics/ava/analysis_type_video.png)

---

### 3.2 Setting the Video Type

- The file extension must match your videos (e.g., `.mp4`).
- Select or type <span style="color: #a34ba1;">**/*.mp4**</span>.

![video type](md_graphics/ava/video_type.png)

---

### 3.3 Choosing the Scaling Type

- Select <span style="color: #a34ba1;">**None**</span> for this tutorial.

![scaling type](md_graphics/ava/scaling_type.png)

Alternatively, you could use <span style="color: #a34ba1;">**Manual**</span> scaling:

- Place two points on a known distance (5, 10, 15, or 20 mm).
- Click <span style="color: #a34ba1;">**Calibrate**</span>.

<img src="\md_graphics/ava/calibrate_button.png" width="600">
<img src="\md_graphics/ava/calibrate.png" width="600">

After calibration, a <span style="color: #299ed9;">**messagebox**</span> shows the pixel distance:

<img src="\md_graphics/ava/calibration_result.png" width="600">

---

### 3.4 Filtering Fascicles

- Select <span style="color: #a34ba1;">**YES**</span> to remove overlapping fascicles.

![filter fasciles](md_graphics/ava/filter_fascicles.png)

Example difference between filtered and unfiltered:

![difference filter fasciles](md_graphics/ava/difference_filter_fascicles.png)

---

### 3.5 Setting Flip Options

- Choose the appropriate flip setting:
  - **Flip** to flip the video vertically,
  - **Don’t Flip** otherwise.

![flip option](md_graphics/ava/flip_option.png)
![flip option 2](md_graphics/ava/flip_option_2.png)

For the example video, flipping is required to correct fascicle orientation.

---

### 3.6 Setting Frame Steps

- Set Frame Step to **1** (every frame analyzed).
- Larger steps (e.g., 3, 10) reduce computation time but skip frames.

![step size](md_graphics/ava/step_size.png)

---

## 4. Adjusting Settings

Open the settings by clicking the <span style="color: #a34ba1;">**Settings Wheel**</span>.

- A txt script `settings.txt` opens in your default editor.
- Default values are listed; detailed descriptions are available at the top of the file.

![analysis parameters](md_graphics/ava/analysis_parameters.PNG)
![settings.py](md_graphics/ava/settings_py.PNG)

You can find an explanation on all setting in [this chapter](LINK).

---

## 5. Running / Breaking DL_Track_US

- Click the <span style="color: #a34ba1;">**Run**</span> button to start analysis.
- Use the <span style="color: #299ed9;">**Break**</span> button if you need to stop the analysis.

![running breaking](md_graphics/ava/running_breaking.png)

---

Once analysis completes, navigate back to `DL_Track_US_example/videos`.

You will find two new files:

- <span style="color: #299ed9;">**calf_raise_proc.avi**</span>:
  - Video showing overlaid segmentation results.

- <span style="color: #2eaf66;">**calf_raise.xlsx**</span>:
  - Excel file containing estimated muscle parameters (fascicle length, pennation angle, muscle thickness).

<img src="\md_graphics/ava/video_folder_after.png">

---

Line graph results include:

- Median Fascicle Length
- Median Filtered Fascicle Length
- Filtered Median Fascicle Length (based on chosen filter)

<img src="\md_graphics/ava/plotted_results.png">

---

## 6. Error Handling

If an error occurs:

- A <span style="color: #299ed9;">**messagebox**</span> will open to explain the issue.

![error](md_graphics/aia/error.PNG)

We have tried to make all error messages as **concise** and **informative** as possible.  
Simply follow the instructions in the error box and restart the analysis after resolving the issue.

> 💬 **Note:**  
> If an unexpected error occurs that is not caught by a message box,  
> please report it in the [DL_Track_US Discussion Forum - Q&A Section](https://github.com/PaulRitsche/DLTrack/discussions/categories/q-a).

When reporting an issue:

- Please include a description of the problem,
- Steps to reproduce the issue,
- And (if possible) screenshots of the GUI and error message.

---

By following these guidelines, we can continuously improve DL_Track_US.