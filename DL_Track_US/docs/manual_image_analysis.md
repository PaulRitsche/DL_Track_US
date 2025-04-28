# 🖼 Manual Image Analysis

This page covers **manual image analysis** in DL_Track_US.  
In this mode, images are evaluated manually by **drawing muscle thickness, fascicle length, and pennation angles** directly onto the images.

**Important:**  
Manual analysis is applicable only for **single images** (not videos).  
All images must be contained in a single folder, e.g., `DL_Track_US_example/images_manual`.

If you have not downloaded the example folder yet, please do so:  
[DL_Track_US - Examples & Models](https://osf.io/7mjsc/?view_only=).

> 📦 Unzip the folder and save it somewhere easily accessible.

---

## 1. Creating Image Directory

- Place all images to be analyzed into a **single folder**.
- The `DL_Track_US_example/images_manual` folder contains <span style="color:#2eaf66;">**2 images**</span>.

![image folder](md_graphics/mia/image_folder.PNG)

- Unlike automated analysis, you **do not** need a `flip_flag.txt` file or neural networks.
- In manual analysis, **you** are the "neural network."

---

## 2. Specifying Input Directories in the GUI

- Click the <span style="color: #a34ba1;">**Inputs**</span> button in the GUI to open a selection window.
- Choose the `DL_Track_US_example/images_manual` folder.
- Click **Select folder** to confirm.

![input button](md_graphics/mia/input_button.PNG)

---

## 3. Specifying Analysis Parameters

- Select <span style="color: #a34ba1;">**image_manual**</span> from the dropdown menu.

![select image manual](md_graphics/mia/select_image_manual.PNG)

Then specify the **Image Type**:

- The file extension must match your images (e.g., `.tif`).
- Either select it from the dropdown or type it manually.
- For this tutorial, select <span style="color: #a34ba1;">**/*.tif**</span>.

![image type](md_graphics/mia/image_type.PNG)

- After setting the image type, click <span style="color: #a34ba1;">**Run**</span> to start the manual analysis.

![run button](md_graphics/mia/run_button.PNG)

---

## 4. Manual Analysis of Image

After clicking **Run**, the **Manual Analysis window** opens:

![manual analysis window](md_graphics/mia/manual_analysis_window.PNG)

### Important rules:

- The start and end points of each line are critical — not the line itself.
- Start drawing by pressing the left mouse button; end by releasing it.
- Avoid any unwanted clicks!  
  If extra clicks happen, restart the current analysis step.

---

### 4.1 Manual Scaling

- Select <span style="color: #a34ba1;">**Scale Image**</span> in the Manual Analysis window.
- Draw a **1-centimetre straight line** based on scaling bars in the image.
- A <span style="color: #299ed9;">**messagebox**</span> will guide you.

![scaling messagebox](md_graphics/mia/scaling_messagebox.PNG)

Example of the drawn line:

![drawn line](md_graphics/mia/drawn_line.PNG)

---

### 4.2 Drawing Aponeuroses

- Select <span style="color: #a34ba1;">**Draw Aponeurosis**</span> to manually extend aponeuroses.
- A <span style="color: #299ed9;">**messagebox**</span> will instruct you.

![draw aponeuroses](md_graphics/mia/draw_aponeuroses.PNG)
![draw aponeuroses 2](md_graphics/mia/draw_aponeuroses_2.PNG)

---

### 4.3 Measuring Muscle Thickness

- Select <span style="color: #a34ba1;">**Muscle Thickness**</span>.
- Draw <span style="color: #f97e25;">**three straight lines**</span> from superficial to deep aponeurosis across the muscle image.

![muscle thickness](md_graphics/mia/muscle_thickness.PNG)
![muscle thickness 2](md_graphics/mia/muscle_thickness_2.PNG)

---

### 4.4 Drawing Fascicles

- Select <span style="color: #a34ba1;">**Muscle Fascicles**</span>.
- Draw at least <span style="color: #f97e25;">**three fascicles**</span> in different regions.
- Each fascicle must have **three segments**:
  - Each segment must start where the previous segment ended.
- Avoid extra mouse clicks.

![muscle fasciles](md_graphics/mia/muscle_fasciles.PNG)
![fascile segments](md_graphics/mia/fascile_segments.PNG)

---

### 4.5 Measuring Pennation Angles

- Select <span style="color: #a34ba1;">**Pennation Angle**</span>.
- Draw at least <span style="color: #f97e25;">**three pennation angles**</span>:
  - Each must have **two segments**:
    1. Along the fascicle
    2. Along the deep aponeurosis

![pennation angle](md_graphics/mia/pennation_angle.png)
![pennation angle 3](md_graphics/mia/pennation_angle_3.png)

---

## 5. Saving / Breaking / Next Image

### 5.1 Saving Results

- Press the <span style="color: #a34ba1;">**Save Results**</span> button after finishing each image.
- It saves:
  - An Excel file (`Manual_Results.xlsx`)
  - A screenshot of your drawing.

Saved results are stored in `DL_Track_US_example/images_manual`.

![save results](md_graphics/mia/save_results.PNG)

---

### 5.2 Going to Next Image

- Click the <span style="color: #299ed9;">**Next Image**</span> button to proceed.
- Always **save results first** before moving to the next image!

![next image](md_graphics/mia/next_image.PNG)
![next image 2](md_graphics/mia/next_image_2.PNG)

---

### 5.3 Breaking Analysis

- Click <span style="color: #2eaf66;">**Break Analysis**</span> to terminate analysis.
- A <span style="color: #299ed9;">**messagebox**</span> will confirm your choice.
- After confirming, you return to the main GUI.

![break analysis](md_graphics/mia/break_analysis.PNG)

---

After saving all results, your folder should contain:

- <span style="color: #2eaf66;">**Input images**</span>
- Saved <span style="color: #299ed9;">**screenshots**</span>
- The <span style="color: #f97e25;">**Manual_Results.xlsx**</span> file

![images folder after](md_graphics/mia/images_folder_after.PNG)

---

## 6. Error Handling

If any error occurs:

- A <span style="color: #299ed9;">**messagebox**</span> will open explaining the issue.

![error](md_graphics/mia/error.PNG)

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