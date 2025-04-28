# ✂️ Remove Video Parts

In DL_Track_US you can **remove parts of a video** to:

- Improve segmentation accuracy:  
  Sometimes the **superficial aponeurosis** can be wrongly segmented if the **skin** or **subcutaneous tissue** is thick or highly echogenic.  
  Cropping out the upper regions can avoid these errors.
- Support **data anonymization**:  
  Sensitive or identifying areas can be cropped to protect subject privacy.

✅ This is especially useful **before running automated segmentation**.

---

## 1. Accessing the Remove Video Parts Tool

- Once the GUI is open, click on the <span style="color: #a34ba1;">**Advanced Methods**</span> button.
- In the <span style="color: #2eaf66;">**Select Method**</span> dropdown, select “**Crop Video**”.
- A separate **Crop Video Window** will pop up.

<img src="\md_graphics\remove_video_parts\advanced_methods.png">
<img src="\md_graphics\remove_video_parts\select_method.png" width="120">
<img src="\md_graphics\remove_video_parts\remove_video_parts_window.png" width="450">

---

## 2. Loading the Video

- Click the <span style="color: #a34ba1;">**Load Video**</span> button.
- Select the video file you want to crop.

<img src="\md_graphics\remove_video_parts\load_video_button.png" width="450">

After successfully loading the video, the UI will look like this:

- Use the **yellow slider** to scroll through the video frames.
- Click and drag **with the left mouse button** to **select the part to keep**.

<img src="\md_graphics\remove_video_parts\selected_area.png" width="600">

Your result (after re-loading) will look like this:

<img src="\md_graphics\remove_video_parts\resulting_video.png" width="600">

---

## 3. Saving the Video

- Click the <span style="color: #a34ba1;">**Browse**</span> button to select the **output folder** where the cropped video will be saved.
- Finally, click <span style="color: #2eaf66;">**Remove Parts**</span> to save the cropped video.

<img src="\md_graphics\remove_video_parts\output_path.png" width="600">

---

> ✅ That's it! Now your video is ready for analysis or anonymized storage.

