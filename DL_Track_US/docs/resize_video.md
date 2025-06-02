# ✂️ Resize Video

In DL_Track_US you can **resize a video** so that only the **selected region** remains:

- **Improve segmentation accuracy:**    
  Resizing the video to focus on the muscle region can **improve model performance** by cropping out irrelevant background information.
- **Support data anonymization:**  
  Sensitive areas (e.g., patient IDs or other identifying marks) can be removed entirely by resizing.

> ✅ This is especially useful **before running automated segmentation**!

---

## 1. Accessing / Loading

- Once the GUI is open, click on the <span style="color: #a34ba1;">**Advanced Methods**</span> button.
- In the <span style="color: #2eaf66;">**Select Method**</span> dropdown, select <span style="color: #2eaf66;">**Resize Video**</span>.
- A separate **Resize Video Window** will pop up.

This process is similar to the [remove video parts](./remove_video_parts.md) function. Have a look there if you are uncertain.

---

## 2. Loading the Video

- Click the <span style="color: #a34ba1;">**Load Video**</span> button.
- Select the video file you want to resize.

After successfully loading the video:

- Use the **yellow slider** to scroll through the video frames.
- Click and drag **with the left mouse button** to **select the part you want to keep**.

![resize](md_graphics\resize_video\video_resize_area.png)

Your result (after re-loading) will look like this:

![resize_result](md_graphics\resize_video\video_result.png)


---

## 3. Saving the Resized Video

- Click the <span style="color: #a34ba1;">**Browse**</span> button to select the **output folder**.
- Then click <span style="color: #2eaf66;">**Resize Video**</span> to crop and save the video.

Again, the process is similar to the [remove video parts](./remove_video_parts.md) function. Have a look there if you are uncertain.

---

> 🧠 **Tip:**  
> Resizing the video can speed up the analysis, reduce errors, and protect privacy  
> — especially in clinical settings or publications. **However, we did not include resized images during the training process.**

---
