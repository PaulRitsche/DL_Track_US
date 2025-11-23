# 🔍 Inspecting Masks

Data quality is of utmost importance when labeling the images.  
In version **0.2.1** of DL_Track_US, we included an option to **inspect labeled images and corresponding masks**.

---

## 1. Accessing the Mask Inspection Tool

- Open the UI.
- In the main window, click on <span style="color: #a34ba1;">**Advanced Methods**</span>.
- In the <span style="color: #2eaf66;">**Select Method**</span> dropdown, choose **"Inspect Masks"**.
- The **Mask Inspection Window** will open.

![adv_meth](md_graphics\inspecting_masks\advanced_methods.png)
![select_meth](md_graphics\inspecting_masks\select_method.png)
![inspect](md_graphics\inspecting_masks\inspecting_masks.png)

---

## 2. Selecting Relevant Directories

You need to specify three directories:

- 📁 **output_images** — contains the original labeled images
- 📁 **fascicle_masks** — contains fascicle masks
- 📁 **aponeurosis_masks** — contains aponeurosis masks

These folders should have been created during the [Image Labeling](training_your_own_networks.md##7-Labeling-Your-Own-Images) process.

> Fascicle and aponeurosis masks must be inspected **separately**.

### Specify:

- Click <span style="color: #a34ba1;">**Image Dir**</span> to select the `output_images` folder.
- Click <span style="color: #2eaf66;">**Mask Dir**</span> to select the respective masks folder (either fascicle or aponeurosis masks).
- Use the <span style="color: #f97e25;">**Start Index**</span> to choose the starting image number.

![mask_button](md_graphics\inspecting_masks\inspecting_masks_buttons.png)

---

## 3. Starting the Inspection

- Click on <span style="color: #299ed9;">**Inspect Masks**</span> to start inspecting.

One of two things will happen:

---

### Case 1: Everything Matches

- Number of images and masks is equal.
- Naming conventions are correct.
- You will see a <span style="color: #299ed9;">**messagebox**</span> confirming everything is OK.

![no_different](md_graphics\inspecting_masks\no_different_images.png)

Click **OK** to continue to the Mask Inspection GUI.

---

### Case 2: Mismatch Detected

- Number of images and masks is **not equal** and/or
- **Naming is incorrect**.

In this case:

- A **table** appears showing:
  - Incorrect image names
  - Which directory they occur in
  - If the number of files differs

> 💡 **Tip:**  
> Adjust the files according to the table, then restart the inspection.

![differen_images](md_graphics\inspecting_masks\different_images.png)

---

## 4. Using the Mask Inspection UI

Once inspection starts:

![inpsection](md_graphics\inspecting_masks\inspection_gui.png)

In this UI:

- The original images are shown.
- The masks are **overlaid in semi-transparent green**.
- Follow the **instructions displayed inside the GUI** carefully.

> 🛑 **Delete Warning:**  
> Clicking the **Delete** button will **permanently delete** the selected image-mask pair!

We recommend making **backups** of the folders **before** starting inspection,  
especially if you might want to correct masks later.

---

✅ You can now **inspect, validate, and clean your datasets**.
---
