# 🧪 Testing Model Training

This page explains how to **test the model training** using the DL_Track_US GUI.

---

## 1. Before You Start

- A **working GPU** is highly recommended; otherwise, model training will take significantly longer.
- Instructions to setup the GUI and environment are found in the [Installation Guidelines](https://github.com/PaulRitsche/DL_Track_US).

- The **test training images and masks** you need are located in:
  - `DL_Track_US_example/tests/model_training/`

---

## 2. Important Setup Instructions

For this test, ensure the following:

- Click on <span style="color: #a34ba1;">**Advanced Methods**</span> and select **"Train Model"** in the dropdown menu.
- Ignore the main GUI window for now — you will only use the **Model Training window**.
- Use the correct training **<span style="color: #2eaf66;">images</span>**:
  - `DL_Track_US_example/tests/model_training/apo_img_example`
- Use the correct training **<span style="color: #a34ba1;">masks</span>**:
  - `DL_Track_US_example/tests/model_training/apo_mask_example`
- Keep the **<span style="color: #299ed9;">parameter settings</span>** exactly as shown.
- **Critical:** Set the number of <span style="color: #f97e25;">**Epochs to 3**</span> (for quick test training).

<img src="\md_graphics\model_training_test\test_setup.png" width="600">

---

## 3. Starting the Training

- After setting all parameters, click <span style="color: #a34ba1;">**Start Training**</span>.

- During the process, you will encounter **several messageboxes**:
  - Confirm each by clicking **OK**.
  - These confirm that:
    - Images and masks have been loaded.
    - Model compilation was successful.
    - Training completed successfully.

---

## 4. After Training

Once training finishes, you should find **three new files** in your selected **output folder**:

- 📄 **Test_apo.xlsx** — Training summary file
- 📄 **Test_apo.h5** — The trained model
- 📄 **Training_results.tif** — A plot of the loss curve over epochs

---

> 📝 **Note:**  
> Because neural network training includes uncertainty, your results (e.g., final loss values) may slightly differ from ours.  
>  
> If the three files are generated correctly, it means your DL_Track_US installation **works properly for model training**!

---
