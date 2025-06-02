# 🧠 Training Your Own Networks

The DL_Track_US package GUI includes the possibility to **train your own neural networks**.

---

## Why train your own model?

- To create models tailored to **your own dataset**.
- To improve segmentation if the example models don't generalize well enough.
- To learn more about **deep learning** for muscle ultrasound.

> 🚨 It’s **highly recommended** to have a working **GPU setup**; otherwise training can take much longer.  
> Check out [our GitHub repository](https://github.com/PaulRitsche/DL_Track_US) for setup instructions.

If you're new to neural networks, we recommend this [introduction course](https://deeplizard.com/learn/video/gZmobeGL0Yg).

> 📝 **Note:**  
> DL_Track_US UI allows **training** but **not modifying** network architectures!

---

The paired **images** and **labeled masks** needed for training are located in:  
📁 `DL_Track_US_example/model_training`

[Download DL_Track_US Examples & Models](https://osf.io/7mjsc/?view_only=) if you haven’t already.

In this tutorial, we will train a model for **aponeurosis segmentation**.  
Training a **fascicle segmentation** model is identical — only the images and masks would differ.

---

## 1. Data Preparation and Image Labeling

Inside the `DL_Track_US_example/model_training` folder:

- 📁 <span style="color: #2eaf66;">**apo_img_example**</span> → Original images  
- 📁 <span style="color: #a34ba1;">**apo_mask_example**</span> → Corresponding labeled masks

![model training folder](md_graphics/training_your_own_networks/model_training_folder.png)

> ⚡ **IMPORTANT:**  
> Image names and mask names **must match exactly**!

Example:

![image mask](md_graphics/training_your_own_networks/image_mask.png)

---

## 2. Specifying Relevant Directories

- Open the UI.
- Click the <span style="color: #a34ba1;">**Advanced Methods**</span> button.
- In the dropdown, select <span style="color: #2eaf66;">**Train Model**</span>.

![advanced methods](md_graphics/training_your_own_networks/advanced_methods.png)
![select method](md_graphics/training_your_own_networks/select_method.png)
![train_model](md_graphics/training_your_own_networks/train_model.png)

Now specify the directories:

### Select Image Directory

- Click <span style="color: #a34ba1;">**Images**</span>.
- Select `DL_Track_US_example/model_training/apo_img_example`.

![images_button](md_graphics/training_your_own_networks/images_button.png)

---

### Select Mask Directory

- Click <span style="color: #a34ba1;">**Masks**</span>.
- Select `DL_Track_US_example/model_training/apo_mask_example`.

![masks_button](md_graphics/training_your_own_networks/masks_button.png)

---

### Select Output Directory

- Click <span style="color: #a34ba1;">**Output**</span>.
- Choose a folder to save the trained model, loss plots, and CSV results.

![output_button](md_graphics/training_your_own_networks/output_button.png)

---

## 3. Image Augmentation (Optional but Recommended)

Image augmentation artificially increases your dataset size by applying random transformations.

> 🚨 Especially recommended if you have fewer than 1500 images.

- Click <span style="color: #a34ba1;">**Augment Images**</span>.

![augment_button](md_graphics/training_your_own_networks/augment_images_button.png)

A messagebox will notify you once augmentation is complete.

---

## 4. Specifying Training Parameters

- Keep the default settings for this tutorial.
- **NEVER** use just 3 <span style="color: #299ed9;">**epochs**</span> for real training.  
  (3 epochs are okay only for testing.)

![hyperparameters](md_graphics/training_your_own_networks/hyperparameters.png)

Now click:

- <span style="color: #a34ba1;">**Start Training**</span>

![training_button](md_graphics/training_your_own_networks/start_training_button.png)

Three messageboxes will guide you during the training process.

Once training is finished, you’ll find:

- Trained model (`Test_Apo.h5`)
- Training loss plot (`Training_Results.tif`)
- Loss values per epoch (`Test_apo.csv`)

---

## 5. Using Your Own Networks

You can use your trained models like this:

- Click <span style="color: #a34ba1;">**Apo Model**</span> or <span style="color: #a34ba1;">**Fasc Model**</span> in the GUI to load your trained model.

![apo model fasc model](md_graphics/training_your_own_networks/apo_model_fasc_model.png)

> ⚡ **IMPORTANT:**  
> Never use the same images for training and inference.  
> Always **validate on unseen data** to check your model's performance.

If unsure, feel free to ask in our [DL_Track_US Discussion Forum](https://github.com/PaulRitsche/DL_Track_US/discussions/categories/q-a)!

---

## 6. Error Handling

Errors during training trigger a <span style="color: #299ed9;">**messagebox**</span>:

![error handling](md_graphics/training_your_own_networks/error_handling.PNG)

Follow the instructions shown.  
Uncaught errors should be reported in the  
[DL_Track_US Discussion Forum](https://github.com/PaulRitsche/DL_Track_US/discussions/categories/q-a).

[See here](https://dltrack.readthedocs.io/en/latest/contribute.html) for guidance on how to best report errors.

---

## 7. Labeling Your Own Images

To train your networks, you must label images correctly.

> 🛠 We provide a semi-automated script!

You need:

- 📁 Original images folder
- 📁 `output_images` folder
- 📁 `fascicle_masks` folder
- 📁 `aponeurosis_masks` folder

You’ll use [ImageJ/Fiji](https://imagej.net/software/fiji/downloads) and our script:  
🗂 [`DL_Track_US/DL_Track_US/gui_helpers/gui_files/Image_Labeling_DL_Track_US.ijm`](https://github.com/PaulRitsche/DL_Track_US/blob/main/DL_Track_US/gui_helpers/gui_files/Image_Labeling_DL_Track_US.ijm)

Drag the `.ijm` file into a running Fiji/ImageJ window to start.

![fiji](md_graphics/training_your_own_networks/fiji.png)
![labelling](md_graphics/training_your_own_networks/labelling_file.png)

---

### Labeling Steps

1. **Set Directories**:  
   a. Input images  
   b. Aponeurosis masks  
   c. Fascicle masks  
   d. Output images

2. **Label Aponeuroses**:  
   Use polygon tool to select **superficial** and then **deep** aponeurosis.

    ![upper_apo](md_graphics/training_your_own_networks/upper_aponeurosis.png)
    ![lower_apo](md_graphics/training_your_own_networks/lower_aponeurosis.png)

3. **Label Fascicles**:  
   Use segmented line tool for clearly visible fascicle parts only.

    ![fascicles](md_graphics/training_your_own_networks/fascicles.png)

4. **Save and Move to Next Image**.

---

> ✅ You are now ready to create your own high-quality training datasets!

---
