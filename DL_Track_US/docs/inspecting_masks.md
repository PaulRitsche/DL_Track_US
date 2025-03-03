Data quality is of utmost importance when labelling the images. In version
0.2.1 of DL_Track_US we included an option to inspect the labelled images and
corresponding masks.

- Once you started the GUI and the main GUI window opened, click on the
<span style="color: #a34ba1;">**Advanced Methods**</span> button to select the relevant directories and model
training parameters.
- In the <span style="color: #2eaf66;">**Select Method**</span> Dropdown select “Inspect Masks”. The separate
“Mask Inspection Window” will pop up. We will explain this window on
the next page.

<img src="\md_graphics\inspecting_masks\advanced_methods.png">
<img src="\md_graphics\inspecting_masks\select_method.png" width ="100">
<img src="\md_graphics\inspecting_masks\inspecting_masks.png" width="250">

- First, you need to specify the relevant directories for the image/mask
inspection.
- Three folders are of relevance here, **“output_images”**, **“fascicle_masks”**,
**“aponeurosis_masks”**. They should have been created during the labelling
process we explained in the previous chapter.
- Given that the number of fascicle/aponeurosis masks might differ, you can
inspect both masks separately.
- Specify the directory containing the **“output_images”** clicking the 
<span style="color: #a34ba1;">**Image Dir**</span> button.
- Specify the directory containing the respective “fascicle/aponeurosis
masks” clicking the <span style="color: #2eaf66;">**Mask Dir**</span> button.
- The <span style="color: #f97e25;">**Start Index**</span> allows you to specify the index/number of the image you
want to start inspecting.

<img src="\md_graphics\inspecting_masks\inspecting_masks_buttons.png">

- Clicking on the <span style="color: #299ed9;">**Inspect Masks**</span> button, you will start the
inspection process.

Given that the number of images and masks as well as the names of images
and masks must be the same, one of two things will happen next:

1. Number of images and masks is equal and naming is correct. You will see a
<span style="color: #299ed9;">**messagebox**</span> telling you so. Click OK to continue.

<img src="\md_graphics\inspecting_masks\no_different_images.png">

2. Number of images and masks is not equal and/or naming is not correct. A
table will appear telling you which image names are incorrect, in which
directory they occur and if the number of images differs between the
directories. Based on this, go on to delete/change the images/image
names.

<!-- Bild noch einfügen -->

Independently of what happened before, the “Mask Inspection GUI” will open
and the previous windows will be closed.

<img src="\md_graphics\inspecting_masks\inspection_gui.png">

- You can now follow the instruction displayed in the GUI.
- The labels will be projected on the image in an opague green.
- Be aware the the Delete button will permanently delete the image/mask
pair in the respective folders. Making copies of the folders priorly might be
advantageous, in case you want to keep the images/masks for corrections.