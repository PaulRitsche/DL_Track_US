This page explains how to test the model training using the GUI.

- It is advantageous for model training testing to have a working GPU setup,
otherwise model training takes much longer.
- How to setup your GUI for DL_Track_US is described in the installation
guidelines of our Github repository.
- The test training images and masks you must use for this test are located at
“DL_Track_US_example/tests/model_training” folder.

For this test make sure that the files used and parameters specified are exactly
as demonstrated below.

- Click on Advanced Methods and select “Train Model” in the dropdown-menu.
- Since you will only make use of the “Model Training window” you can
disregard the main GUI.
- Make sure to use the right training <span style="color: #2eaf66;">**images**</span>
(“DL_Track_US_example/tests/model_training/apo_img_example”)
- Make sure to use the right training <span style="color: #a34ba1;">**masks**</span>
(“DL_Track_US_example/tests/model_training/apo_mask_example”).
- Keep the pre-specified <span style="color: #299ed9;">**parameter settings**</span>
as they are shown below.
- Especially make sure that the number of <span style="color: #f97e25;">**Epochs is 3**</span> (otherwise training for
test purposes takes to long).
- Click the Start Training button to start the training process.

<img src="\md_graphics\model_training_test\test_setup.png" width="600">

Several messageboxes will appear during the training process. Always click
“OK”. The messageboxes simply tell you that the images and masks have
successfully been loaded, the model was successfully compiled and that the
analysis was successfully completed.

When the analysis is complete, three new files were created in the specified
output folder.

- The **Test_apo.xlsx** file
- The **Test_apo.h5** file
- The **Training_results.tif** file

Since each training process results in slightly different models, we cannot
directly compare your results to ours. However, if the three files were
created in the “DL_Track_US_example/tests/model_training” folder, the
DL_Track_US package works properly for model training!