This page explains how to test the automated image analysis.

- The images are evaluated without user input.
- For this test, single images (not videos) are a prerequisite.
- The test images and the flip_flag.txt file you must use for this test are
located in the “DL_Track_US_example/tests/test_images_automatic” folder.

For this test make sure that the files used and parameters specified are exactly
as demonstrated below.

- Make sure to use the right <span style="color: #2eaf66;">**images**</span>
(“DL_Track_US_example/tests/test_images_automated”).
- Make sure to use the provided pre-trained <span style="color: #f97e25;">**models**</span>
(“DL_Track_US_example/DLTrack_models”).
- Keep the pre-specified parameter settings in the settings.py file (accessible through the <span style="color: #e61d25;">**Settings Wheel**</span>) as they are.
- **In v0.2.1 of the GUI, select “NO” in the “Filter Fascicles” option.**
- Make sure to use the right <span style="color: #299ed9;">**flip_flag.txt**</span> file
(“DL_Track_US_example/tests/test_images_automated/flip_flags.txt”).
- Click the Run button to start the analysis.

<img src="\md_graphics\aia_test\test_setup.png">

Once the analysis is complete, two new files were created in the
“DL_Track_US_example/tests/test_images_automated” folder.

- The **ResultImages.pdf** file
- The **Results.xlsx** file

Open the Results.xlsx file and compare the analysis results to the ones
demonstrated below. If the results are similar, the DL_Track_US package works properly for
automated images analysis!

<img src="\md_graphics\aia_test\results.png">