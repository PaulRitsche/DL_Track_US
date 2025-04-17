This page explains how to test the automated video analysis.

- Single video frames are evaluated automatically without user input.
- For this test, videos are a prerequisite.
- The test video you must use for this test is located in the
“DL_Track_US_example/tests/test_video_automated” folder.

For this test make sure that the files used and parameters specified are exactly
as demonstrated below

- Make sure to use the right <span style="color: #2eaf66;">**video**</span>
(“DL_Track_US_example/tests/test_video_automated”).
- Make sure to use the provided pre-trained <span style="color: #f97e25;">**models**</span>.
- Keep the pre-specified <span style="color: #e61d25;">**parameter settings**</span> in the settings.py file as they are.
- Click the Run button to start the analysis.

<img src="\md_graphics\ava_test\test_setup.png">

When the analysis is complete, two new files were created in the
“DL_Track_US_example/tests/test_video_automated” folder.

- The **calf_raise_proc.avi** file
- The **calf_raise.xlsx** file

Open the calf_raise.xlsx file.

- Take the average value from all calculated fascicle length values in all
frames, all calculated pennation angles in all frames, all calculated muscle
thickness values in all frames and all calculated upper (x_high) and lower
(x_low) aponeuroses edge coordinates in all frames.
- If the results are similar to those demonstrated below, the DL_Track_US
package works properly for automated images analysis!

<img src="\md_graphics\ava_test\results.png">