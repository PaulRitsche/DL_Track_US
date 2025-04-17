This page explains how to test the manual image / video analysis.

- For this test, single images (not videos) are a prerequisite.
- The test image you must use for this test is located in the
“DL_Track_US_example/tests/test_images_manual” folder.
- The analysis types manual image analysis and manual video analysis make
use of the same python class (called “ManualAnalysis” and located in the
manual_tracing.py file).
- In our opinion, it is thus not necessary to test both analysis types.

For this test make sure that the files used and parameters specified are exactly
as demonstrated below.

- Make sure to use the right <span style="color: #2eaf66;">**image**</span>
(“DL_Track_US_example/tests/test_image_manual”).
- Click the Run button to start the analysis.

<img src="\md_graphics\miva_test\test_setup.png">

The “Manual Analysis window” should pop up containing the image as
demonstrated below.

<img src="\md_graphics\miva_test\reanalysing.png">

For testing the DL_Track_US manual image / video analysis, simply reanalyse
the drawn lines.

- First, <span style="color: #299ed9;">**scale the image**</span> by following the one centimetre long scaling line in the
left of the image.
- Then, redraw the superficial and deep <span style="color: #2eaf66;">**aponeurosis extension lines**</span>.
- Subsequently, re-analyse the <span style="color: #a34ba1;">**three vertical muscle thickness lines**</span> using one
segment each, the <span style="color: #f97e25;">**three diagonal fascicle lines**</span> using three segments each
and the <span style="color: #e61d25;">**three pennation angles**</span> using two segments each.
- Always choose the Radiobutton corresponding to the parameter you are
analysing.
- Once you have re-analysed all the lines image, click on the Save Results
button to save your analysis results.

One new file was created in the
“DL_Track_US_example/tests/test_image_manual”, the **Manual_Results.xlsx**
file. Open the file and compare the analysis results to the
ones demonstrated below. If the results are similar, the DL_Track_US package
works properly for manual image / video analysis!

<img src="\md_graphics\miva_test\results.png">