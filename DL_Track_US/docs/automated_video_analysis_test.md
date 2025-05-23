# 🖼 Testing Automated Video Analysis

This page explains how to test the **automated video analysis** functionality in DL_Track_US.

- Single video frames are evaluated **automatically** without user input.
- For this test, **videos** are required.
- The test video is located in:  
  `DL_Track_US_example/tests/test_video_automated`.

---

Before running the test, ensure that:

- You are using the correct <span style="color: #2eaf66;">**video**</span> located at `DL_Track_US_example/tests/test_video_automated`.
- You are using the provided pre-trained <span style="color: #f97e25;">**models**</span>, located at `DL_Track_US_example/models/` (use IFSS for fascicles).
- You have kept the default <span style="color: #e61d25;">**parameter settings**</span> in the `settings.json` file unchanged.
- You click the <span style="color: #a34ba1;">**Run**</span> button in the GUI to start the analysis.

<img src="\md_graphics/ava_test/test_setup.png">

---

When the analysis is complete, two new files will appear in the `DL_Track_US_example/tests/test_video_automated` folder:

- **calf_raise_proc.avi** (processed video with predictions)
- **calf_raise.xlsx** (results file)

---

### How to verify the results:

1. Open the `calf_raise.xlsx` file.
2. For all frames, calculate the **average** values for:
   - Fascicle length
   - Pennation angle
   - Muscle thickness

3. Compare your results to the reference results shown below:

<img src="\md_graphics/ava_test/results.png">

---

✅ If your values are similar, DL_Track_US is working correctly for **automated video analysis**!
