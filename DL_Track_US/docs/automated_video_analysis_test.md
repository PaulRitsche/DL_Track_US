# 🧪 Testing Automated Video Analysis

This page explains how to test the **automated video analysis** functionality in DL_Track_US.

- Single video frames are evaluated **automatically** without user input.
- For this test, **videos** are required.
- The test video is located in:  
  `DL_Track_US_example/tests/test_video_automated`.

---

Before running the test, ensure that:

- You are using the correct <span style="color: #2eaf66;">**video**</span> located at `DL_Track_US_example/tests/test_video_automated`.
- You are using the provided pre-trained <span style="color: #f97e25;">**models**</span>, located at `DL_Track_US_example/models/` (use IFSS for fascicles).
- You have kept the default <span style="color: #e61d25;">**parameter settings**</span> in the `settings.json` file unchanged (i.e.: "aponeurosis_detection_threshold": 0.01, "aponeurosis_length_threshold":300, "fascicle_detection_threshold": 0.01, "fascicle_length_threshold": 40, "minimal_muscle_width": 60, "minimal_pennation_angle": 10, "maximal_pennation_angle": 30,).
- You click the <span style="color: #a34ba1;">**Run**</span> button in the GUI to start the analysis.

![setup](md_graphics/ava_test/test_setup.png)
---

When the analysis is complete, two new files will appear in the `DL_Track_US_example/tests/test_video_automated` folder:

- **calf_raise_proc.avi** (processed video with predictions)
- **calf_raise.xlsx** (results file)

---

### How to verify the results:

Open the `calf_raise.xlsx` file.
For all frames, calculate the **average** values for:

- Fasc_length_filtered_median
- Pennation_filtered_median
- Muscle thickness

Compare your results to the reference results shown below:

![results](md_graphics/ava_test/results.png)

---

✅ If your values are similar, DL_Track_US is working correctly for **automated video analysis**!
