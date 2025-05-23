# 🧪 Testing Automated Image Analysis

This page explains how to test the **automated image analysis** in DL_Track_US.

---

## Test Preparation

Before starting:

- **Images** are evaluated **without user input**.
- Only **single images** (not videos) are required.
- Test data is located at:  
  `DL_Track_US_example/tests/test_images_automatic`

Make sure the following are correct:

- Use the correct <span style="color: #2eaf66;">**images**</span>:  
  `DL_Track_US_example/tests/test_images_automated`
- Use the provided pre-trained <span style="color: #f97e25;">**models**</span>:  
  `DL_Track_US_example/DLTrack_models`
- Keep all parameter settings in `settings.py` (accessible via the <span style="color: #e61d25;">**Settings Wheel**</span>) as they are.
- **In v0.2.1**, select <span style="color: #a34ba1;">**NO**</span> for the **Filter Fascicles** option.
- Use the correct <span style="color: #299ed9;">**flip_flag.txt**</span> file:  
  `DL_Track_US_example/tests/test_images_automated/flip_flags.txt`
- Then click the **Run** button to start the analysis.

<img src="\md_graphics\aia_test\test_setup.png">

---

## After Running the Test

After running the analysis, two new files will be created:

- <span style="color: #f97e25;">**ResultImages.pdf**</span>  
- <span style="color: #2eaf66;">**Results.xlsx**</span>  

Both will appear in:  
`DL_Track_US_example/tests/test_images_automated`

---

## Validating the Results

- Open the <span style="color: #2eaf66;">**Results.xlsx**</span> file.
- Compare the analysis results to the expected outputs shown below.

<img src="\md_graphics\aia_test\results.png">

---

✅ If the results are similar, **DL_Track_US works properly** for automated image analysis!