# 🧪 Testing Manual Image / Video Analysis

This page explains how to test the **manual image** and **manual video analysis** modes.

- For this test, **single images** (not videos) are a prerequisite.
- The test image you must use is located in the `DL_Track_US_example/tests/test_images_manual` folder.
- Both **manual image analysis** and **manual video analysis** use the **same Python class** (`ManualAnalysis` located in `manual_tracing.py`).
- Therefore, **testing one** (manual image analysis) is sufficient.

---

## Test Preparation

Before starting:

- Make sure to use the correct <span style="color: #2eaf66;">**image**</span>:  
  `DL_Track_US_example/tests/test_image_manual`.
- Click the **Run** button to start the analysis.

![test_setup](md_graphics\miva_test\test_setup.png)

---

## Running the Test

After clicking **Run**:

- The **Manual Analysis window** should pop up containing the test image.

![reanalysis](md_graphics\miva_test\reanalysing.png)

---

## Reanalysing the Test Image

Follow these steps:

1. **Scale the image**  
   - Follow the **one-centimetre long scaling line** shown on the left of the image.  
   - <span style="color: #299ed9;">**Scale the image accordingly.**</span>

2. **Draw Aponeuroses**  
   - Redraw the superficial and deep <span style="color: #2eaf66;">**aponeurosis extension lines**</span>.

3. **Measure Muscle Thickness**  
   - Redraw the <span style="color: #a34ba1;">**three vertical muscle thickness lines**</span> using **one segment each**.

4. **Trace Fascicles**  
   - Redraw the <span style="color: #f97e25;">**three diagonal fascicle lines**</span> using **three segments each**.

5. **Measure Pennation Angles**  
   - Redraw the <span style="color: #e61d25;">**three pennation angles**</span> using **two segments each**.

> ⚡ **Important:**  
> Always select the **correct Radiobutton** corresponding to the parameter you are analyzing.

---

After reanalyzing all lines:

- Click the <span style="color: #a34ba1;">**Save Results**</span> button to save your analysis.

A new file will be created:

- `Manual_Results.xlsx` inside the `DL_Track_US_example/tests/test_image_manual` folder.

---

## Validating the Results

- Open the newly created `Manual_Results.xlsx` file.
- Compare the analysis results to the expected results shown below.

If the results are similar, **DL_Track_US works properly** for manual image and video analysis!

![results](md_graphics\miva_test\results.png)

---
