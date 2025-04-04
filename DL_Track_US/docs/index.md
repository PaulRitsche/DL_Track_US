# Welcome to DL_Track_US

**DL_Track_US** is a Python toolkit for the automated and manual analysis of muscle architecture in human lower limb ultrasonography images and videos. It provides a user-friendly graphical interface and supports fully automated batch processing of both images and videos.

The method uses deep learning models trained on longitudinal ultrasound images of the **gastrocnemius medialis**, **tibialis anterior**, **soleus**, and **vastus lateralis** muscles. It extracts key architectural parameters:

✅ Fascicle length  
✅ Pennation angle  
✅ Muscle thickness  

Originally introduced by Cronin, Seynnes, and Finni in 2020, the method was substantially expanded in 2022 by Paul Ritsche and colleagues. The full method was published in 2023 and 2024:  

- 📄 [Journal of Open Source Software, 2023](https://doi.org/10.21105/joss.05206)
- 📄 [Ultrasound in Medicine & Biology, 2024](https://doi.org/10.1016/j.ultrasmedbio.2024.01.004)


---

## Why use it?

- 🔍 **Objective analysis**  
  The automated pipeline minimizes user influence by removing the need for manual input during processing.

- 🚅 **Fast performance**  
  Each image or frame is processed in under one second—much faster than manual analysis.

- 💾 **Efficient batch processing**  
  Analyze full folders of images or videos in a single run.

- 👓 **Graphical interface**  
  No command-line experience needed. Use the GUI to configure settings and monitor progress.

---

## Before You Start

- Test the pretrained models on your own data. Retraining may be necessary for other muscles or devices.
- Image quality is critical. Ensure good contrast, brightness, and visibility of fascicles and aponeuroses.
- Generalization may be limited. Device types and acquisition settings can affect results.
- Poor predictions? Visually inspect model output and compare it to manual labels. Adjust parameters or train new models if needed.
- Use the test scripts in the `DL_Track_US/tests` folder to confirm everything runs as expected on your system.

---

## Known Limitations

- No unit tests are currently included.
- Pretrained models are limited to lower limb muscles.
- Only four ultrasound devices were represented in training.
- Long video files (e.g., >2000 frames) may still take several minutes to process.
- The training data was manually annotated, so some subjectivity remains despite automation.

---

## Contributing

DL_Track_US is open source and community-driven. If you encounter issues, have suggestions, or want to contribute improvements, visit the project on [GitHub](https://github.com/PaulRitsche/DL_Track_US).
