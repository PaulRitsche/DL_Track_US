"""
Module docstring explaining how to change the GUI settings.

The settings can be related to both the GUI appearence and the analyses
functions. Parameters for the analyses functions are clustered based on the
openhdemg library's modules (here described as #----- MODULE NAME -----) and
can be better known from the API section of the openhdemg website.

Each parameter for the analyses functions is named FunctionName__Parameter.
An extensive explanation of each "Parameter" can be found in the specific
API module and in the specific "FunctionName".

A tutorial on how to use this settings file is available at:
https://www.giacomovalli.com/openhdemg/gui_settings/

If you mess up with the settings, you can restore the default ones from
backup_settings.py
"""

"""Instance method to open new window for analysis parameter input.
The window is opened upon pressing of the "analysis parameters"
button.

Several parameters are displayed.
- Apo Threshold:
The user must input the aponeurosis threshold
either by selecting from the dropdown list or
entering a value. By varying this threshold, different
structures will be classified as aponeurosis as the threshold for
classifying a pixel as aponeurosis is changed. Float, be non-zero and
non-negative.
- Apo length threshold:
The user must input the threshold applied to predicted
aponeurosis length in pixels. By varying this
threshold, different structures will be classified as
aponeurosis depending on their length. Must be non-zero and
non-negative.
- Fasc Threshold:
The user must input the fascicle threshold
either by selecting from the dropdown list or
entering a value. By varying this threshold, different
structures will be classified as fascicle as the threshold for
classifying a pixel as fascicle is changed.
Float, must be non-negative and non-zero.
- Fasc Cont Threshold:
The user must input the fascicle contour threshold
either by selecting from the dropdown list or
entering a value. By varying this threshold, different
structures will be classified as fascicle. By increasing, longer
fascicle segments will be considered, by lowering shorter segments.
Integer, must be non-zero and non-negative.
- Minimal Width:
The user must input the minimal with either by selecting from the
dropdown list or entering a value. The aponeuroses must be at least
this distance apart to be detected. The distance is specified in
pixels. Integer, must be non-zero and non-negative.
- Min Pennation:
The user must enter the minimal pennation angle physiologically apt
occuring in the analyzed image/video. Fascicles with lower pennation
angles will be excluded. The pennation angle is calculated as the angle
of insertion between extrapolated fascicle and detected aponeurosis.
Integer, must be non-negative.
- Max Pennation:
The user must enter the minimal pennation angle physiologically apt
occuring in the analyzed image/video. Fascicles with lower pennation
angles will be excluded. The pennation angle is calculated as the angle
of insertion between extrapolated fascicle and detected aponeurosis.
Integer, must be non-negative.

The parameters are set upon pressing the "set parameters" button.
"""

aponeurosis_detection_threshold = 0.2
aponeurosis_length_threshold = 600
fascicle_detection_threshold = 0.05 #0.05
fascicle_length_threshold = 40
minimal_muscle_width = 60
minimal_pennation_angle = 1
maximal_pennation_angle = 100
