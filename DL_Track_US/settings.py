"""
Module docstring explaining how to change the GUI settings.

The settings can be related to both the GUI appearence and the analyses
functions. Parameters for the analyses functions are clustered.

Each parameter for the analyses functions is named according to their usage.

If you mess up with the settings, you can restore the default ones from
backup_settings.py

Several parameters are displayed can be chosen from this settings file.
The default values are suggested by the authors and can be modified according
to need.


- aponeurosis_detection_threshold:
The user must input the aponeurosis threshold
either by selecting from the dropdown list or
entering a value. By varying this threshold, different
structures will be classified as aponeurosis as the threshold for
classifying a pixel as aponeurosis is changed. Float, be non-zero and
non-negative.

- aponeurosis_length_threshold:
The user must input the threshold applied to predicted
aponeurosis length in pixels. By varying this
threshold, different structures will be classified as
aponeurosis depending on their length. Must be non-zero and
non-negative.

- fascicle_detection_threshold:
The user must input the fascicle threshold
either by selecting from the dropdown list or
entering a value. By varying this threshold, different
structures will be classified as fascicle as the threshold for
classifying a pixel as fascicle is changed.
Float, must be non-negative and non-zero.

- fascicle_length_threshold:
The user must input the fascicle contour threshold
either by selecting from the dropdown list or
entering a value. By varying this threshold, different
structures will be classified as fascicle. By increasing, longer
fascicle segments will be considered, by lowering shorter segments.
Integer, must be non-zero and non-negative.

- minimal_muscle_width:
The user must input the minimal with either by selecting from the
dropdown list or entering a value. The aponeuroses must be at least
this distance apart to be detected. The distance is specified in
pixels. Integer, must be non-zero and non-negative
.
- minimal_pennation_angle:
The user must enter the minimal pennation angle physiologically apt
occuring in the analyzed image/video. Fascicles with lower pennation
angles will be excluded. The pennation angle is calculated as the angle
of insertion between extrapolated fascicle and detected aponeurosis.
Integer, must be non-negative.

- maximal_pennation_angle:
The user must enter the minimal pennation angle physiologically apt
occuring in the analyzed image/video. Fascicles with lower pennation
angles will be excluded. The pennation angle is calculated as the angle
of insertion between extrapolated fascicle and detected aponeurosis.
Integer, must be non-negative.

- fascicle_calculation_method: 
The user must enter an approach by which the fascicle length is calculated.
Can either be linear_extrapolation, curve_polyfitting, curve_connect_linear, curve_connect_poly or orientation_map.
linear_extrapolation calculates the fascicle length and pennation angle according to a linear extrapolation of the each detechted fascicle fragment.
curve_polyfitting calculates the fascicle length and pennation angle according to a second order polynomial fitting (see documentation of function curve_polyfitting).
curve_connect_linear and curve_connect_poly calculate the fascicle length and pennation angle according to a linear connection between the fascicles fascicles (see documentation of function curve_connect).
orientation_map calculates an orientation map and gives an estimate for the median angle of the image (see documentation of function orientation_map).

- fascicle_contour_tolerance:
The user must enter a positive value to specify the tolerance for fasicle fragment detection.
The lower the tolerance, the less likely it is that a contour is considered part of an extrapolated fascicle.
The value is specifing the permissible range in pixels in the positive and negative y-direction within which the next
contour can be located to still be considered a part of the extrapolated fascicle.
This is not used when the linear_extrapolation method is used.

- aponneurosis_distannce_tolerance: 
The user must enter a positive value to specify the tolerance for the distance of fascicle fragments to be away from the aponeurosis.
The lower this value, the nearer a fascicle fragment must be to the aponeurosis. This increases certainty of pennation angle calculation and extrapolation.
This is not used when the linear extrapolation method is used.


The parameters are set automatically at each run.
"""

# ------------------------------------------------------------------------------
# Prediction Parameters
aponeurosis_detection_threshold = 0.2  # 0.2
aponeurosis_length_threshold = 400  # 400
fascicle_detection_threshold = 0.05  # 0.05
fascicle_length_threshold = 40  # 40
minimal_muscle_width = 150  # 60


# ------------------------------------------------------------------------------
# Muscle Architecture Calculation Parameters
minimal_pennation_angle = 10
maximal_pennation_angle = 40
fascicle_calculation_method = "linear_extrapolation"
fascicle_contour_tolerance = 10  # 10
aponeurosis_distance_tolerance = 100
