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
fascicle_detection_threshold = 0.05
fascicle_length_threshold = 40
minimal_muscle_width = 60
minimal_pennation_angle = 10
maximal_pennation_angle = 40


# --------------------------------- openfiles ---------------------------------

# In emg_from_demuse()
emg_from_demuse__ignore_negative_ipts = False

# In emg_from_otb()
emg_from_otb__ext_factor = 8
emg_from_otb__refsig = [True, "fullsampled"]
emg_from_otb__extras = None
emg_from_otb__ignore_negative_ipts = False

# In refsig_from_otb()
refsig_from_otb__refsig = "fullsampled"
refsig_from_otb__extras = None

# In emg_from_delsys()
emg_from_delsys__emg_sensor_name = "Galileo sensor"
emg_from_delsys__refsig_sensor_name = "Trigno Load Cell"
emg_from_delsys__filename_from = "mus_directory"

# In refsig_from_delsys()
refsig_from_delsys__refsig_sensor_name = "Trigno Load Cell"

# In emg_from_customcsv()
emg_from_customcsv__ref_signal = "REF_SIGNAL"
emg_from_customcsv__raw_signal = "RAW_SIGNAL"
emg_from_customcsv__ipts = "IPTS"
emg_from_customcsv__mupulses = "MUPULSES"
emg_from_customcsv__binary_mus_firing = "BINARY_MUS_FIRING"
emg_from_customcsv__accuracy = "ACCURACY"
emg_from_customcsv__extras = "EXTRAS"
emg_from_customcsv__fsamp = 2048
emg_from_customcsv__ied = 8

# In refsig_from_customcsv()
refsig_from_customcsv__ref_signal = "REF_SIGNAL"
refsig_from_customcsv__extras = "EXTRAS"
refsig_from_customcsv__fsamp = 2048

# In save_json_emgfile()
save_json_emgfile__compresslevel = 4


# ---------------------------------- analysis ---------------------------------

# In compute_thresholds()
compute_thresholds__n_firings = 1

# In compute_dr()
compute_dr__idr_range = None

# In basic_mus_properties()
basic_mus_properties__n_firings_rt_dert = 1
basic_mus_properties__accuracy = "default"
basic_mus_properties__ignore_negative_ipts = False
basic_mus_properties__constrain_pulses = [True, 3]
basic_mus_properties__idr_range = None


# ----------------------------------- tools -----------------------------------

# In resize_emgfile()
resize_emgfile__how = "ref_signal"
resize_emgfile__accuracy = "recalculate"
resize_emgfile__ignore_negative_ipts = False

# In compute_svr()
compute_svr__gammain = 1 / 1.6
compute_svr__regparam = 1 / 0.370
compute_svr__endpointweights_numpulses = 5
compute_svr__endpointweights_magnitude = 5
compute_svr__discontfiring_dur = 1.0


# ------------------------------------ pic ------------------------------------
# In compute_deltaf()
compute_deltaf__recruitment_difference_cutoff = 1.0
compute_deltaf__corr_cutoff = 0.7
compute_deltaf__controlunitmodulation_cutoff = 0.5


# ------------------------------------ muap -----------------------------------
# In tracking()
tracking__firings = "all"
tracking__derivation = "sd"
tracking__gui = True
tracking__gui_addrefsig = True
tracking__gui_csv_separator = "\t"

# In remove_duplicates_between()
remove_duplicates_between__firings = "all"
remove_duplicates_between__derivation = "sd"

# In MUcv_gui()
MUcv_gui__n_firings = [0, 50]
MUcv_gui__muaps_timewindow = 50
MUcv_gui__figsize = [25, 20]
MUcv_gui__csv_separator = "\t"


# --------------------------------- electrodes --------------------------------
# This custom sorting order is valid for all the GUI windows, although the
# documentation is accessible in the api of the electrodes module.
custom_sorting_order = None
