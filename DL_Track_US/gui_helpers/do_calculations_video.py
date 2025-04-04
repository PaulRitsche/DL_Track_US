"""
Description
-----------
This module contains functions to caculate muscle architectural
parameters based on binary segmentations by convolutional neural networks.
The parameters include muscle thickness, pennation angle and fascicle length.
First, input images are segmented by the CNNs. Then the predicted aponeuroses
and fascicle fragments are thresholded and filtered. Fascicle fragments
and aponeuroses are extrapolated and the intersections determined.
This module is specifically designed for video analysis and is predisposed
for execution from a tk.TK GUI instance.
The architectural parameters are calculated. The results are plotted and
converted to an output video displaying the segmentations. Each frame
is evaluated separately, independently from the previous frames.

Functions scope
---------------
doCalculations
    Function to compute muscle architectural parameters based on
    convolutional neural netwrok segmentation.

Notes
-----
Additional information and usage examples can be found at the respective
functions documentations. See specifically do_calculations.py.

See Also
--------
do_calculations.py
"""

from __future__ import division

import math
import tkinter as tk
from sys import platform
import time

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from skimage.morphology import skeletonize
from skimage.transform import resize
from tensorflow.keras.utils import img_to_array
import tensorflow as tf

from DL_Track_US.gui_helpers.do_calculations import (
    contourEdge,
    sortContours,
    filter_fascicles,
)


def optimize_fascicle_loop(
    contoursF3,
    new_Y_UA,
    new_Y_LA,
    new_X_UA,
    new_X_LA,
    width,
    min_pennation,
    max_pennation,
    filter_fascicles_func,
    fasc_cont_thresh,
    calib_dist,
):
    """
    Extracts, extrapolates, and filters fascicle contours based on angle and length criteria.

    This function processes a list of fascicle contour candidates, fits a linear function
    to each fascicle, extrapolates it, and computes intersections with extrapolated aponeuroses.
    Fascicles that meet a specified range of pennation angles and are long enough are retained.
    Optionally, filtering of overlapping fascicles can be applied. The result is returned
    as a structured `pandas.DataFrame`.

    Parameters
    ----------
    contoursF3 : list of np.ndarray
        List of fascicle contours (as arrays of shape (N, 1, 2)) returned by OpenCV `findContours`.
    new_Y_UA : np.ndarray
        Y-values of the extrapolated upper aponeurosis curve (from polynomial fit).
    new_Y_LA : np.ndarray
        Y-values of the extrapolated lower aponeurosis curve (from polynomial fit).
    new_X_UA : np.ndarray
        X-values corresponding to `new_Y_UA`, used to estimate the aponeurosis slope.
    new_X_LA : np.ndarray
        X-values corresponding to `new_Y_LA`, used to estimate the aponeurosis slope.
    width : int
        Image width, used to define extrapolation range.
    min_pennation : float
        Minimum allowable pennation angle for valid fascicle inclusion (in degrees).
    max_pennation : float
        Maximum allowable pennation angle for valid fascicle inclusion (in degrees).
    filter_fascicles_func : callable or None
        Optional function to apply filtering to remove overlapping or invalid fascicles.
        Should accept and return a `pandas.DataFrame`.
    fasc_cont_thresh : int
        Minimum number of contour points to consider a fascicle candidate.
    calib_dist : float or int
        Calibration distance in pixels between two 10 mm markers. If given,
        fascicle lengths are scaled to mm.

    Returns
    -------
    fascicle_data : pandas.DataFrame
        A DataFrame containing columns:
        - 'x_low': int, start x-coordinate
        - 'x_high': int, end x-coordinate
        - 'y_low': int, start y-coordinate
        - 'y_high': int, end y-coordinate
        - 'coordsX': np.ndarray, x-points along fascicle
        - 'coordsY': np.ndarray, y-points along fascicle
        - 'fasc_l': float, fascicle length (in pixels or mm)
        - 'penn_a': float, fascicle pennation angle (degrees)

        If no valid fascicles are found, the DataFrame will be empty with correct column names.

    Notes
    -----
    - Angles are computed relative to the slope of the lower aponeurosis.
    - Fascicle extrapolation uses a 1st-degree polynomial fit.
    - Fascicles extending beyond the extrapolation range are ignored.

    Examples
    --------
    >>> df = optimize_fascicle_loop(
    ...     contoursF3=contours,
    ...     new_Y_UA=upper_y,
    ...     new_Y_LA=lower_y,
    ...     new_X_UA=upper_x,
    ...     new_X_LA=lower_x,
    ...     width=512,
    ...     min_pennation=10,
    ...     max_pennation=40,
    ...     filter_fascicles_func=my_filter_function,
    ...     fasc_cont_thresh=40,
    ...     calib_dist=98
    ... )
    >>> df.head()
       x_low  x_high  y_low  y_high   fasc_l   penn_a
    0    125     212    220     280  43.2175  21.3247
    """

    data_rows = []
    newX = np.linspace(-400, width + 400, 5000)

    for cnt in contoursF3:
        if len(cnt) <= fasc_cont_thresh:
            continue

        x, y = contourEdge("B", cnt)
        z = np.polyfit(np.array(x), np.array(y), 1)
        f = np.poly1d(z)
        newY = f(newX)

        diffU = np.abs(newY - new_Y_UA)
        diffL = np.abs(newY - new_Y_LA)

        locU = np.argmin(diffU)
        locL = np.argmin(diffL)

        if locL >= 4950:
            continue

        coordsX = newX[locL:locU]
        coordsY = newY[locL:locU]

        try:
            angle_numer = new_Y_LA[locL] - new_Y_LA[locL + 50]
            angle_denom = new_X_LA[locL + 50] - new_X_LA[locL]
            Apoangle = 90 + abs(np.arctan(angle_numer / angle_denom) * 180 / np.pi)
        except Exception:
            continue

        if len(coordsX) > 0 and not np.isnan(Apoangle):
            try:
                FascAng = (
                    float(
                        np.arctan(
                            (coordsX[0] - coordsX[-1])
                            / (new_Y_LA[locL] - new_Y_UA[locU])
                        )
                        * 180
                        / np.pi
                    )
                    * -1
                )
                ActualAng = Apoangle - FascAng
            except Exception:
                continue

            if min_pennation <= ActualAng <= max_pennation:
                length1 = np.sqrt(
                    (newX[locU] - newX[locL]) ** 2
                    + (new_Y_UA[locU] - new_Y_LA[locL]) ** 2
                )

                data_rows.append(
                    {
                        "x_low": int(coordsX[0]),
                        "x_high": int(coordsX[-1]),
                        "y_low": int(coordsY[0]),
                        "y_high": int(coordsY[-1]),
                        "coordsX": coordsX,
                        "coordsY": coordsY,
                        "fasc_l": length1,
                        "penn_a": Apoangle - FascAng,
                    }
                )

    if not data_rows:
        fascicle_data = pd.DataFrame(
            columns=[
                "x_low",
                "x_high",
                "y_low",
                "y_high",
                "coordsX",
                "coordsY",
                "fasc_l",
                "penn_a",
            ]
        )
    else:
        fascicle_data = pd.DataFrame(data_rows)

    if filter_fascicles_func:
        fascicle_data = filter_fascicles_func(fascicle_data)

    if calib_dist and not fascicle_data.empty:
        fascicle_data["fasc_l"] = fascicle_data["fasc_l"] / (calib_dist / 10)

    return fascicle_data


def doCalculationsVideo(
    vid_len: int,
    cap,
    vid_out,
    flip: str,
    apo_model,
    fasc_model,
    calib_dist: int,
    dic: dict,
    step: int,
    filter_fasc: bool,
    gui,
    segmentation_mode: str,
    frame_callback=None,
):
    """
    Function to compute muscle architectural parameters based on
    convolutional neural network segmentation in videos.

    Firstly, images are segmented by the network. Then, predictions
    are thresholded and filtered. The aponeuroses edges are computed and
    the fascicle length and pennation angle calculated. This is done
    by extrapolating fascicle segments above a threshold length. Then
    the intersection between aponeurosis edge and fascicle structures are
    computed.
    Returns none when not more than one aponeurosis contour is
    detected in the image.

    Parameters
    ----------
    vid_len : int
        Integer variable containing the number of frames present
        in cap.
    cap : cv2.VideoCapture
        Object that contains the video in a np.ndarrray format.
        In this way, seperate frames can be accessed.
    vid_out : cv2.VideoWriter
        Object that is stored in the vpath folder.
        Contains the analyzed video frames and is titled "..._proc.avi"
        The name can be changed but must be different than the input
        video.
    flip : {"no_flip", "flip"}
        String variable defining wheter an image should be flipped.
        This can be "no_flip" (video is not flipped) or "flip"
        (video is flipped).
    apo_model
        Aponeurosis neural network.
    fasc_model
        Fascicle neural network.
    calib_dist : int
        Integer variable containing the distance between the two
        specified point in pixel units. The points must be 10mm
        apart. Must be non-negative. If "None", the values are outputted in
        pixel units.
    dic : dict
        Dictionary variable containing analysis parameters.
        These include must include apo_threshold, fasc_threshold,
        fasc_cont_threshold, min_width, max_pennation,
        min_pennation.
    step : int
        Integer variable containing the step for the range of video frames.
        If step != 1, frames are skipped according to the size of step.
        This might decrease processing time but also accuracy.
    filter_fasc : bool
        If True, fascicles will be filtered so that no crossings are included.
        This may reduce number of totally detected fascicles.
    gui : tk.TK
        A tkinter.TK class instance that represents a GUI. By passing this
        argument, interaction with the GUI is possible i.e., stopping
        the calculation process after each image.
    segmentation_mode : str
        String variable containing the segmentation mode. This is used to
        determine the segmentation model used. Choose between "stacked" and
        and "None". When "stacked" is chosen, the frames are loaded in stacks of
        three.
    display_frame : bool
        Boolean variable determining whether the current frame is displayed in main
        UI.

    Returns
    -------
    fasc_l_all : list
        List of arrays contianing the estimated fascicle lengths
        based on the segmented fascicle fragments in pixel units
        as float. If calib_dist is specified, then the length is computed
        in centimeter. This is computed for each frame in the video.
    pennation_all : list
        List of lists containing the estimated pennation angles
        based on the segmented fascicle fragments and aponeuroses
        as float. This is computed for each frame in the video.
    x_lows_all : list
        List of lists containing the estimated x-coordinates
        of the lower edge from the upper aponeurosis as integers.
        This is computed for each frame in the video.
    x_highs_all : list
        List of lists containing the estimated x-coordinates
        of the upper edge from the lower aponeurosis as integers.
        This is computed for each frame in the video.
    midthick_all : list
        List variable containing the estimated distance
        between the lower and upper aponeurosis in pixel units.
        If calib_dist is specified, then the distance is computed
        in centimeter.
        This is computed for each frame in the video.


    Examples
    --------
    >>> doCalculations(vid_len=933, cap=< cv2.VideoCapture 000002BFAD0560B0>,
                        vid_out=< cv2.VideoWriter 000002BFACEC0130>,
                        flip="no_flip",
                        apo_modelpath="C:/Users/admin/Documents/DL_Track/Models_DL_Track/Final_models/model-VGG16-fasc-BCE-512.h5",
                        fasc_modelpath="C:/Users/admin/Documents/DL_Track/Models_DL_Track/Final_models/model-apo-VGG-BCE-512.h5",
                        calib_dist=98,
                        dic={'apo_treshold': '0.2', 'fasc_threshold': '0.05',
                        'fasc_cont_thresh': '40', 'min_width': '60',
                        'min_pennation': '10', 'max_pennation': '40'},
                        filter_fasc = False,
                        gui=<__main__.DL_Track_US object at 0x000002BFA7528190>,
                        segmentation_mode=None,
                        display_frame=True)
    [array([60.5451731 , 58.86892027, 64.16011534, 55.46192704, 63.40711356]), ..., array([64.90849385, 60.31621836])]
    [[19.124207107383114, 19.409753216521565, 18.05706763600641, 20.54453899050867, 17.808652286488794], ..., [17.26241882195032, 16.284803480359543]]
    [[148, 5, 111, 28, -164], [356, 15, 105, -296], [357, 44, -254], [182, 41, -233], [40, 167, 42, -170], [369, 145, 57, -139], [376, 431, 32], [350, 0]]
    [[725, 568, 725, 556, 444], [926, 572, 516, 508], [971, 565, 502], [739, 578, 474], [554, 766, 603, 475], [1049, 755, 567, 430], [954, 934, 568], [968, 574]]
    [23.484416057267826, 22.465452189555794, 21.646971767045816, 21.602856412413924, 21.501286239714894, 21.331137350026623, 21.02446763240188, 21.250352548097883]
    """

    try:

        # Extract dictionary parameters
        fasc_cont_thresh = int(dic["fascicle_length_threshold"])
        min_width = int(dic["minimal_muscle_width"])
        max_pennation = int(dic["maximal_pennation_angle"])
        min_pennation = int(dic["minimal_pennation_angle"])
        apo_threshold = float(dic["aponeurosis_detection_threshold"])
        apo_length_thresh = float(dic["aponeurosis_length_threshold"])
        fasc_threshold = float(dic["fascicle_detection_threshold"])

        # Define empty lists for parameter storing
        fasc_l_all, pennation_all, x_lows_all, x_highs_all, thickness_all = (
            [] for _ in range(5)
        )

        height, width = 512, 512
        frames_single = []
        frames_stacked = []

        for i in range(vid_len):
            ret, frame = cap.read()

            if not ret:
                break

            if flip == "flip":
                frame = cv2.flip(frame, 1)

            # single frame part
            img = img_to_array(frame)
            img = resize(img, (512, 512, 3))
            img_normalized = img / 255.0
            img_input = np.expand_dims(img_normalized, axis=0)
            frames_single.append(img_input)

            # stacked frames for IFSS
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.resize(gray_frame, (512, 512))
            gray_frame = gray_frame.astype(np.float32) / 255.0
            gray_frame = np.expand_dims(gray_frame, axis=-1)
            frames_stacked.append(gray_frame)

        print("Total frames read:", len(frames_stacked))

        # Loop through each frame of the video
        for a in range(0, len(frames_single) - 2, step):
            if gui.should_stop:
                break  # there was an input to stop the calculations

            # time the video frame processing
            start_time = time.time()

            # Get original image for plotting
            img_orig = (frames_single[a][0] * 255).astype(np.uint8)

            # Predict aponeurosis and fascicle segments
            pred_apo = apo_model.predict(frames_single[a])
            pred_apo_t = (pred_apo > apo_threshold).astype(np.uint8)
            pred_apo_t = resize(pred_apo_t[0], (512, 512))

            # Get image for fascicle prediction
            if segmentation_mode == "stacked":
                if a + 2 >= len(frames_stacked):
                    break  # not enough frames to make a stack
                stacked = np.stack(
                    [frames_stacked[a], frames_stacked[a + 1], frames_stacked[a + 2]],
                    axis=0,
                )
                fasc_input = np.expand_dims(stacked, axis=0)  # (1, 3, 512, 512, 1)
                pred_fasc = fasc_model([fasc_input], training=False)
                pred_fasc = tf.clip_by_value(pred_fasc, 0, 1).numpy()

            else:
                fasc_input = frames_single[a]
                pred_fasc = fasc_model.predict(fasc_input)

            if segmentation_mode == "stacked":
                pred_fasc = np.array(pred_fasc[:, 1, :, :, 0])  # take only middle frame
                pred_fasc = np.expand_dims(pred_fasc, axis=-1)

            # Threshold predictions
            pred_fasc_t = (pred_fasc > fasc_threshold).astype(np.uint8)
            pred_fasc_t = resize(pred_fasc_t[0], (512, 512))

            # Compute the contours to identify aponeuroses
            _, thresh = cv2.threshold(pred_apo_t, 0, 255, cv2.THRESH_BINARY)
            thresh = thresh.astype("uint8")
            # Find contours in thresholded image
            contours, hierarchy = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )

            # include contours of long length
            contours = [i for i in contours if len(i) > apo_length_thresh]
            # Sort contours from top to bottom
            contours, _ = sortContours(contours)

            # Sort contours based on x-values
            contours_re2 = []
            for contour in contours:
                pts_arr = np.squeeze(np.array(contour))
                pts_arr = pts_arr[np.lexsort((pts_arr[:, 1], pts_arr[:, 0]))]
                app = pts_arr
                contours_re2.append(app)

            # Define variables for contour merging
            xs1 = []
            xs2 = []
            ys1 = []
            ys2 = []
            maskT = np.zeros(thresh.shape, np.uint8)

            # Append coordinates
            for cnt in contours_re2:
                ys1.append(cnt[0][1])
                ys2.append(cnt[-1][1])
                xs1.append(cnt[0][0])
                xs2.append(cnt[-1][0])
                cv2.drawContours(maskT, [cnt], 0, 255, -1)

            # Merge nearby contours
            for countU in range(0, len(contours_re2) - 1):
                if (
                    xs1[countU + 1] > xs2[countU]
                ):  # Check if x of contour2 is higher than x of contour 1
                    y1 = ys2[countU]
                    y2 = ys1[countU + 1]
                    if y1 - 10 <= y2 <= y1 + 10:
                        m = np.vstack((contours_re2[countU], contours_re2[countU + 1]))
                        cv2.drawContours(maskT, [m], 0, 255, -1)
                countU += 1

            # Make binary
            maskT[maskT > 0] = 1
            # Skeletonize o detect edges
            skeleton = skeletonize(maskT).astype(np.uint8)
            kernel = np.ones((3, 7), np.uint8)
            # Dilate and erode contours to detect edges
            dilate = cv2.dilate(skeleton, kernel, iterations=15)
            erode = cv2.erode(dilate, kernel, iterations=10)

            # Find contour edges
            contoursE, hierarchy = cv2.findContours(
                erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )

            # Create new mask later used for plotting
            mask_apoE = np.zeros(thresh.shape, np.uint8)

            # Select only long contours and draw to mask
            contoursE = [i for i in contoursE if len(i) > apo_length_thresh]
            for contour in contoursE:
                cv2.drawContours(mask_apoE, [contour], 0, 255, -1)

            # Sort contours again from top to bottom
            contoursE, _ = sortContours(contoursE)

            # Continue only when 2 or more aponeuroses were detected
            if len(contoursE) >= 2:

                # Get the x,y coordinates of the upper/lower edge of the 2
                # aponeuroses
                upp_x, upp_y = contourEdge("B", contoursE[0])
                if contoursE[1][0, 0, 1] > (contoursE[0][0, 0, 1] + min_width):
                    low_x, low_y = contourEdge("T", contoursE[1])
                else:
                    low_x, low_y = contourEdge("T", contoursE[2])

                # Filter data one-dimensionally to extend the data
                upp_y_new = savgol_filter(upp_y, 81, 2)
                low_y_new = savgol_filter(low_y, 81, 2)

                # Make a binary mask
                ex_mask = np.zeros(thresh.shape, np.uint8)

                x_common = sorted(list(set(upp_x).intersection(set(low_x))))
                for x in x_common:
                    idx_u = np.where(upp_x == x)[0][0]
                    idx_l = np.where(low_x == x)[0][0]

                    ymin = int(np.floor(upp_y_new[idx_u]))
                    ymax = int(np.ceil(low_y_new[idx_l]))

                    if 0 <= x < ex_mask.shape[1]:
                        ex_mask[ymin:ymax, x] = 255

                # Calculate slope of central portion of each aponeurosis
                # & use this to compute muscle thickness
                Alist = list(set(upp_x).intersection(low_x))
                Alist = sorted(Alist)
                Alen = len(
                    list(set(upp_x).intersection(low_x))
                )  # How many values overlap between x-axes
                A1 = int(Alist[0] + (0.33 * Alen))
                A2 = int(Alist[0] + (0.66 * Alen))
                mid = int((A2 - A1) / 2 + A1)
                mindist = 10000
                upp_ind = np.where(upp_x == mid)

                if upp_ind == len(upp_x):
                    upp_ind -= 1

                for val in range(A1, A2):
                    if val >= len(low_x):
                        continue
                    else:
                        dist = math.dist(
                            (upp_x[upp_ind], upp_y_new[upp_ind]),
                            (low_x[val], low_y_new[val]),
                        )
                        if dist < mindist:
                            mindist = dist

                # Add aponeuroses to a mask for display
                imgT = np.zeros((height, width, 3), np.uint8)

                # Compute functions to approximate the shape of the aponeuroses
                zUA = np.polyfit(upp_x, upp_y_new, 1)  # 1st order polynomial
                g = np.poly1d(zUA)
                zLA = np.polyfit(low_x, low_y_new, 1)
                h = np.poly1d(zLA)

                mid = (low_x[-1] - low_x[0]) / 2 + low_x[
                    0
                ]  # Find middle of the aponeurosis
                x1 = np.linspace(
                    low_x[0] - 700, low_x[-1] + 700, 10000
                )  # Extrapolate polynomial fits to either side
                y_UA = g(x1)
                y_LA = h(x1)

                new_X_UA = np.linspace(
                    mid - 700, mid + 700, 5000
                )  # Extrapolate x,y data using f function
                new_Y_UA = g(new_X_UA)
                new_X_LA = np.linspace(
                    mid - 700, mid + 700, 5000
                )  # Extrapolate x,y data using f function
                new_Y_LA = h(new_X_LA)

                # Fascicle calculation part
                # Compute contours to identify fascicles / fascicle orientation
                _, threshF = cv2.threshold(pred_fasc_t, 0, 255, cv2.THRESH_BINARY)
                threshF = threshF.astype("uint8")
                contoursF, hierarchy = cv2.findContours(
                    threshF, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                # Remove any contours that are very small
                maskF = np.zeros(threshF.shape, np.uint8)
                for contour in contoursF:
                    if len(contour) > fasc_cont_thresh:
                        cv2.drawContours(maskF, [contour], 0, 255, -1)

                # Only include fascicles within the region of the 2 aponeuroses
                # mask_Fi = maskF & ex_mask
                contoursF3, hierarchy = cv2.findContours(  # contoursF2
                    maskF, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )

                # Define lists to store analysis parameters
                fasc_l = []
                pennation = []
                x_low1 = []
                x_high1 = []

                fascicle_data = optimize_fascicle_loop(
                    contoursF3,
                    new_Y_UA,
                    new_Y_LA,
                    new_X_UA,
                    new_X_LA,
                    width,
                    min_pennation,
                    max_pennation,
                    filter_fascicles if filter_fasc == 1 else None,
                    fasc_cont_thresh,
                    calib_dist,
                )

                # Generate color map once
                num_colors = len(fascicle_data)
                if num_colors > 0:
                    gradient = np.linspace(0, 255, num_colors, dtype=np.uint8).reshape(
                        -1, 1
                    )
                    color_map = cv2.applyColorMap(gradient, cv2.COLORMAP_RAINBOW)
                    fascicle_coords = fascicle_data[["coordsX", "coordsY"]].values

                    for i in range(num_colors):
                        color = tuple(int(c) for c in color_map[i][0])
                        x = fascicle_coords[i][0].astype(np.int32)
                        y = fascicle_coords[i][1].astype(np.int32)
                        coords = np.stack((x, y), axis=-1)
                        cv2.polylines(
                            imgT, [coords], isClosed=False, color=color, thickness=3
                        )

                # Store the results for each frame and normalise using scale
                # factor (if calibration was done above)
                try:
                    midthick = mindist[0]  # Muscle thickness
                except:
                    midthick = mindist

            # Return empty variables when no two aponeuroses were detected
            else:
                fasc_l = []
                pennation = []
                x_low1 = []
                x_high1 = []
                imgT = np.zeros((height, width, 3), np.uint8)
                fasc_l.append(float("nan"))
                pennation.append(float("nan"))
                x_low1.append(float("nan"))
                x_high1.append(float("nan"))
                midthick = float("nan")

            df = pd.DataFrame(
                {
                    "Fascicles": fascicle_data["fasc_l"],
                    "Pennation": fascicle_data["penn_a"],
                    "X_low": fascicle_data["x_low"],
                    "X_high": fascicle_data["x_high"],
                }
            )

            # Sorting the DataFrame according to X_low
            df_sorted = df.sort_values(by="X_low")

            # Append parameters to overall list
            fasc_l_all.append(df_sorted["Fascicles"].tolist())
            pennation_all.append(df_sorted["Pennation"].tolist())
            x_lows_all.append(df_sorted["X_low"].tolist())
            x_highs_all.append(df_sorted["X_high"].tolist())
            thickness_all.append(midthick)

            # Display each processed frame
            img_orig[mask_apoE > 0] = (235, 25, 42)

            comb = cv2.addWeighted(img_orig.astype(np.uint8), 1, imgT, 0.8, 0)

            vid_out.write(comb)  # Write each image to video file
            cv2.putText(
                comb,
                ("Frame: " + str(a + 1) + " of " + str(vid_len)),
                (125, 380),
                cv2.FONT_HERSHEY_DUPLEX,
                0.75,
                (249, 249, 249),
            )
            cv2.putText(
                comb,
                (
                    "Pennation angle: "
                    + str("%.1f" % np.median(pennation_all[-1]))
                    + " deg"
                ),
                (125, 440),
                cv2.FONT_HERSHEY_DUPLEX,
                0.75,
                (249, 249, 249),
            )
            if calib_dist:
                cv2.putText(
                    comb,
                    (
                        "Fascicle length: "
                        + str("%.2f" % np.median(fasc_l_all[-1]) + " mm")
                    ),
                    (125, 380),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.75,
                    (249, 249, 249),
                )
                cv2.putText(
                    comb,
                    ("Thickness at centre: " + str("%.1f" % thickness_all[-1]) + " mm"),
                    (125, 410),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.75,
                    (249, 249, 249),
                )
            else:
                cv2.putText(
                    comb,
                    (
                        "Fascicle length: "
                        + str("%.2f" % np.median(fasc_l_all[-1]) + " px")
                    ),
                    (125, 410),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.75,
                    (249, 249, 249),
                )
                cv2.putText(
                    comb,
                    ("Thickness at centre: " + str("%.1f" % thickness_all[-1]) + " px"),
                    (125, 470),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.75,
                    (249, 249, 249),
                )

            # Print the time taken for processing the frame
            print("Time taken to process frame:", time.time() - start_time)

            # Display the processed image
            # Send frame to UI
            if frame_callback:
                frame_callback(comb)

        # Release video and close all analysis windows when analysis finished
        cap.release()
        vid_out.release()
        cv2.destroyAllWindows()

        return fasc_l_all, pennation_all, x_lows_all, x_highs_all, thickness_all

    # Check if model path is correct
    except OSError:
        tk.messagebox.showerror("Information", "Apo/Fasc model path is incorrect.")
        gui.should_stop = False
        gui.is_running = False
        gui.do_break()
        return

    finally:
        # clean up
        cap.release()
        vid_out.release()
        cv2.destroyAllWindows()

        gui.should_stop = False
        gui.is_running = False
