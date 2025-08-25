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
import traceback

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from scipy.signal import savgol_filter
from skimage.morphology import skeletonize
from skimage.transform import resize
import tensorflow as tf
from collections import deque
from DL_Track_US.gui_helpers.do_calculations import (
    contourEdge,
    sortContours,
    filter_fascicles,
)

def build_apo_from_edges(
    upp_x, upp_y, low_x, low_y, w,
    smooth_win: int = 81, smooth_poly: int = 2,
    tail_frac: float = 0.20,   # ← use 40% of each side
):
    """
    Build dense, smoothed, and extrapolated aponeurosis curves from detected edges.

    This function constructs upper and lower aponeuroses on a shared dense x-grid
    spanning [-0.5*w, 1.5*w]. The detected aponeurosis edges are smoothed with a
    Savitzky–Golay filter, interpolated within their detected domain, and
    extrapolated linearly on both left and right sides. Extrapolation uses
    a tangent estimated from a fraction of points at each end.

    Parameters
    ----------
    upp_x : array-like of shape (N,)
        X-coordinates of the detected upper aponeurosis edge.
    upp_y : array-like of shape (N,)
        Y-coordinates of the detected upper aponeurosis edge.
    low_x : array-like of shape (M,)
        X-coordinates of the detected lower aponeurosis edge.
    low_y : array-like of shape (M,)
        Y-coordinates of the detected lower aponeurosis edge.
    w : int
        Image width in pixels. Used to set the dense extrapolation grid.
    smooth_win : int, optional
        Window length for Savitzky–Golay smoothing of detected edges.
        Must be odd and smaller than input length. Default is 81.
    smooth_poly : int, optional
        Polynomial order for Savitzky–Golay smoothing. Default is 2.
    tail_frac : float, optional
        Fraction of points from each side (left and right) used to estimate
        tangent slope for extrapolation. Must be between 0 and 1.
        Default is 0.20 (20% of each side, minimum 5 points).

    Returns
    -------
    new_X : ndarray of shape (5000,)
        Dense shared x-grid spanning [-0.5*w, 1.5*w].
    new_Y_UA : ndarray of shape (5000,)
        Smoothed + extrapolated y-values of the upper aponeurosis on `new_X`.
        Values outside valid regions may be NaN if extrapolation fails.
    new_Y_LA : ndarray of shape (5000,)
        Smoothed + extrapolated y-values of the lower aponeurosis on `new_X`.
        Values outside valid regions may be NaN if extrapolation fails.
    segs_upper : list of (K,2) ndarrays
        List of polyline segments (detected, left extrapolated, right extrapolated)
        for drawing the upper aponeurosis. Each segment contains integer (x,y).
    segs_lower : list of (K,2) ndarrays
        List of polyline segments (detected, left extrapolated, right extrapolated)
        for drawing the lower aponeurosis.

    Notes
    -----
    - Extrapolation uses linear regression (1st-order polynomial) on the
      first/last `max(5, ceil(tail_frac * N))` points.
    - Detected aponeurosis parts are smoothed with Savitzky–Golay to reduce noise.
    - This function ensures both aponeuroses are represented on the same dense grid.

    Examples
    --------
    >>> import numpy as np
    >>> # Simulated aponeurosis edges
    >>> upp_x = np.array([50, 100, 200, 300, 400])
    >>> upp_y = np.array([100, 105, 110, 115, 120])
    >>> low_x = np.array([50, 100, 200, 300, 400])
    >>> low_y = np.array([300, 305, 310, 315, 320])
    >>> new_X, new_Y_UA, new_Y_LA, segs_upper, segs_lower = build_apo_from_edges(
    ...     upp_x, upp_y, low_x, low_y, w=512, tail_frac=0.4
    ... )
    >>> new_X.shape
    (5000,)
    >>> np.isfinite(new_Y_UA).sum() > 0
    True
    >>> len(segs_upper), len(segs_lower)
    (3, 3)
    """
    #TODO docs

    new_X = np.linspace(-0.5 * w, 1.5 * w, 5000).astype(np.float32)

    def _savgol(y):
        y = np.asarray(y, float)
        if y.size < 7:
            return y
        win = min(y.size if y.size % 2 == 1 else y.size - 1, smooth_win)
        win = max(7, win)
        if win % 2 == 0:
            win -= 1
        return savgol_filter(y, window_length=win, polyorder=min(smooth_poly, win - 1))

    def _both_side_extrap(x, y_sm):
        """
        Given detected x, smoothed y_sm, return y over new_X where:
          - inside [x0, x1]: interpolation of detected
          - left  of x0: left-tangent line (fit on left tail)
          - right of x1: right-tangent line (fit on right tail)
        Also return segments (detected, left-extrap, right-extrap) for drawing.
        """
        #TODO docs
        y_out = np.full_like(new_X, np.nan, dtype=np.float32)
        segs = []

        if len(x) < 2:
            return y_out, segs

        x = np.asarray(x, float)
        # detected interpolation in the observed domain
        yi = np.interp(new_X, x, y_sm, left=np.nan, right=np.nan)
        dom_mask = (new_X >= x[0]) & (new_X <= x[-1])
        y_out[dom_mask] = yi[dom_mask]

        # how many points in each tail
        n = len(x)
        k = max(5, int(np.ceil(tail_frac * n)))  # at least 5 points
        k = min(k, n)  # cap to length

        # LEFT tangent (first k points)
        if k >= 2:
            mL, bL = np.polyfit(x[:k], y_sm[:k], 1)
            left_mask = new_X < x[0]
            y_out[left_mask] = (mL * new_X[left_mask] + bL).astype(np.float32)

        # RIGHT tangent (last k points)
        if k >= 2:
            mR, bR = np.polyfit(x[-k:], y_sm[-k:], 1)
            right_mask = new_X > x[-1]
            y_out[right_mask] = (mR * new_X[right_mask] + bR).astype(np.float32)

        # build segments to draw
        # detected
        xr = new_X[dom_mask]
        yr = y_out[dom_mask]
        if xr.size > 1:
            segs.append(np.stack([xr.astype(int), yr.astype(int)], axis=1))
        # left extrapolated
        lm = new_X < x[0]
        xr = new_X[lm]
        yr = y_out[lm]
        if xr.size > 1 and np.isfinite(yr).any():
            segs.append(np.stack([xr.astype(int), yr.astype(int)], axis=1))
        # right extrapolated
        rm = new_X > x[-1]
        xr = new_X[rm]
        yr = y_out[rm]
        if xr.size > 1 and np.isfinite(yr).any():
            segs.append(np.stack([xr.astype(int), yr.astype(int)], axis=1))

        return y_out, segs

    # ---- Upper apo ----
    new_Y_UA = np.full_like(new_X, np.nan, dtype=np.float32)
    segs_upper = []
    if len(upp_x) > 1:
        ux = np.asarray(upp_x, float)
        uy_sm = _savgol(upp_y)
        new_Y_UA, segs_upper = _both_side_extrap(ux, uy_sm)

    # ---- Lower apo ----
    new_Y_LA = np.full_like(new_X, np.nan, dtype=np.float32)
    segs_lower = []
    if len(low_x) > 1:
        lx = np.asarray(low_x, float)
        ly_sm = _savgol(low_y)
        new_Y_LA, segs_lower = _both_side_extrap(lx, ly_sm)

    return new_X, new_Y_UA, new_Y_LA, segs_upper, segs_lower

def compute_muscle_thickness(upp_x, upp_y, low_x, low_y):
    """
    Estimate muscle thickness by computing the shortest distance between
    upper and lower aponeuroses in the central overlapping third of their x-values.

    Parameters
    ----------
    upp_x : np.ndarray of shape (N,)
        X-coordinates of the upper aponeurosis contour.
    upp_y : np.ndarray of shape (N,)
        Y-coordinates of the upper aponeurosis contour.
    low_x : np.ndarray of shape (M,)
        X-coordinates of the lower aponeurosis contour.
    low_y : np.ndarray of shape (M,)
        Y-coordinates of the lower aponeurosis contour.

    Returns
    -------
    float
        The minimum vertical distance (in pixels) between the upper and lower aponeuroses
        in the middle third of their overlapping x-range.

    Raises
    ------
    ValueError
        If there are fewer than 3 overlapping x-values between upper and lower aponeuroses.

    Notes
    -----
    This function assumes that both aponeuroses are already extracted as contours
    and that the x-values are integers or can be exactly matched.

    Examples
    --------
    >>> thickness = compute_muscle_thickness(
    ...     upp_x=np.array([10, 20, 30, 40, 50]),
    ...     upp_y=np.array([100, 95, 90, 85, 80]),
    ...     low_x=np.array([20, 30, 40, 50, 60]),
    ...     low_y=np.array([130, 125, 120, 115, 110])
    ... )
    >>> print(f"Muscle thickness (px): {thickness:.2f}")
    Muscle thickness (px): 30.00
    """
    shared_x = sorted(set(upp_x).intersection(set(low_x)))
    n_shared = len(shared_x)

    if n_shared < 3:
        return float("nan")  # instead of raising ValueError


    start_idx = int(n_shared * 0.33)
    end_idx = int(n_shared * 0.66)
    central_x_vals = shared_x[start_idx:end_idx]

    mindist = float("inf")

    for x in central_x_vals:
        try:
            upp_idx = np.where(upp_x == x)[0][0]
            low_idx = np.where(low_x == x)[0][0]

            dist = math.dist(
                (upp_x[upp_idx], upp_y[upp_idx]),
                (low_x[low_idx], low_y[low_idx])
            )

            if dist < mindist:
                mindist = dist

        except IndexError:
            continue

    return mindist

def optimize_fascicle_loop(
    contoursF3,
    new_Y_UA, new_Y_LA,
    new_X_UA, new_X_LA,
    width,
    min_pennation, max_pennation,
    filter_fascicles_func,
    fasc_cont_thresh,
    calib_dist,
):
    """
    Extract, extrapolate, and filter fascicle contours based on angle and length criteria.

    This function fits a line to each detected fascicle contour, extrapolates it 
    across the image width, and finds its intersections with the extrapolated upper 
    and lower aponeuroses. Fascicles that intersect both aponeuroses, have lengths 
    above the threshold, and fall within the allowed pennation angle range are kept.
    Optionally, overlapping fascicles can be filtered.

    Parameters
    ----------
    contoursF3 : list of (N, 1, 2) ndarray
        List of fascicle contours detected in the binary fascicle mask.
    new_Y_UA : ndarray of shape (M,)
        Y-values of the extrapolated upper aponeurosis curve.
    new_Y_LA : ndarray of shape (M,)
        Y-values of the extrapolated lower aponeurosis curve.
    new_X_UA : ndarray of shape (M,)
        X-values corresponding to `new_Y_UA`.
    new_X_LA : ndarray of shape (M,)
        X-values corresponding to `new_Y_LA`.
    width : int
        Image width in pixels. Used to define extrapolation grid.
    min_pennation : float
        Minimum acceptable pennation angle in degrees.
    max_pennation : float
        Maximum acceptable pennation angle in degrees.
    filter_fascicles_func : callable or None
        Optional function to further filter valid fascicles. Must accept and return
        a pandas.DataFrame with fascicle data.
    fasc_cont_thresh : int
        Minimum number of contour points required to consider a fascicle candidate.
    calib_dist : float or int
        Calibration distance in pixels between two 10 mm markers. 
        If provided, fascicle lengths are scaled to millimeters.

    Returns
    -------
    fascicle_data : pandas.DataFrame
        DataFrame with one row per accepted fascicle and columns:
        
        - ``x_low`` : int  
          Starting x-coordinate (lower intersection with aponeurosis).
        - ``x_high`` : int  
          Ending x-coordinate (upper intersection with aponeurosis).
        - ``y_low`` : int  
          Starting y-coordinate (lower intersection).
        - ``y_high`` : int  
          Ending y-coordinate (upper intersection).
        - ``coordsX`` : ndarray  
          X coordinates of fascicle line between intersections.
        - ``coordsY`` : ndarray  
          Y coordinates of fascicle line between intersections.
        - ``fasc_l`` : float  
          Fascicle length (px or mm if `calib_dist` is provided).
        - ``penn_a`` : float  
          Pennation angle in degrees relative to lower aponeurosis.

        If no valid fascicles are found, an empty DataFrame with these columns is returned.

    Notes
    -----
    - Fascicles are fitted with a first-order polynomial (straight line).
    - Intersections are determined by minimizing vertical distance to aponeuroses.
    - Angles are computed relative to the local slope of the lower aponeurosis.
    - `filter_fascicles_func` can be used to remove overlapping or outlier fascicles.

    Examples
    --------
    >>> import numpy as np
    >>> import cv2
    >>> import pandas as pd
    >>> # Dummy apo (straight lines) and fascicle contour
    >>> new_X = np.linspace(0, 512, 5000)
    >>> new_Y_UA = 100 + 0*new_X
    >>> new_Y_LA = 400 + 0*new_X
    >>> cnt = np.array([[[100, 120]], [[200, 250]], [[300, 380]]])  # fake fascicle
    >>> df = optimize_fascicle_loop(
    ...     contoursF3=[cnt],
    ...     new_Y_UA=new_Y_UA,
    ...     new_Y_LA=new_Y_LA,
    ...     new_X_UA=new_X,
    ...     new_X_LA=new_X,
    ...     width=512,
    ...     min_pennation=10,
    ...     max_pennation=40,
    ...     filter_fascicles_func=None,
    ...     fasc_cont_thresh=3,
    ...     calib_dist=98
    ... )
    >>> df[["fasc_l", "penn_a"]]
        fasc_l     penn_a
    0  39.7951  18.7423
    """
    #TODO docs 
    newX = np.linspace(-0.5*width, 1.5*width, 5000)

    data_rows = []
    dbg_good, dbg_bad = [], []

    print(f"Detected fascicle contours: {len(contoursF3)}")

    for cnt_idx, cnt in enumerate(contoursF3):
        if len(cnt) <= fasc_cont_thresh:   # ⚠️ was reversed in your code
        #     print(f"❌ Skipping contour {cnt_idx}: too short ({len(cnt)} pts)")
            continue

        x_f, y_f = contourEdge("B", cnt)
        if len(x_f) < 2:
        #     print(f"❌ Contour {cnt_idx}: not enough edge points")
            continue

        # Line fit
        z = np.polyfit(np.array(x_f, float), np.array(y_f, float), 1)
        f = np.poly1d(z)
        newY = f(newX)

        difU = np.abs(newY - (new_Y_UA))
        difL = np.abs(newY - new_Y_LA)


        locU, locL = int(np.argmin(difU)), int(np.argmin(difL))
        xU, yU = newX[locU], newY[locU]
        xL, yL = newX[locL], newY[locL]

        # Assume bad until proven good
        dbg_bad.append((xL, yL, xU, yU))

        # Reject conditions
        if locU <= 0 or locU >= len(newX)-1: 
        #    print(f"❌ Contour {cnt_idx}: U out of bounds")
            continue
        if locL <= 0 or locL >= len(newX)-1: 
        #    print(f"❌ Contour {cnt_idx}: L out of bounds")
            continue
        if locL >= locU:
        #     print(f"❌ Contour {cnt_idx}: L >= U (invalid order)")
             continue

        coordsX, coordsY = newX[locL:locU], newY[locL:locU]
        if len(coordsX) < 2:
        #    print(f"❌ Contour {cnt_idx}: <2 points after crop")
            continue

        step = 10 # segment length of apo to calculate angle
        max_step = (len(newX) - 1) - locL
        j = int(min(step, max(1, max_step)))  # ensure 1 <= j <= max_step

        # slope of lower apo segment near the intersection with the fascicle
        den = (new_X_LA[locL + j] - new_X_LA[locL])
        if den == 0 or not np.isfinite(den):
            continue

        apo_slope = (new_Y_LA[locL] - new_Y_LA[locL + j]) / den
        Apoangle = 90.0 + abs(np.degrees(np.arctan(apo_slope)))

        # FascAng
        num = (coordsX[0] - coordsX[-1])
        den = (new_Y_LA[locL] - new_Y_UA[locU])
        if den == 0 or not np.isfinite(den):
            continue

        FascAng = -np.degrees(np.arctan(num / den))
        ActualAng = Apoangle - FascAng

        if not (min_pennation <= ActualAng <= max_pennation):
            print(f"❌ Contour {cnt_idx}: pennation {ActualAng:.1f}° out of bounds")
            continue

        length1 = float(np.hypot(coordsX[-1] - coordsX[0], coordsY[-1] - coordsY[0]))

        # Valid fascicle
        dbg_good.append((xL, yL, xU, yU))
        dbg_bad.pop()   # remove from rejected list

        data_rows.append({
            "x_low": int(coordsX[0]), "x_high": int(coordsX[-1]),
            "y_low": int(coordsY[0]), "y_high": int(coordsY[-1]),
            "coordsX": coordsX, "coordsY": coordsY,
            "fasc_l": length1, "penn_a": float(ActualAng),
        })

    fascicle_data = pd.DataFrame(data_rows) if data_rows else pd.DataFrame(
        columns=["x_low","x_high","y_low","y_high","coordsX","coordsY","fasc_l","penn_a"]
    )

    if filter_fascicles_func and not fascicle_data.empty:
        fascicle_data = filter_fascicles_func(fascicle_data)

    if calib_dist and not fascicle_data.empty:
        fascicle_data["fasc_l"] /= (calib_dist / 10)

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
    frame_callback : bool
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
        fasc_cont_thresh, min_width, max_pennation, min_pennation = [
            int(dic[key])
            for key in [
                "fascicle_length_threshold",
                "minimal_muscle_width",
                "maximal_pennation_angle",
                "minimal_pennation_angle",
            ]
        ]
        apo_threshold, apo_length_thresh, fasc_threshold = [
            float(dic[key])
            for key in [
                "aponeurosis_detection_threshold",
                "aponeurosis_length_threshold",
                "fascicle_detection_threshold",
            ]
        ]

        # Define empty lists for parameter storing
        fasc_l_all, pennation_all, x_lows_all, x_highs_all, thickness_all = (
            [] for _ in range(5)
        )


        last3 = deque(maxlen=3)

        height, width = 512, 512

        print("\nProcessing frames...")

        # Process sequentially
        frame_idx = 0
        while frame_idx < vid_len:
            if gui.should_stop:
                break

            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()

            if flip == "flip":
                frame = cv2.flip(frame, 1)

            if frame_idx % step != 0:
                # still write original frame to video for sync
                vid_out.write(frame)

                # append NaNs so output lists keep same length as video
                fasc_l_all.append([float("nan")])
                pennation_all.append([float("nan")])
                x_lows_all.append([float("nan")])
                x_highs_all.append([float("nan")])
                thickness_all.append(float("nan"))

                frame_idx += 1
                continue

            img_orig = frame.copy()
            h, w = img_orig.shape[:2]   # FIX unpack order

            # Build gray frame for IFSS
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
            gray = gray.astype(np.float32) / 255.0
            gray = gray[..., None]  # (H, W, 1)
            last3.append(gray)

            # Build single-frame input only on step frames
            if frame_idx % step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
                single_input = (rgb.astype(np.float32) / 255.0)[None, ...]  # (1, H, W, 3)

                # ----- PREDICT APO -----
                h, w = img_orig.shape[:2]
                pred_apo = apo_model.predict(single_input)
                pred_apo_t = (pred_apo > apo_threshold).astype(np.uint8)
                pred_apo = cv2.resize(pred_apo[0], (w, h), interpolation=cv2.INTER_NEAREST)
                pred_apo_t = cv2.resize(pred_apo_t[0], (w, h), interpolation=cv2.INTER_NEAREST)

                # ----- PREDICT FASC -----
                if segmentation_mode == "stacked":
                    if len(last3) < 3:
                        frame_idx += 1
                        continue  # not enough frames for a stack
                    stacked = np.stack(last3, axis=0)[None, ...]  # (1, 3, H, W, 1)
                    pred_fasc = fasc_model([stacked], training=False)
                    pred_fasc = tf.clip_by_value(pred_fasc, 0, 1).numpy()
                    pred_fasc = np.array(pred_fasc[:, 1, :, :, 0])[..., None]
                else:
                    pred_fasc = fasc_model.predict(single_input)

                pred_fasc_t = (pred_fasc > fasc_threshold).astype(np.uint8)
                pred_fasc = cv2.resize(pred_fasc[0], (w, h), interpolation=cv2.INTER_NEAREST)
                pred_fasc_t = cv2.resize(pred_fasc_t[0], (w, h), interpolation=cv2.INTER_NEAREST)

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
                pts = list(contour)
                ptsT = sorted(pts, key=lambda k: [k[0][0], k[0][1]])
                allx = []
                ally = []
                for aa in range(0, len(ptsT)):
                    allx.append(ptsT[aa][0, 0])
                    ally.append(ptsT[aa][0, 1])
                app = np.array(list(zip(allx, ally)))
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
            for i in range(len(contours_re2)):
                for j in range(i + 1, len(contours_re2)):
                    x_gap = xs1[j] - xs2[i]
                    y_diff = abs(ys2[i] - ys1[j])

                    if 0 < x_gap <= 100 and y_diff <= 20:
                        m = np.vstack((contours_re2[i], contours_re2[j]))
                        cv2.drawContours(maskT, [m], 0, 255, -1)


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
            mask_apoE = np.zeros((thresh.shape), np.uint8)

            # Select only long contours and draw to mask
            contoursE = [i for i in contoursE if len(i) > apo_length_thresh]
            for contour in contoursE:
                cv2.drawContours(mask_apoE, [contour], 0, 255, -1)

            # Sort contours again from top to bottom
            contoursE, _ = sortContours(contoursE)

            # Continue only when 2 or more aponeuroses were detected
            if len(contoursE) >= 2:

                # --- Use detected edges ---
                # (recommend: upper = top edge of the upper contour; lower = top edge of lower contour)
                upp_x, upp_y = contourEdge("B", contoursE[0])
                if contoursE[1][0, 0, 1] > (contoursE[0][0, 0, 1] + min_width):
                    low_x, low_y = contourEdge("T", contoursE[1])
                else:
                    low_x, low_y = contourEdge("T", contoursE[2])

                # --- Build apo arrays from edges, extrapolating ONLY where needed ---
                new_X, new_Y_UA, new_Y_LA, segs_upper, segs_lower = build_apo_from_edges(
                    upp_x=upp_x, upp_y=upp_y,
                    low_x=low_x, low_y=low_y,
                    w=w,                 # image width
                    smooth_win=81,       
                    smooth_poly=2
                )

                # --- Thickness (central third of valid overlap) ---
                valid = np.isfinite(new_Y_UA) & np.isfinite(new_Y_LA)
                if np.count_nonzero(valid) >= 30:
                    idx = np.where(valid)[0]
                    s = idx[int(0.33*len(idx))]
                    e = idx[int(0.66*len(idx))]
                    midthick = float(np.nanmin(np.abs(new_Y_LA[s:e] - new_Y_UA[s:e])))
                else:
                    midthick = float("nan")

                # --- Region mask between extrapolated aponeuroses ---
                ex_mask = np.zeros((h, w), np.uint8)

                xs = new_X.astype(int)
                ua = new_Y_UA
                la = new_Y_LA
                for xi, yu, yl in zip(xs, ua, la):
                    if np.isnan(yu) or np.isnan(yl):
                        continue
                    if xi < 0 or xi >= w: 
                        continue
                    y0 = int(np.floor(min(yu, yl)))
                    y1 = int(np.ceil (max(yu, yl)))
                    if y1 <= y0:
                        continue
                    y0 = max(0, min(h-1, y0))
                    y1 = max(0, min(h-1, y1))
                    ex_mask[y0:y1, xi] = 255


                # --- Draw detected & extrapolated apo on imgT (overlay) ---
                # Upper: draw detected in green-ish, extrapolated in darker green
                
                imgT = np.zeros((h, w, 3), np.uint8)
                for k, seg in enumerate(segs_upper):
                    if len(seg) < 2: 
                        continue
                    pts = seg.reshape(-1, 1, 2)
                    color = (0, 255, 0) if k == 0 else (0, 170, 0)
                    cv2.polylines(imgT, [pts], isClosed=False, color=color, thickness=3)

                # Lower: draw detected in blue-ish, extrapolated in darker blue
                for k, seg in enumerate(segs_lower):
                    if len(seg) < 2:
                        continue
                    pts = seg.reshape(-1, 1, 2)
                    color = (0, 128, 255) if k == 0 else (0, 90, 180)
                    cv2.polylines(imgT, [pts], isClosed=False, color=color, thickness=3)

                # --- Feed to existing fascicle pipeline (unchanged) ---
                _, threshF = cv2.threshold(pred_fasc_t, 0, 255, cv2.THRESH_BINARY)
                threshF = threshF.astype("uint8")
                contoursF, _ = cv2.findContours(threshF, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                maskF = np.zeros(threshF.shape, np.uint8)
                for contour in contoursF:
                    if len(contour) > fasc_cont_thresh:
                        cv2.drawContours(maskF, [contour], 0, 255, -1)

                mask_Fi = maskF & ex_mask
                # Keep all fascicle contours from maskF
                contoursF2, _ = cv2.findContours(maskF, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                contoursF3 = [c for c in contoursF2 if len(c) > fasc_cont_thresh]


                # (angles/lengths etc. – keep your existing call)
                fascicle_data = optimize_fascicle_loop(
                    contoursF3,
                    new_Y_UA=new_Y_UA,
                    new_Y_LA=new_Y_LA,
                    new_X_UA=new_X,
                    new_X_LA=new_X,
                    width=w,
                    min_pennation=min_pennation,
                    max_pennation=max_pennation,
                    filter_fascicles_func=filter_fascicles if filter_fasc == 1 else None,
                    fasc_cont_thresh=fasc_cont_thresh,
                    calib_dist=calib_dist,
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
                
                dbg_good = getattr(fascicle_data, "attrs", {}).get("dbg_good", [])
                dbg_bad  = getattr(fascicle_data, "attrs", {}).get("dbg_bad", [])

                # draw good fascicle endpoints
                for (xL, yL, xU, yU) in dbg_good:
                    cv2.circle(imgT, (xL, yL), 4, (0,255,255), -1)  # yellow
                    cv2.circle(imgT, (xU, yU), 4, (255,255,0), -1)  # cyan

                # draw rejected fascicles (small red x)
                for (xL, yL, xU, yU) in dbg_bad:
                    cv2.drawMarker(imgT, (xL, yL), (0,0,255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=8, thickness=2)
                    cv2.drawMarker(imgT, (xU, yU), (0,0,255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=8, thickness=2)

            # Return empty variables when no two aponeuroses were detected
            else:
                fasc_l = []
                pennation = []
                x_low1 = []
                x_high1 = []
                imgT = np.zeros((h, w, 3), np.uint8)
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

            if calib_dist:
                unit = "mm"
            else:
                unit = "px"

            # --- make overlay inputs consistent (same size, 3 channels, uint8) ---
            base = img_orig
            overlay = imgT

            # ensure size match
            if overlay.shape[:2] != base.shape[:2]:
                overlay = cv2.resize(overlay, (base.shape[1], base.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)

            # ensure 3 channels on both
            if base.ndim == 2:
                base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
            if overlay.ndim == 2:
                overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

            # ensure dtype uint8
            if base.dtype != np.uint8:
                base = base.astype(np.uint8)
            if overlay.dtype != np.uint8:
                overlay = overlay.astype(np.uint8)

            # blend
            comb = cv2.addWeighted(base, 1.0, overlay, 0.8, 0.0)

            vid_out.write(comb)  # Write each image to video file
            cv2.putText(
                comb,
                ("Frame: " + str(frame_idx + 1) + " of " + str(vid_len)),
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
            cv2.putText(
                comb,
                (
                    "Fascicle length: "
                    + str("%.2f" % np.median(fasc_l_all[-1]) + f" {unit}")
                ),
                (125, 410),
                cv2.FONT_HERSHEY_DUPLEX,
                0.75,
                (249, 249, 249),
            )
            cv2.putText(
                comb,
                (
                    "Thickness at centre: "
                    + str("%.1f" % thickness_all[-1])
                    + f" {unit}"
                ),
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

            
            frame_idx +=1

        # Release video and close all analysis windows when analysis finished
        cap.release()
        vid_out.release()
        cv2.destroyAllWindows()

        return fasc_l_all, pennation_all, x_lows_all, x_highs_all, thickness_all

    # Check if model path is correct
    except OSError:
        error_details = traceback.format_exc()
        tk.messagebox.showerror(
            "Information", "Apo/Fasc model path is incorrect.\n\n" + error_details
        )
        gui.should_stop = False
        gui.is_running = False
        gui.do_break()
        return

    except ValueError:
        error_details = traceback.format_exc()
        tk.messagebox.showerror(
            "'segmentation_mode' Error",
            "Choose the correct segmentation mode for your model.\n\n" + error_details,
        )
        gui.should_stop = False
        gui.is_running = False
        gui.do_break()
        return

    except TypeError:
        error_details = traceback.format_exc()
        tk.messagebox.showerror(
            "'fascicle_detection_threshold' Error",
            "Choose a higher fascicle detection threshold.\n\n" + error_details,
        )
        gui.should_stop = False
        gui.is_running = False
        gui.do_break()
        return

    except UnboundLocalError:
        error_details = traceback.format_exc()
        tk.messagebox.showerror(
            "'flip' Error",
            "Choose the correct flip value according to fascicle orientation.\n\n"
            + error_details,
        )
        gui.should_stop = False
        gui.is_running = False
        gui.do_break()
        return

    except IndexError:
        error_details = traceback.format_exc()
        tk.messagebox.showerror(
            "Fascicle detection Error",
            "Adapt analysis parameters for valid detection.\n\n" + error_details,
        )
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
