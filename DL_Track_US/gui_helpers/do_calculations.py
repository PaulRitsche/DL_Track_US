"""
Description
-----------
This module contains functions to calculate muscle architectural
parameters based on binary segmentations by convolutional neural networks.
The parameters include muscle thickness, pennation angle and fascicle length.
First, input images are segmented by the CNNs. Then the predicted aponeuroses
and fascicle fragments are thresholded and filtered. Fascicle fragments
and aponeuroses are extrapolated and the intersections determined.
This module is specifically designed for single image analysis.
The architectural parameters are calculated and the results are plotted.

Functions scope
---------------
sortContours
    Function to sort detected contours from proximal to distal.
contourEdge
    Function to find only the coordinates representing one edge
    of a contour.
doCalculations
    Function to compute muscle architectural parameters based on
    convolutional neural network segmentation.

Notes
-----
Additional information and usage examples can be found at the respective
functions documentations.
"""

import math
import tkinter as tk

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import copy

from scipy.signal import savgol_filter
from skimage.morphology import skeletonize
from skimage.transform import resize


def sortContours(cnts: list):
    """Function to sort detected contours from proximal to distal.

    The input contours belond to the aponeuroses and are sorted
    based on their coordinates, from smallest to largest.
    Moreover, for each detected contour a bounding box is built.
    The bounding boxes are sorted as well. They are however not
    needed for further analyses.

    Parameters
    ----------
    cnts : list
        List of arrays containing the detected aponeurosis
        contours.

    Returns
    -------
    cnts : tuple
        Tuple containing arrays of sorted contours.
    bounding_boxes : tuple
        Tuple containing tuples with sorted bounding boxes.

    Examples
    --------
    >>> sortContours(cnts=[array([[[928, 247]], ... [[929, 247]]],
    dtype=int32),
    ((array([[[228,  97]], ... [[229,  97]]], dtype=int32),
    (array([[[228,  97]], ... [[229,  97]]], dtype=int32),
    (array([[[928, 247]], ... [[929, 247]]], dtype=int32)),
    ((201, 97, 747, 29), (201, 247, 750, 96))
    """
    try:
        # initialize the reverse flag and sort index
        i = 1
        # construct the list of bounding boxes and sort them from top to bottom
        bounding_boxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, bounding_boxes) = zip(
            *sorted(zip(cnts, bounding_boxes), key=lambda b: b[1][i], reverse=False)
        )
    except ValueError:
        tk.messagebox.showerror("Information", "Aponeurosis length threshold too big.")

    return (cnts, bounding_boxes)


def contourEdge(edge: str, contour: list) -> np.ndarray:
    """Function to find only the coordinates representing one edge
    of a contour.

    Either the upper or lower edge of the detected contours is
    calculated. From the contour detected lower in the image,
    the upper edge is searched. From the contour detected
    higher in the image, the lower edge is searched.

    Parameters
    ----------
    edge : {"T", "B"}
        String variable defining the type of edge that is
        searched. The variable can be either "T" (top) or
        "B" (bottom).
    contour : list
        List variable containing sorted contours.

    Returns
    -------
    x : np.ndarray
        Array variable containing all x-coordinates from the
        detected contour.
    y : np.ndarray
        Array variable containing all y-coordinated from the
        detected contour.

    Examples
    --------
    >>> contourEdge(edge="T", contour=[[[195 104]] ... [[196 104]]])
    [196 197 198 199 200 ... 952 953 954 955 956 957],
    [120 120 120 120 120 ... 125 125 125 125 125 125]
    """
    # Turn tuple into list
    pts = list(contour)
    # sort conntours
    ptsT = sorted(pts, key=lambda k: [k[0][0], k[0][1]])

    # Get x and y coordinates from contour
    allx = []
    ally = []
    for a in range(0, len(ptsT)):
        allx.append(ptsT[a][0, 0])
        ally.append(ptsT[a][0, 1])
    # Get rid of doubles
    un = np.unique(allx)

    # Filter x and y coordinates from cont according to selected edge
    leng = len(un) - 1
    x = []
    y = []
    for each in range(5, leng - 5):  # Ignore 1st and last 5 points
        indices = [i for i, x in enumerate(allx) if x == un[each]]
        if edge == "T":
            loc = indices[0]
        else:
            loc = indices[-1]
        x.append(ptsT[loc][0, 0])
        y.append(ptsT[loc][0, 1])

    return np.array(x), np.array(y)


def filter_fascicles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out fascicles that intersect with their neighboring fascicles based on their x_low and x_high values.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing the fascicle data. Expected columns include 'x_low', 'y_low', 'x_high', and 'y_high'.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the fascicles that do not intersect with their neighbors.

    ExampleS
    --------
    >>> data = {'x_low': [1, 3, 5], 'y_low': [1, 2, 3], 'x_high': [4, 6, 7], 'y_high': [4, 5, 6]}
    >>> df = pd.DataFrame(data)
    >>> print(filter_fascicles(df))
       x_low  y_low  x_high  y_high
    0      1      1       4       4
    2      5      3       7       6
    """

    df = df.sort_values(by="x_low").reset_index(drop=True)
    df["keep"] = True

    x_lows = df["x_low"].values
    x_highs = df["x_high"].values

    for i in range(len(df)):
        for j in range(
            i + 1, min(i + 3, len(df))
        ):  # Check the current fascicle and the two next ones
            # Check if the next fascicle(s) intersect
            if x_lows[i] <= x_lows[j] and x_highs[i] >= x_highs[j]:
                df.at[i, "keep"] = False
            elif x_lows[j] <= x_lows[i] and x_highs[j] >= x_highs[i]:
                df.at[j, "keep"] = False

    return df[df["keep"]].drop(columns=["keep"])

def optimize_fascicle_loop(
    contoursF3,
    new_Y_UA,
    new_Y_LA,
    new_X_UA,
    new_X_LA,
    width,
    min_pennation,
    max_pennation,
    filter_fascicles_func=None,
    fasc_cont_thresh=40,
    calib_dist=None,
    spacing=10,
):
    """
    Extract, extrapolate, and filter fascicle contours.

    This version is usable for both image and video analysis.
    If calib_dist is given, fascicle length is converted from pixels to mm
    using calib_dist / spacing.
    """
    newX = np.linspace(-0.5 * width, 1.5 * width, 5000)

    data_rows = []

    for cnt_idx, cnt in enumerate(contoursF3):
        if len(cnt) <= fasc_cont_thresh:
            continue

        x_f, y_f = contourEdge("B", cnt)

        if len(x_f) < 2:
            continue

        # Fit fascicle fragment and extrapolate it
        z = np.polyfit(np.array(x_f, float), np.array(y_f, float), 1)
        f = np.poly1d(z)
        newY = f(newX)

        # Find closest intersections with upper and lower aponeuroses
        difU = np.abs(newY - new_Y_UA)
        difL = np.abs(newY - new_Y_LA)

        locU = int(np.argmin(difU))
        locL = int(np.argmin(difL))

        # Reject invalid intersection positions
        if locU <= 0 or locU >= len(newX) - 1:
            continue

        if locL <= 0 or locL >= len(newX) - 1:
            continue

        if locL >= locU:
            continue

        coordsX = newX[locL:locU]
        coordsY = newY[locL:locU]

        if len(coordsX) < 2:
            continue

        # Compute local lower aponeurosis angle
        step = 10
        max_step = (len(newX) - 1) - locL
        j = int(min(step, max(1, max_step)))

        den = new_X_LA[locL + j] - new_X_LA[locL]

        if den == 0 or not np.isfinite(den):
            continue

        apo_slope = (new_Y_LA[locL] - new_Y_LA[locL + j]) / den
        Apoangle = 90.0 + abs(np.degrees(np.arctan(apo_slope)))

        # Compute fascicle angle
        num = coordsX[0] - coordsX[-1]
        den = new_Y_LA[locL] - new_Y_UA[locU]

        if den == 0 or not np.isfinite(den):
            continue

        FascAng = -np.degrees(np.arctan(num / den))
        ActualAng = Apoangle - FascAng

        if not (min_pennation <= ActualAng <= max_pennation):
            continue

        # Fascicle length in pixels
        length1 = float(
            np.hypot(
                coordsX[-1] - coordsX[0],
                coordsY[-1] - coordsY[0],
            )
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
                "penn_a": float(ActualAng),
            }
        )

    fascicle_data = (
        pd.DataFrame(data_rows)
        if data_rows
        else pd.DataFrame(
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
    )

    if filter_fascicles_func is not None and not fascicle_data.empty:
        fascicle_data = filter_fascicles_func(fascicle_data)

    if calib_dist and not fascicle_data.empty:
        scale = calib_dist / int(spacing)
        fascicle_data["fasc_l"] = fascicle_data["fasc_l"] / scale

    return fascicle_data

def build_apo_from_edges(
    upp_x, upp_y, low_x, low_y, w,
    smooth_win: int = 81,
    smooth_poly: int = 2,
    tail_frac: float = 0.50,
):
    """
    Build dense, smoothed, and extrapolated aponeurosis curves from detected edges.
    This matches the aponeurosis logic used in do_calculations_video.py.
    """
    new_X = np.linspace(-0.5 * w, 1.5 * w, 5000).astype(np.float32)

    def _savgol(y):
        y = np.asarray(y, float)
        if y.size < 7:
            return y

        win = min(y.size if y.size % 2 == 1 else y.size - 1, smooth_win)
        win = max(7, win)

        if win % 2 == 0:
            win -= 1

        return savgol_filter(
            y,
            window_length=win,
            polyorder=min(smooth_poly, win - 1),
        )

    def _both_side_extrap(x, y_sm):
        y_out = np.full_like(new_X, np.nan, dtype=np.float32)
        segs = []

        if len(x) < 2:
            return y_out, segs

        x = np.asarray(x, float)

        # Interpolate detected part
        yi = np.interp(new_X, x, y_sm, left=np.nan, right=np.nan)
        dom_mask = (new_X >= x[0]) & (new_X <= x[-1])
        y_out[dom_mask] = yi[dom_mask]

        # Tail length for extrapolation
        n = len(x)
        k = max(5, int(np.ceil(tail_frac * n)))
        k = min(k, n)

        # Left extrapolation
        if k >= 2:
            mL, bL = np.polyfit(x[:k], y_sm[:k], 1)
            left_mask = new_X < x[0]
            y_out[left_mask] = (mL * new_X[left_mask] + bL).astype(np.float32)

        # Right extrapolation
        if k >= 2:
            mR, bR = np.polyfit(x[-k:], y_sm[-k:], 1)
            right_mask = new_X > x[-1]
            y_out[right_mask] = (mR * new_X[right_mask] + bR).astype(np.float32)

        # Drawing segments
        xr = new_X[dom_mask]
        yr = y_out[dom_mask]
        if xr.size > 1:
            segs.append(np.stack([xr.astype(int), yr.astype(int)], axis=1))

        left_mask = new_X < x[0]
        xr = new_X[left_mask]
        yr = y_out[left_mask]
        if xr.size > 1 and np.isfinite(yr).any():
            segs.append(np.stack([xr.astype(int), yr.astype(int)], axis=1))

        right_mask = new_X > x[-1]
        xr = new_X[right_mask]
        yr = y_out[right_mask]
        if xr.size > 1 and np.isfinite(yr).any():
            segs.append(np.stack([xr.astype(int), yr.astype(int)], axis=1))

        return y_out, segs

    # Upper aponeurosis
    new_Y_UA = np.full_like(new_X, np.nan, dtype=np.float32)
    segs_upper = []

    if len(upp_x) > 1:
        ux = np.asarray(upp_x, float)
        uy_sm = _savgol(upp_y)
        new_Y_UA, segs_upper = _both_side_extrap(ux, uy_sm)

    # Lower aponeurosis
    new_Y_LA = np.full_like(new_X, np.nan, dtype=np.float32)
    segs_lower = []

    if len(low_x) > 1:
        lx = np.asarray(low_x, float)
        ly_sm = _savgol(low_y)
        new_Y_LA, segs_lower = _both_side_extrap(lx, ly_sm)

    return new_X, new_Y_UA, new_Y_LA, segs_upper, segs_lower

def doCalculations(  
    original_image: np.ndarray,
    img_copy: np.ndarray,
    h: int,
    w: int,
    calib_dist: int,
    spacing: int,
    model_apo,
    model_fasc,
    dictionary: dict,
    filter_fasc: bool,
    image_callback=None,
):
    """Function to compute muscle architectural parameters based on
    convolutional neural network segmentation in images.

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
    original_image : np.ndarray
            Normalized, reshaped and rescaled rayscale image to be
            analysed as a numpy array. The image must
            be loaded prior to model inputting, specifying a path
            is not valid.
    img_copy : np.ndarray
        A copy of the input image.
    h : int
        Integer variable containing the height of the input image (img).
    w : int
        Integer variable containing the width of the input image (img).
    calib_dist : int
        Integer variable containing the distance between the two
        specified point in pixel units. This value was either computed
        automatically or manually. Must be non-negative. If "None", the
        values are outputted in pixel units.
    spacing : {10, 5, 15, 20}
        Integer variable containing the known distance in milimeter
        between the two placed points by the user or the scaling bars
        present in the image. This can be 5, 10, 15 or 20 milimeter.
        Must be non-negative and non-zero.
    model_apo :
        Contains keras model for prediction of aponeuroses
    model_fasc :
        Contains keras model for prediction of fascicles
    dictionary : dict
        Dictionary variable containing analysis parameters.
        These include must include apo_threshold, apo_length_tresh, fasc_threshold,
        fasc_cont_threshold, min_width, max_pennation,
        min_pennation.
    filter_fasc : bool
        If True, fascicles will be filtered so that no crossings are included.
        This may reduce number of totally detected fascicles.
    image_callback:
        Callback function to update the image display. If None, no callback is used.

    Returns
    -------
    fasc_l : list
        List variable contianing the estimated fascicle lengths
        based on the segmented fascicle fragments in pixel units
        as float. If calib_dist is specified, then the length is computed
        in centimeter.
    pennation : list
        List variable containing the estimated pennation angles
        based on the segmented fascicle fragments and aponeuroses
        as float.
    x_low1 : list
        List variable containing the estimated x-coordinates
        of the lower edge from the upper aponeurosis as integers.
    x_high1 : list
        List variable containing the estimated x-coordinates
        of the upper edge from the lower aponeurosis as integers.
    midthick : float
        Float variable containing the estimated distance
        between the lower and upper aponeurosis in pixel units.
        If calib_dist is specified, then the distance is computed
        in centimeter.
    fig : matplotlib.figure
        Figure including the input image, the segmented aponeurosis and
        the extrapolated fascicles.

    Notes
    -----
    For more detailed documentation, see the respective functions
    documentation.

    Examples
    --------
    >>> doCalculations(img=[[[[0.10113753 0.09391343 0.09030136]
                           [0.10878626 0.10101581 0.09713058]
                           [0.10878634 0.10101589 0.09713066]
                           ...
                           [0.         0.         0.        ]
                           [0.         0.         0.        ]
                           [0.         0.         0.        ]]]],
                       img_copy=[[[[0.10113753 0.09391343 0.09030136]
                           [0.10878626 0.10101581 0.09713058]
                           [0.10878634 0.10101589 0.09713066]
                           ...
                           [0.         0.         0.        ]
                           [0.         0.         0.        ]
                           [0.         0.         0.        ]]]],
                        h=512, w=512,calib_dist=None, spacing=10,
                        filename=test1,
                        apo_modelpath="C:/Users/admin/Documents/DL_Track/Models_DL_Track/Final_models/model-VGG16-fasc-BCE-512.h5",
                        fasc_modelpath="C:/Users/admin/Documents/DL_Track/Models_DL_Track/Final_models/model-apo-VGG-BCE-512.h5",
                        scale_statement=None,
                        dictionary={'apo_treshold': '0.2', 'apo_length_tresh': '600', fasc_threshold': '0.05', 'fasc_cont_thresh': '40', 'min_width': '60', 'min_pennation': '10', 'max_pennation': '40'},
                        filter_fasc = False)
    [1030.1118966321328, 1091.096002143386, ..., 1163.07073327008, 1080.0001937069776, 976.6099281240987]
    [19.400700671533016, 18.30126098122986, ..., 18.505345607096586, 18.727693601171197, 22.03704574228162]
    [441, 287, 656, 378, 125, 15, ..., -392, -45, -400, -149, -400]
    [1410, 1320, 1551, 1351, 1149, ..., 885, 937, 705, 869, 507]
    348.1328577
    """
    matplotlib.use("Agg")

    # Get settings
    dic = dictionary

    # Get variables from dictionary
    fasc_cont_thresh = int(dic["fascicle_length_threshold"])
    min_width = int(dic["minimal_muscle_width"])
    max_pennation = int(dic["maximal_pennation_angle"])
    min_pennation = int(dic["minimal_pennation_angle"])
    apo_threshold = float(dic["aponeurosis_detection_threshold"])
    fasc_threshold = float(dic["fascicle_detection_threshold"])
    apo_length_tresh = int(dic["aponeurosis_length_threshold"])

    pred_apo = model_apo.predict(original_image)
    pred_apo_t = (pred_apo > apo_threshold).astype(np.uint8)
    pred_apo_t = resize(pred_apo_t, (1, h, w, 1))
    apo_image = np.reshape(pred_apo_t, (h, w))

    # load the fascicle model
    pred_fasc = model_fasc.predict(original_image)
    pred_fasc_t = (pred_fasc > fasc_threshold).astype(np.uint8)  # SET FASC THS
    pred_fasc_t = resize(pred_fasc_t, (1, h, w, 1))
    fas_image = np.reshape(pred_fasc_t, (h, w))
    tf.keras.backend.clear_session()

    fasc_l = []
    pennation = []
    x_low = []
    x_high = []

    # Compute contours to identify the aponeuroses
    _, thresh = cv2.threshold(apo_image, 0, 255, cv2.THRESH_BINARY)
    thresh = thresh.astype("uint8")
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    contours_re = []
    for contour in contours:  # Remove any contours that are very small
        if len(contour) > apo_length_tresh:
            contours_re.append(contour)
    contours = contours_re

    # Check whether contours are detected
    # If not, break function
    if len(contours) < 1:
        return None, None, None, None, None, None

    contours, _ = sortContours(contours)  # Sort contours from top to bottom

    # mask_apo = np.zeros(thresh.shape,np.uint8)
    contours_re2 = []
    for contour in contours:
        #     cv2.drawContours(mask_apo,[contour],0,255,-1)
        pts = list(contour)
        ptsT = sorted(
            pts, key=lambda k: [k[0][0], k[0][1]]
        )  # Sort each contour based on x values
        allx = []
        ally = []
        for a in range(0, len(ptsT)):
            allx.append(ptsT[a][0, 0])
            ally.append(ptsT[a][0, 1])
        app = np.array(list(zip(allx, ally)))
        contours_re2.append(app)

    # Merge nearby contours
    # countU = 0
    xs1 = []
    xs2 = []
    ys1 = []
    ys2 = []
    maskT = np.zeros(thresh.shape, np.uint8)
    for cnt in contours_re2:
        ys1.append(cnt[0][1])
        ys2.append(cnt[-1][1])
        xs1.append(cnt[0][0])
        xs2.append(cnt[-1][0])
        cv2.drawContours(maskT, [cnt], 0, 255, -1)

    for i in range(len(contours_re2)):
        for j in range(i + 1, len(contours_re2)):
            x_gap = xs1[j] - xs2[i]
            y_diff = abs(ys2[i] - ys1[j])

            if 0 < x_gap <= 100 and y_diff <= 20:
                m = np.vstack((contours_re2[i], contours_re2[j]))
                cv2.drawContours(maskT, [m], 0, 255, -1)

    maskT[maskT > 0] = 1
    skeleton = skeletonize(maskT).astype(np.uint8)
    kernel = np.ones((3, 7), np.uint8)
    dilate = cv2.dilate(skeleton, kernel, iterations=15)
    erode = cv2.erode(dilate, kernel, iterations=10)

    contoursE, hierarchy = cv2.findContours(
        erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    mask_apoE = np.zeros(thresh.shape, np.uint8)

    contoursE = [
        i for i in contoursE if len(i) > apo_length_tresh
    ]  # Remove any contours that are very small

    for contour in contoursE:
        cv2.drawContours(mask_apoE, [contour], 0, 255, -1)
    contoursE, _ = sortContours(contoursE)

    # Only continues beyond this point if 2 aponeuroses can be detected
    if len(contoursE) >= 2:
        # Get the x,y coordinates of the upper/lower edge of the 2 aponeuroses
        upp_x, upp_y = contourEdge("B", contoursE[0])

        if contoursE[1][0, 0, 1] > contoursE[0][0, 0, 1] + min_width:
            low_x, low_y = contourEdge("T", contoursE[1])
        else:
            low_x, low_y = contourEdge("T", contoursE[2])

        # Build aponeurosis arrays exactly like do_calculations_video.py
        new_X, new_Y_UA, new_Y_LA, segs_upper, segs_lower = build_apo_from_edges(
            upp_x=upp_x,
            upp_y=upp_y,
            low_x=low_x,
            low_y=low_y,
            w=w,
            smooth_win=81,
            smooth_poly=2,
        )

        # Thickness: central third of valid overlap
        valid = np.isfinite(new_Y_UA) & np.isfinite(new_Y_LA)

        if np.count_nonzero(valid) >= 30:
            idx = np.where(valid)[0]
            s = idx[int(0.4 * len(idx))]
            e = idx[int(0.6 * len(idx))]
            midthick = float(np.nanmean(np.abs(new_Y_LA[s:e] - new_Y_UA[s:e])))
        else:
            midthick = float("nan")

        # Store thickness region for plotting
        thickness_region_x = None
        thickness_region_ua = None
        thickness_region_la = None

        if np.count_nonzero(valid) >= 30:
            thickness_region_x = new_X[s:e]
            thickness_region_ua = new_Y_UA[s:e]
            thickness_region_la = new_Y_LA[s:e]

        # Region mask between extrapolated aponeuroses
        ex_mask = np.zeros((h, w), np.uint8)

        xs = new_X.astype(int)

        for xi, yu, yl in zip(xs, new_Y_UA, new_Y_LA):
            if np.isnan(yu) or np.isnan(yl):
                continue

            if xi < 0 or xi >= w:
                continue

            y0 = int(np.floor(min(yu, yl)))
            y1 = int(np.ceil(max(yu, yl)))

            if y1 <= y0:
                continue

            y0 = max(0, min(h - 1, y0))
            y1 = max(0, min(h - 1, y1))

            ex_mask[y0:y1, xi] = 255

        # For compatibility with the existing plotting code
        upp_y_new = np.interp(upp_x, new_X, new_Y_UA)
        low_y_new = np.interp(low_x, new_X, new_Y_LA)

        new_X_UA = new_X
        new_X_LA = new_X

        # Compute contours to identify fascicles/fascicle orientation
        _, threshF = cv2.threshold(fas_image, 0, 255, cv2.THRESH_BINARY)
        threshF = threshF.astype("uint8")
        contoursF, hierarchy = cv2.findContours(
            threshF, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Remove any contours that are very small
        maskF = np.zeros(threshF.shape, np.uint8)
        for contour in contoursF:  # Remove any contours that are very small
            if len(contour) > fasc_cont_thresh:
                cv2.drawContours(maskF, [contour], 0, 255, -1)

        # Only include fascicles within the region of the 2 aponeuroses
        mask_Fi = maskF & ex_mask
        contoursF2, hierarchy = cv2.findContours(
            mask_Fi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        contoursF3 = [i for i in contoursF2 if len(i) > fasc_cont_thresh]

        fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figure size for better clarity

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
            calib_dist=None,  # important: keep scaling outside for image function
        )

        data = fascicle_data.sort_values(by="x_low").reset_index(drop=True)
  
        # Sort data by x_low
        data = data.sort_values(by="x_low").reset_index(drop=True)

        # Display the image and fill the plot
        ax.imshow(
            img_copy,
            cmap="gray",
            aspect="auto",
            extent=[0, img_copy.shape[1], img_copy.shape[0], 0],
        )

        # Plot aponeuroses
        ax.plot(
            low_x,
            low_y_new,
            marker="p",
            color="blue",
            linewidth=2,
            alpha=0.8,
            label="Lower Aponeurosis",
        )
        ax.plot(
            upp_x,
            upp_y_new,
            marker="p",
            color="blue",
            linewidth=2,
            alpha=0.8,
            label="Upper Aponeurosis",
        )

        # Plot thickness region if available
        if thickness_region_x is not None and thickness_region_ua is not None and thickness_region_la is not None:
            ax.fill_between(
                thickness_region_x,
                thickness_region_ua,
                thickness_region_la,
                color="deepskyblue",
                alpha=0.25,
                label="Thickness Region",
            )
        
        # Plot fascicles with unique colors
        colormap = plt.cm.get_cmap("rainbow", len(data))
        handles = []  # For legend
        labels = []  # For legend
        for index, row in data.iterrows():
            color = colormap(index)
            (line,) = ax.plot(
                row["coordsX"],
                row["coordsY"],
                color=color,
                alpha=0.8,
                linewidth=2,
                label=f"Fascicle {row['x_low']}",
            )
            handles.append(line)
            labels.append(f"Fascicle {index}")

        # get fascicle length & pennation from dataframe
        fasc_l = np.asarray(data["fasc_l"], dtype=float)
        pennation = np.asarray(data["penn_a"], dtype=float)

        unit = "pix"

        if calib_dist:
            scale = calib_dist / int(spacing)
            fasc_l = fasc_l / scale
            midthick = midthick / scale
            unit = "mm"

        fasc_l = fasc_l.tolist()
        pennation = pennation.tolist()

        unit = "pix"
        # scale data
        if calib_dist:
            fasc_l = fasc_l / (calib_dist / int(spacing))
            midthick = midthick / (calib_dist / int(spacing))
            unit = "mm"

        # Add annotations
        xplot, yplot = 50, img_copy.shape[0] - 150
        ax.text(
            xplot,
            yplot,
            f"Median Fascicle Length: {np.median(fasc_l):.2f} {unit}",
            fontsize=12,
            color="white",
        )
        ax.text(
            xplot,
            yplot + 30,
            f"Median Pennation Angle: {np.median(pennation):.1f}°",
            fontsize=12,
            color="white",
        )
        ax.text(
            xplot,
            yplot + 60,
            f"Thickness at Centre: {midthick:.1f} {unit}",
            fontsize=12,
            color="white",
        )

        # Remove grid and ticks for a cleaner look
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

        # Add the legend with sorted handles and labels
        ax.legend(handles, labels, loc="upper right", fontsize=10)
        plt.tight_layout()  # Adjust layout for text and plot

        if image_callback:
            fig_copy = copy.deepcopy(fig)
            image_callback(fig_copy)

        return (
            fasc_l,
            pennation,
            data["x_low"].tolist(),
            data["x_high"].tolist(),
            midthick,
            fig,
        )

    else:

        return None, None, None, None, None, None
