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

        upp_y_new = savgol_filter(upp_y, 81, 2)  # window size 51, polynomial 3
        low_y_new = savgol_filter(low_y, 81, 2)

        # Make a binary mask to only include fascicles within the region
        # between the 2 aponeuroses
        ex_mask = np.zeros(thresh.shape, np.uint8)
        ex_1 = 0
        ex_2 = np.minimum(len(low_x), len(upp_x))

        for ii in range(ex_1, ex_2):
            ymin = int(np.floor(upp_y_new[ii]))
            ymax = int(np.ceil(low_y_new[ii]))

            ex_mask[:ymin, ii] = 0
            ex_mask[ymax:, ii] = 0
            ex_mask[ymin:ymax, ii] = 255

        # Calculate slope of central portion of each aponeurosis & use this to
        # compute muscle thickness
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
                    (upp_x[upp_ind], upp_y_new[upp_ind]), (low_x[val], low_y_new[val])
                )
                if dist < mindist:
                    mindist = dist

        # Compute functions to approximate the shape of the aponeuroses
        zUA = np.polyfit(upp_x, upp_y_new, 1)
        g = np.poly1d(zUA)
        zLA = np.polyfit(low_x, low_y_new, 1)
        h = np.poly1d(zLA)

        mid = (low_x[-1] - low_x[0]) / 2 + low_x[0]  # Find middle
        x1 = np.linspace(
            low_x[0] - 700, low_x[-1] + 700, 10000
        )  # Extrapolate polynomial fits to either side of the mid-point
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

        for contour in contoursF2:
            x, y = contourEdge("B", contour)
            if len(x) == 0:
                continue
            z = np.polyfit(np.array(x), np.array(y), 1)
            f = np.poly1d(z)
            newX = np.linspace(
                -400, w + 400, 5000
            )  # Extrapolate x,y data using f function
            newY = f(newX)

            # Find intersection between each fascicle and the aponeuroses.
            diffU = newY - new_Y_UA  # Find intersections
            locU = np.where(diffU == min(diffU, key=abs))[0]
            diffL = newY - new_Y_LA
            locL = np.where(diffL == min(diffL, key=abs))[0]
            # Get coordinates of fascicle between the two aponeuroses
            coordsX = newX[int(locL) : int(locU)]

            coordsY = newY[
                int(locL) : int(locU)
            ]  # These are the coordinates of the fascicles between the two aponeuroses

            # Get angle of aponeurosis in region close to fascicle intersection
            if locL >= 4950:
                Apoangle = int(
                    np.arctan(
                        (new_Y_LA[locL - 50] - new_Y_LA[locL - 50])
                        / (new_X_LA[locL] - new_X_LA[locL - 50])
                    )
                    * 180
                    / np.pi
                )
            else:
                Apoangle = int(
                    np.arctan(
                        (new_Y_LA[locL] - new_Y_LA[locL + 50])
                        / (new_X_LA[locL + 50] - new_X_LA[locL])
                    )
                    * 180
                    / np.pi
                )  # Angle relative to horizontal
            Apoangle = 90.0 + abs(Apoangle)

            # Don't include fascicles that are completely outside of the FoV
            # those that don't pass through central 1/3 of the image
            if (
                np.sum(coordsX) > 0
                and coordsX[-1] > 0
                and coordsX[0] < np.maximum(upp_x[-1], low_x[-1])
                and Apoangle != float("nan")
            ):
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

                if (
                    ActualAng <= max_pennation and ActualAng >= min_pennation
                ):  # Don't include 'fascicles' beyond a range of PA
                    length1 = np.sqrt(
                        (newX[locU] - newX[locL]) ** 2 + (y_UA[locU] - y_LA[locL]) ** 2
                    )
                    fascicle_data_temp = pd.DataFrame(
                        {
                            "x_low": [coordsX[0].astype("int32")],
                            "x_high": [coordsX[-1].astype("int32")],
                            "y_low": [coordsY[0].astype("int32")],
                            "y_high": [coordsY[-1].astype("int32")],
                            "coordsX": [coordsX],
                            "coordsY": [coordsY],
                            "fasc_l": [length1[0]],
                            "penn_a": Apoangle - FascAng,
                        }
                    )
                    fascicle_data = pd.concat(
                        [fascicle_data, fascicle_data_temp], ignore_index=True
                    )

        # Filter out fascicles that intersect with their right neighbors
        if filter_fasc == 1:
            data = filter_fascicles(fascicle_data)
        else:
            data = fascicle_data

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

        # Store the results for each frame and normalise using scale factor
        # (if calibration was done above)
        try:
            midthick = mindist[0]  # Muscle thickness
        except:
            midthick = mindist

        # get fascicle length & pennation from dataframe
        fasc_l = list(data["fasc_l"])
        pennation = list(data["penn_a"])

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
