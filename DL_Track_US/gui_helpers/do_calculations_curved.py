"""
Description
-----------
This module contains functions for three different approaches to calculate
muscle architectural parameters accounted for fascicle curvature. The 
parameters include muscle thickness, pennation angle and fascicle length.
All calculations are based on a fascicle mask and aponeurses mask.
Fascicle fragments are connected and extrapolated and the intersection points
with the extrapolated aponeuroses are determined. The architectural parameters
are calculated and the results are plotted. Additionally, it is possible to
calculate an orientation map of the fascicles based on the fascicle mask.
This module is specifically designed for single image analysis.

Functions scope
---------------
curve_polyfitting
    Function to calculate the fascicle length and pennation angle accounted
    for curvature following a second order polynomial fitting approach.
curve_connect
    Function to calculate the fascicle length and pennation angle accounted
    for curvature following a approach of connecting fascicles.
orientation_map
    Function to calculate an orientation map based on the fascicle mask.
doCalculations_curved
    Function to compute muscle architectural parameters accounted for fascicle curvature.

Notes
-----
Additional information can be found at the respective functions documentations.
"""

import math
import time

import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import orientationpy
import pandas as pd
import tensorflow as tf
from DL_Track_US.gui_helpers.curved_fascicles_functions import *
from keras.models import load_model
from scipy.interpolate import Rbf
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from skimage.morphology import skeletonize
from skimage.transform import resize


def curve_polyfitting(
    contours_sorted: list,
    ex_x_LA: list,
    ex_y_LA: list,
    ex_x_UA: list,
    ex_y_UA: list,
    original_image: np.ndarray,
    parameters: dict,
    filter_fasc: bool,
):
    """
    Function to calculate the fascicle length and pennation angle accounted for curvature following a second order polynomial fitting approach

    This function identifies individual fascicle contours and connects them if they are likely part of the same fascicle. A second-order polynomial curve is fitted through these contours; if the curvature exceeds a specified range, a linear fit is used instead. By knowing the positions of the aponeuroses, the intersection points between the fascicles and the lower and upper aponeuroses can be determined. Using these intersection points, the fascicle length and pennation angle are calculated.

     Parameters
     ----------
     contours_sorted : list
         List containing all (x,y)-coordinates of each detected contour
     ex_x_LA : list
         List containing all x-values of the extrapolated lower aponeurosis
     ex_y_LA: list
         List containing all y-values of the extrapolated lower aponeurosis
     ex_x_UA : list
         List containing all x-values of the extrapolated upper aponeurosis
     ex_y_UA : list
         List containing all y-values of the extrapolated upper aponeurosis
     original_image : np.ndarray
         Ultrasound image to be analysed
     parameters : dict
         Dictionary variable containing analysis parameters.
         These include apo_length_threshold, apo_length_thresh, fasc_cont_thresh, min_width, max_pennation,min_pennation, tolerance, tolerance_to_apo, coeff_limit
     filter_fasc : bool
         If True, fascicles will be filtered so that no crossings are included.
         This may reduce number of totally detected fascicles.

     Returns
     -------
     fascicle_length : list
         List variable containing the estimated fascicle lengths
         based on the segmented fascicle fragments in pixel units
         as float.
     pennation_angle : list
         List variable containing the estimated pennation angles
         based on the segmented fascicle fragments and aponeuroses
         as float.
     fig : matplot.figure
         Figure including the input ultrasound image, the segmented aponeuroses and
         the found fascicles extrapolated between the two aponeuroses.

    Example
    -------
    >>> curve_polyfitting(contours_sorted=[array([ 5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36], dtype=int32), array([166, 166, 166, 165, 165, 165, 164, 164, 164, 163, 163, 163, 162, 162, 162, 161, 161, 161, 160, 160, 160, 159, 159, 159, 158, 158, 158, 157, 157, 157, 156, 156], dtype=int32), ...], ex_x_LA=[-256.0, -255.79515903180635, -255.59031806361273, -255.38547709541908, -255.18063612722545, -254.9757951590318, ...], ex_y_LA=[203.6459743268554, 203.64809836232556, 203.65022013210233, 203.6523396361857, ...], ex_x_UA=[-256.0, -255.79515903180635, -255.59031806361273, -255.38547709541908, -255.18063612722545, ...], ex_y_UA=[45.83649948451378, 45.829729965913046, 45.82296488688939, 45.81620424744281, 45.80944804757331, ...], original_image=array([[[160, 160, 160],[159, 159, 159],[158, 158, 158],...[158, 158, 158],[147, 147, 147],[  1,   1,   1]],...,[[  0,   0,   0],[  0,   0,   0],[  0,   0,   0],...,[  4,   4,   4],[  3,   3,   3],[  3,   3,   3]]], dtype=uint8), parameters={apo_length_thresh=600, fasc_cont_thresh=5, min_width=60, max_pennation=40,min_pennation=5, tolerance=10, tolerance_to_apo=100}, filter_fascicles=True)
    """

    # Set parameters
    tolerance = int(parameters["tolerance"])
    tolerance_to_apo = int(parameters["tolerance_to_apo"])
    max_pennation = int(parameters["max_pennation"])
    min_pennation = int(parameters["min_pennation"])
    coeff_limit = 0.000583

    # get upper edge of each contour
    contours_sorted_x = []
    contours_sorted_y = []
    for i in range(len(contours_sorted)):
        contours_sorted[i][0], contours_sorted[i][1] = adapted_contourEdge(
            "B", contours_sorted[i]
        )
        contours_sorted_x.append(contours_sorted[i][0])
        contours_sorted_y.append(contours_sorted[i][1])

    # initialize important variables
    label = {x: False for x in range(len(contours_sorted))}
    number_contours = []
    all_fascicles_x = []
    all_fascicles_y = []
    width = original_image.shape[1]
    mid = width / 2
    LA_curve = list(zip(ex_x_LA, ex_y_LA))
    UA_curve = list(zip(ex_x_UA, ex_y_UA))

    fascicle_data = pd.DataFrame(
        columns=[
            "number_contours",
            "linear_fit",
            "coordsX",
            "coordsY",
            "coordsXY",
            "locU",
            "locL",
        ]
    )

    # calculate merged fascicle edges
    for i in range(len(contours_sorted)):

        if label[i] is False and len(contours_sorted_x[i]) > 1:

            (
                ex_current_fascicle_x,
                ex_current_fascicle_y,
                linear_fit,
                inner_number_contours,
            ) = find_complete_fascicle(
                i,
                contours_sorted_x,
                contours_sorted_y,
                contours_sorted,
                label,
                mid,
                width,
                tolerance,
                coeff_limit,
            )

            # Find intersection between fascicle and aponeuroses
            diffU = ex_current_fascicle_y - ex_y_UA
            diffU_short = diffU[0:4000]
            locU = np.where(diffU == min(diffU_short, key=abs))[0]
            if locU == 3999:
                locU = np.where(diffU == min(diffU, key=abs))[0]

            diffL = ex_current_fascicle_y - ex_y_LA
            locL = np.where(diffL == min(diffL, key=abs))[0]

            # Get coordinates of fascicle between the two aponeuroses
            coordsX = ex_current_fascicle_x[int(locL) : int(locU)]
            coordsY = ex_current_fascicle_y[int(locL) : int(locU)]
            coordsXY = list(zip(coordsX, coordsY))

            # only include fascicles that have intersection points with both aponeuroses
            fas_curve = list(zip(ex_current_fascicle_x, ex_current_fascicle_y))

            if do_curves_intersect(fas_curve, LA_curve) and do_curves_intersect(
                fas_curve, UA_curve
            ):

                all_fascicles_x.append(
                    ex_current_fascicle_x
                )  # store all points of fascicle, beyond apos
                all_fascicles_y.append(ex_current_fascicle_y)
                number_contours.append(inner_number_contours)

                fascicle_data_temp = pd.DataFrame(
                    {
                        "number_contours": [inner_number_contours],
                        "linear_fit": linear_fit,
                        "coordsX": [coordsX],
                        "coordsY": [coordsY],
                        "coordsXY": [coordsXY],
                        "locU": [locU],
                        "locL": [locL],
                    }
                )

                fascicle_data = pd.concat(
                    [fascicle_data, fascicle_data_temp], ignore_index=True
                )

    # filter overlapping fascicles
    if filter_fasc == 1:
        data = adapted_filter_fascicles(fascicle_data, tolerance_to_apo)
    else:
        data = fascicle_data

    all_coordsX = list(data["coordsX"])
    all_coordsY = list(data["coordsY"])
    all_locU = list(data["locU"])
    all_locL = list(data["locL"])
    data["fascicle_length"] = np.nan
    data["pennation_angle"] = np.nan

    for i in range(len(all_coordsX)):

        if len(all_coordsX[i]) > 0:
            # calculate length of fascicle
            x = all_coordsX[i]
            y = all_coordsY[i]

            dx = np.diff(x)
            dy = np.diff(y)

            segment_lengths = np.sqrt(dx**2 + dy**2)
            curve_length = np.sum(segment_lengths)

            # calculate pennation angle of fascicle
            apoangle = np.arctan(
                (ex_y_LA[all_locL[i]] - ex_y_LA[all_locL[i] + 50])
                / (ex_x_LA[all_locL[i] + 50] - ex_x_LA[all_locL[i]])
            ) * (180 / np.pi)
            fasangle = np.arctan(
                (all_coordsY[i][0] - all_coordsY[i][50])
                / (ex_x_LA[all_locL[i] + 50] - ex_x_LA[all_locL[i]])
            ) * (180 / np.pi)
            penangle = fasangle - apoangle

            data.iloc[i, data.columns.get_loc("fascicle_length")] = curve_length

            if (
                penangle <= max_pennation and penangle >= min_pennation
            ):  # Don't include 'fascicles' beyond a range of PA
                data.iloc[i, data.columns.get_loc("pennation_angle")] = penangle

    print(data)

    # plot filtered curves between detected fascicles between the two aponeuroses
    fig = plt.figure()
    colormap = plt.get_cmap("rainbow", len(all_coordsX))
    number_contours = list(data["number_contours"])  # contours after filtering

    for i in range(len(all_coordsX)):

        color = colormap(i)
        x = all_coordsX[i]
        y = all_coordsY[i]
        points = np.zeros(2 * len(number_contours[i]))
        points_arrays = [[] for _ in range(2 * len(number_contours[i]) + 2)]

        for j in range(len(number_contours[i])):
            points[2 * j] = contours_sorted_x[number_contours[i][j]][0]
            points[2 * j + 1] = contours_sorted_x[number_contours[i][j]][-1]

        if len(number_contours[i]) == 1:
            points_arrays[0] = x[x <= points[0]]
            points_arrays[1] = y[x <= points[0]]
            plt.plot(points_arrays[0], points_arrays[1], color=color, alpha=0.4)
            points_arrays[-2] = x[x >= points[-1]]
            points_arrays[-1] = y[x >= points[-1]]
            plt.plot(points_arrays[-2], points_arrays[-1], color=color, alpha=0.4)
            plt.plot(
                contours_sorted_x[number_contours[i][0]],
                contours_sorted_y[number_contours[i][0]],
                color="gold",
                alpha=0.6,
            )
        else:
            for j in range(len(number_contours[i])):
                if j == 0:
                    points_arrays[0] = x[x <= points[0]]
                    points_arrays[1] = y[x <= points[0]]
                    plt.plot(points_arrays[0], points_arrays[1], color=color, alpha=0.4)
                elif j == len(number_contours[i]) - 1:
                    points_arrays[-4] = x[(x >= points[-3]) & (x <= points[-2])]
                    points_arrays[-3] = y[(x >= points[-3]) & (x <= points[-2])]
                    plt.plot(
                        points_arrays[-4], points_arrays[-3], color=color, alpha=0.4
                    )
                    points_arrays[-2] = x[x >= points[-1]]
                    points_arrays[-1] = y[x >= points[-1]]
                    plt.plot(
                        points_arrays[-2], points_arrays[-1], color=color, alpha=0.4
                    )
                else:
                    points_arrays[2 * j] = x[
                        (x >= points[2 * j - 1]) & (x <= points[2 * j])
                    ]
                    points_arrays[2 * j + 1] = y[
                        (x >= points[2 * j - 1]) & (x <= points[2 * j])
                    ]
                    plt.plot(
                        points_arrays[2 * j],
                        points_arrays[2 * j + 1],
                        color=color,
                        alpha=0.4,
                    )

                plt.plot(
                    contours_sorted_x[number_contours[i][j]],
                    contours_sorted_y[number_contours[i][j]],
                    color="gold",
                    alpha=0.6,
                )

    plt.imshow(original_image)
    plt.plot(ex_x_LA, ex_y_LA, color="blue", alpha=0.5)
    plt.plot(ex_x_UA, ex_y_UA, color="blue", alpha=0.5)

    return data["fascicle_length"].tolist(), data["pennation_angle"].tolist(), fig


def curve_connect(
    contours_sorted: list,
    ex_x_LA: list,
    ex_y_LA: list,
    ex_x_UA: list,
    ex_y_UA: list,
    original_image: np.ndarray,
    parameters: dict,
    filter_fasc: bool,
    approach: str,
):
    """
    Function to calculate the fascicle length and pennation angle accounted for curvature following linear connection between fascicles

    This function identifies individual fascicle contours and connects them if they are likely part of the same fascicle. A second-order polynomial curve is fitted through these contours; if the curvature exceeds a specified range, a linear fit is used instead. This fit is solely for detecting the contours.
    curve_connect_linear: The first contour of the fascicle is extrapolated to determine its intersection point with the lower aponeurosis.
    curve_connect_poly: The lower aponeurosis and the first contour are connected using a second-order polynomial fit based on all detected contours.
    Following common path for both approaches: After the initial extrapolation, the first contour is added. A linear connection is made from the last point of the current contour to the first point of the next contour, and this next contour is added. This process continues until the final contour is reached. The final contour is then used for a linear extrapolation to determine the intersection point with the upper aponeurosis.
    Adding all these parts together, the function calculates the fascicle length and pennation angle.

    Parameters
    ----------
    contours_sorted : list
        List containing all (x,y)-coordinates of each detected contour
    ex_x_LA : list
        List containing all x-values of the extrapolated lower aponeurosis
    ex_y_LA: list
        List containing all y-values of the extrapolated lower aponeurosis
    ex_x_UA : list
        List containing all x-values of the extrapolated upper aponeurosis
    ex_y_UA : list
        List containing all y-values of the extrapolated upper aponeurosis
    original_image : np.ndarray
        Ultrasound image to be analysed
    parameters : dict
        Dictionary variable containing analysis parameters.
        These include apo_length_threshold, apo_length_thresh, fasc_cont_thresh, min_width, max_pennation,min_pennation, tolerance, tolerance_to_apo, coeff_limit
    filter_fasc : bool
        If True, fascicles will be filtered so that no crossings are included.
        This may reduce number of totally detected fascicles.
    approach: str
        Can either be curve_connect_linear or curve_connect_poly. If curve_connect_linear is used, a linear extrapolation between the lower aponeurosis and the first fascicle contour is used. If curve_connect_poly is used, a seconde order polynomial extrapolation between the lower and aponeurosis and the first fascicle contour is used; if the curvature exceeds a specified range, a linear fit is used instead.

    Returns
    -------
    fascicle_length : list
        List variable containing the estimated fascicle lengths
        based on the segmented fascicle fragments in pixel units
        as float.
    pennation_angle : list
        List variable containing the estimated pennation angles
        based on the segmented fascicle fragments and aponeuroses
        as float.
    fig : matplot.figure
        Figure including the input ultrasound image, the segmented aponeuroses and
        the found fascicles extrapolated between the two aponeuroses.

    Example
    ------
    >>> curve_connect(contours_sorted=[array([ 5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36], dtype=int32), array([166, 166, 166, 165, 165, 165, 164, 164, 164, 163, 163, 163, 162, 162, 162, 161, 161, 161, 160, 160, 160, 159, 159, 159, 158, 158, 158, 157, 157, 157, 156, 156], dtype=int32), ...], ex_x_LA=[-256.0, -255.79515903180635, -255.59031806361273, -255.38547709541908, -255.18063612722545, -254.9757951590318, ...], ex_y_LA=[203.6459743268554, 203.64809836232556, 203.65022013210233, 203.6523396361857, ...], ex_x_UA=[-256.0, -255.79515903180635, -255.59031806361273, -255.38547709541908, -255.18063612722545, ...], ex_y_UA=[45.83649948451378, 45.829729965913046, 45.82296488688939, 45.81620424744281, 45.80944804757331, ...], original_image=array([[[160, 160, 160],[159, 159, 159],[158, 158, 158],...[158, 158, 158],[147, 147, 147],[  1,   1,   1]],...,[[  0,   0,   0],[  0,   0,   0],[  0,   0,   0],...,[  4,   4,   4],[  3,   3,   3],[  3,   3,   3]]], dtype=uint8), parameters={apo_length_thresh=600, fasc_cont_thresh=5, min_width=60, max_pennation=40,min_pennation=5, tolerance=10, tolerance_to_apo=100}, filter_fascicles=True, approach="curve_connect_linear")
    """

    # set parameteres
    tolerance = int(parameters["tolerance"])
    tolerance_to_apo = int(parameters["tolerance_to_apo"])
    max_pennation = int(parameters["max_pennation"])
    min_pennation = int(parameters["min_pennation"])
    coeff_limit = 0.000583

    # get upper edge of each contour
    contours_sorted_x = []
    contours_sorted_y = []
    for i in range(len(contours_sorted)):
        contours_sorted[i][0], contours_sorted[i][1] = adapted_contourEdge(
            "B", contours_sorted[i]
        )
        contours_sorted_x.append(contours_sorted[i][0])
        contours_sorted_y.append(contours_sorted[i][1])

    # initialize some important variables
    label = {x: False for x in range(len(contours_sorted))}
    coefficient_label = []
    number_contours = []
    all_fascicles_x = []
    all_fascicles_y = []
    width = original_image.shape[1]
    mid = width / 2
    LA_curve = list(zip(ex_x_LA, ex_y_LA))
    UA_curve = list(zip(ex_x_UA, ex_y_UA))

    fascicle_data = pd.DataFrame(
        columns=[
            "number_contours",
            "linear_fit",
            "coordsX",
            "coordsY",
            "coordsX_combined",
            "coordsY_combined",
            "coordsXY",
            "locU",
            "locL",
        ]
    )

    # calculate merged fascicle edges
    for i in range(len(contours_sorted)):

        if label[i] is False and len(contours_sorted_x[i]) > 1:

            (
                ex_current_fascicle_x,
                ex_current_fascicle_y,
                linear_fit,
                inner_number_contours,
            ) = find_complete_fascicle(
                i,
                contours_sorted_x,
                contours_sorted_y,
                contours_sorted,
                label,
                mid,
                width,
                tolerance,
                coeff_limit,
            )

            all_fascicles_x.append(ex_current_fascicle_x)
            all_fascicles_y.append(ex_current_fascicle_y)
            coefficient_label.append(linear_fit)
            number_contours.append(inner_number_contours)

            fascicle_data_temp = pd.DataFrame(
                {
                    "number_contours": [inner_number_contours],
                    "linear_fit": linear_fit,
                    "coordsX": None,
                    "coordsY": None,
                    "coordsX_combined": None,
                    "coordsY_combined": None,
                    "coordsXY": None,
                    "locU": None,
                    "locL": None,
                }
            )

            fascicle_data = pd.concat(
                [fascicle_data, fascicle_data_temp], ignore_index=True
            )

    number_contours = list(fascicle_data["number_contours"])

    for i in range(len(number_contours)):

        if approach == "curve_connect_linear":
            # calculate linear fit through first contour of fascicle, extrapolate over the complete image and compute intersection point with lower aponeurosis
            coefficients = np.polyfit(
                contours_sorted_x[number_contours[i][0]],
                contours_sorted_y[number_contours[i][0]],
                1,
            )
            g = np.poly1d(coefficients)
            ex_current_fascicle_x = np.linspace(mid - width, mid + width, 5000)
            ex_current_fascicle_y = g(ex_current_fascicle_x)

            fas_LA_curve = list(zip(ex_current_fascicle_x, ex_current_fascicle_y))
            fas_LA_intersection = do_curves_intersect(LA_curve, fas_LA_curve)

            # calculate intersection point with lower aponeurosis
            diffL = ex_current_fascicle_y - ex_y_LA
            locL = np.where(diffL == min(diffL, key=abs))[0]

            # find index of first item of first contour
            first_item = contours_sorted_x[number_contours[i][0]][0]
            differences = np.abs(ex_current_fascicle_x - first_item)
            index_first_item = np.argmin(differences)

            # get extrapolation from the intersection with the lower aponeurosis to the beginning of the first fascicle
            ex_current_fascicle_x = ex_current_fascicle_x[int(locL) : index_first_item]
            ex_current_fascicle_y = ex_current_fascicle_y[int(locL) : index_first_item]

        if approach == "curve_connect_poly":
            # calculate line from lower aponeurosis to first fascicle according to computed fit from before
            fas_LA_curve = list(zip(all_fascicles_x[i], all_fascicles_y[i]))
            fas_LA_intersection = do_curves_intersect(LA_curve, fas_LA_curve)

            # calculate intersection point with lower aponeurosis
            diffL = all_fascicles_y[i] - ex_y_LA
            locL = np.where(diffL == min(diffL, key=abs))[0]

            # find index of first item of first contour
            first_item = contours_sorted_x[number_contours[i][0]][0]
            differences = np.abs(all_fascicles_x[i] - first_item)
            index_first_item = np.argmin(differences)

            # get extrapolation from the intersection with the lower aponeurosis to the beginning of the first fascicle
            ex_current_fascicle_x = all_fascicles_x[i][int(locL) : index_first_item]
            ex_current_fascicle_y = all_fascicles_y[i][int(locL) : index_first_item]

        # convert to list, want list of sections in one list (list in list)
        coordsX = [list(ex_current_fascicle_x)]
        coordsY = [list(ex_current_fascicle_y)]

        # append first contour to list
        coordsX.append(contours_sorted_x[number_contours[i][0]])
        coordsY.append(contours_sorted_y[number_contours[i][0]])

        # append gap between contours and following contours to the list
        if len(number_contours[i]) > 1:
            for j in range(len(number_contours[i]) - 1):
                end_x = contours_sorted_x[number_contours[i][j]][-1]
                end_y = contours_sorted_y[number_contours[i][j]][-1]
                start_x = contours_sorted_x[number_contours[i][j + 1]][0]
                start_y = contours_sorted_y[number_contours[i][j + 1]][0]
                coordsX.append([end_x, start_x])
                coordsY.append([end_y, start_y])
                coordsX.append(contours_sorted_x[number_contours[i][j + 1]])
                coordsY.append(contours_sorted_y[number_contours[i][j + 1]])

        # calculate linear fit for last contour, extrapolate over complete image to get intersection point with upper aponeurosis
        coefficients = np.polyfit(
            contours_sorted_x[number_contours[i][-1]],
            contours_sorted_y[number_contours[i][-1]],
            1,
        )
        g = np.poly1d(coefficients)
        ex_current_fascicle_x_2 = np.linspace(mid - width, mid + width, 5000)
        ex_current_fascicle_y_2 = g(ex_current_fascicle_x_2)

        fas_UA_curve = list(zip(ex_current_fascicle_x_2, ex_current_fascicle_y_2))
        fas_UA_intersection = do_curves_intersect(UA_curve, fas_UA_curve)

        # calulate intersection point with upper aponeurosis
        diffU = ex_current_fascicle_y_2 - ex_y_UA
        locU = np.where(diffU == min(diffU, key=abs))[0]

        # find index of last item of last contour
        last_item = contours_sorted_x[number_contours[i][-1]][-1]
        differences_2 = np.abs(ex_current_fascicle_x_2 - last_item)
        index_last_item = np.argmin(differences_2)

        # get extrapolation from the end of the last fascicle to the upper aponeurosis
        ex_current_fascicle_x_2 = ex_current_fascicle_x_2[index_last_item : int(locU)]
        ex_current_fascicle_y_2 = ex_current_fascicle_y_2[index_last_item : int(locU)]

        # append to list
        coordsX.append(list(ex_current_fascicle_x_2))
        coordsY.append(list(ex_current_fascicle_y_2))

        # get new list in which the different lists are not separated
        coordsX_combined = []
        coordsX_combined = [item for sublist in coordsX for item in sublist]

        coordsY_combined = []
        coordsY_combined = [item for sublist in coordsY for item in sublist]

        coordsXY = list(zip(coordsX_combined, coordsY_combined))

        fascicle_data.at[i, "coordsX"] = (
            coordsX  # x-coordinates of all sections as list in list
        )
        fascicle_data.at[i, "coordsY"] = (
            coordsY  # y-coordinates of all sections as list in list
        )
        fascicle_data.at[i, "coordsX_combined"] = (
            coordsX_combined  # x-coordinates of all sections as one list
        )
        fascicle_data.at[i, "coordsY_combined"] = (
            coordsY_combined  # y-coordinates of all sections as one list
        )
        fascicle_data.at[i, "coordsXY"] = (
            coordsXY  # x- and y-coordinates of all sections as one list
        )
        fascicle_data.at[i, "locU"] = locU
        fascicle_data.at[i, "locL"] = locL
        fascicle_data.at[i, "intersection_LA"] = fas_LA_intersection
        fascicle_data.at[i, "intersection_UA"] = fas_UA_intersection

    fascicle_data = fascicle_data[fascicle_data["intersection_LA"]].drop(
        columns="intersection_LA"
    )  # .reset_index()
    fascicle_data = fascicle_data[fascicle_data["intersection_UA"]].drop(
        columns="intersection_UA"
    )  # .reset_index()
    fascicle_data = fascicle_data.reset_index(drop=True)

    # filter overlapping fascicles
    if filter_fasc == 1:
        data = adapted_filter_fascicles(fascicle_data, tolerance_to_apo)
    else:
        data = fascicle_data

    all_coordsX = list(data["coordsX"])
    all_coordsY = list(data["coordsY"])
    all_locU = list(data["locU"])
    all_locL = list(data["locL"])
    data["fascicle_length"] = np.nan
    data["pennation_angle"] = np.nan

    for i in range(len(all_coordsX)):

        # calculate length of fascicle
        curve_length_total = 0

        for j in range(len(all_coordsX[i])):

            x = all_coordsX[i][j]
            y = all_coordsY[i][j]

            dx = np.diff(x)
            dy = np.diff(y)

            segment_lengths = np.sqrt(dx**2 + dy**2)
            curve_length = np.sum(segment_lengths)
            curve_length_total += curve_length

        # calculate pennation angle
        if len(all_coordsX[i][0]) > 1:
            apoangle = np.arctan(
                (ex_y_LA[all_locL[i]] - ex_y_LA[all_locL[i] + 50])
                / (ex_x_LA[all_locL[i] + 50] - ex_x_LA[all_locL[i]])
            ) * (180 / np.pi)
            fasangle = np.arctan(
                (all_coordsY[i][0][0] - all_coordsY[i][0][-1])
                / (all_coordsX[i][0][-1] - all_coordsX[i][0][0])
            ) * (180 / np.pi)
            penangle = fasangle - apoangle
        else:
            apoangle = np.arctan(
                (ex_y_LA[all_locL[i]] - ex_y_LA[all_locL[i] + 50])
                / (ex_x_LA[all_locL[i] + 50] - ex_x_LA[all_locL[i]])
            ) * (180 / np.pi)
            fasangle = np.arctan(
                (all_coordsY[i][1][0] - all_coordsY[i][1][-1])
                / (all_coordsX[i][1][-1] - all_coordsX[i][1][0])
            ) * (180 / np.pi)
            penangle = fasangle - apoangle

        data.iloc[i, data.columns.get_loc("fascicle_length")] = curve_length_total

        if (
            penangle <= max_pennation and penangle >= min_pennation
        ):  # Don't include 'fascicles' beyond a range of PA
            data.iloc[i, data.columns.get_loc("pennation_angle")] = penangle

    print(data)

    fig = plt.figure()
    colormap = plt.get_cmap("rainbow", len(all_coordsX))

    plt.imshow(original_image)
    for i in range(len(all_coordsX)):
        color = colormap(i)
        for j in range(len(all_coordsX[i])):
            if j == 0:
                plt.plot(all_coordsX[i][j], all_coordsY[i][j], color=color, alpha=0.4)
            if j % 2 == 1:
                plt.plot(all_coordsX[i][j], all_coordsY[i][j], color="gold", alpha=0.6)
            else:
                plt.plot(all_coordsX[i][j], all_coordsY[i][j], color=color, alpha=0.4)
    plt.plot(ex_x_LA, ex_y_LA, color="blue", alpha=0.5)
    plt.plot(ex_x_UA, ex_y_UA, color="blue", alpha=0.5)

    return data["fascicle_length"].tolist(), data["pennation_angle"].tolist(), fig


def orientation_map(
    fas_image: np.ndarray,
    apo_image: np.ndarray,
    g: np.poly1d,
    h: np.poly1d,
):
    """Function to calculate an orientation map based on the fascicle mask

    The function calculates the orientations of fascicles based on the fascicle mask using the OrientationPy package. It then uses linear inter- and extrapolation to determine the orientation of all points in the region between the two aponeuroses using the Rbf package. Finally, the resulting vectors are smoothed with a Gaussian filter. To approximate the median angle, the image is divided into six sections: two horizontally and three vertically. The median angle is then calculated for each of these sections. In the plot only the median for the part in the lower half and middle of the image is displayed.

    Parameters
    ----------
    original_image : np.ndarray
        Ultrasound image to be analysed
    fas_image : np.ndarray
        Mask of fascicles
    apo_image: np.ndarray
        Mask of aponeuroses
    g : np.poly1d
        Containing coefficients to calculate second order polynomial fit for upper aponeurosis
    h : np.poly1d
        Containing coefficients to calculate second order polynomial fit for lower aponeurosis

    Returns
    -------
    split_angles_deg_median : list
        List variable containing the estimated pennation angles for the six parts of the image
    fig : matplot.figure
        Figure showing the estimated slope at different points in the region between the two aponeuroses as a heat map

    Example
    ------
    >>> orientation_map(original_image=array([[[160, 160, 160],[159, 159, 159],[158, 158, 158],...[158, 158, 158],[147, 147, 147],[  1,   1,   1]],...,[[  0,   0,   0],[  0,   0,   0],[  0,   0,   0],...,[  4,   4,   4],[  3,   3,   3],[  3,   3,   3]]], dtype=uint8), fas_image = array([[0, 0, 0, ..., 0, 0, 0],[0, 0, 0, ..., 0, 0, 0],[0, 0, 0, ..., 0, 0, 0],...,[0, 0, 0, ..., 0, 0, 0],[0, 0, 0, ..., 0, 0, 0],[0, 0, 0, ..., 0, 0, 0]], dtype=uint8), apo_image = array([[[0, 0, 0],[0, 0, 0],[0, 0, 0],...,[0, 0, 0],[0, 0, 0],[0, 0, 0]]], dtype=uint8), g=poly1d([ 0.   , -0.006, 40.841]), h=poly1d([ -0.   ,  -0.003, 204.533]))
    """

    width = apo_image.shape[1]

    ex_x_UA = np.linspace(0, width - 1, width)  # Extrapolate x,y data using f function
    # new_X_UA = np.linspace(-200, 800, 5000)
    ex_y_UA = g(ex_x_UA)
    # new_X_LA = np.linspace(-200, 800, 5000)
    ex_x_LA = np.linspace(0, width - 1, width)  # Extrapolate x,y data using f function
    ex_y_LA = h(ex_x_LA)

    image_rgb = cv2.cvtColor(fas_image, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    for n, mode in enumerate(["finite_difference", "gaussian", "splines"]):
        Gy, Gx = orientationpy.computeGradient(image_gray, mode=mode)

    structureTensor = orientationpy.computeStructureTensor([Gy, Gx], sigma=2)

    orientations = orientationpy.computeOrientation(
        structureTensor, computeEnergy=True, computeCoherency=True
    )

    # The coherency measures how strongly aligned the image is locally
    orientations["coherency"][fas_image == 0] = 0

    boxSizePixels = 7

    structureTensorBoxes = orientationpy.computeStructureTensorBoxes(
        [Gy, Gx],
        [boxSizePixels, boxSizePixels],
    )

    # The structure tensor in boxes is passed to the same function to compute
    # The orientation
    orientationsBoxes = orientationpy.computeOrientation(
        structureTensorBoxes,
        mode="fibre",
        computeEnergy=True,
        computeCoherency=True,
    )

    # We normalise the energy, to be able to hide arrows in the subsequent quiver plot
    orientationsBoxes["energy"] /= orientationsBoxes["energy"].max()

    # Compute box centres
    boxCentresY = (
        np.arange(orientationsBoxes["theta"].shape[0]) * boxSizePixels
        + boxSizePixels // 2
    )
    boxCentresX = (
        np.arange(orientationsBoxes["theta"].shape[1]) * boxSizePixels
        + boxSizePixels // 2
    )

    # get grid points where vectors originate
    boxCentres_grid = np.meshgrid(boxCentresX, boxCentresY)
    boxCentres_grid_X = boxCentres_grid[0]
    boxCentres_grid_Y = boxCentres_grid[1]
    boxCentres_grid_X = [item for sublist in boxCentres_grid_X for item in sublist]
    boxCentres_grid_Y = [item for sublist in boxCentres_grid_Y for item in sublist]

    # get number of points along the x- and y-axis (size of grid)
    # attention with the dimensions, before a system (x,y) like normal coordinates was used,
    # in an array the convention is the other way around array[row (y in image), column (x in image)]
    size_y = boxCentresX.shape[0]
    size_x = boxCentresY.shape[0]

    # Compute X and Y components of the vector
    boxVectorsYX = orientationpy.anglesToVectors(orientationsBoxes)

    # Vectors with low energy reset
    boxVectorsYX[:, orientationsBoxes["energy"] < 0.05] = 0.0

    # only allow vectors which have an angle between 50° and 10°
    boxVectorsYX[:, orientationsBoxes["theta"] > 50] = 0.0
    boxVectorsYX[:, orientationsBoxes["theta"] < 10] = 0.0

    # boxVectorsYX = [list(zip(x, y)) for x, y in zip(boxVectorsYX[0], boxVectorsYX[1])]
    boxVectorsX = [item for sublist in boxVectorsYX[1] for item in sublist]
    boxVectorsY = [item for sublist in boxVectorsYX[0] for item in sublist]

    # create mask representing valid vectors (not 0)
    mask = (np.array(boxVectorsX) != 0) & (np.array(boxVectorsY) != 0)

    # find grid points which are the origin of valid vectors
    valid_boxCentres_grid_X = np.array(boxCentres_grid_X)[mask]
    valid_boxCentres_grid_Y = np.array(boxCentres_grid_Y)[mask]

    # get valid vectors separate
    valid_boxVectorsX = np.array(boxVectorsX)[mask]
    valid_boxVectorsY = np.array(boxVectorsY)[mask]

    # interpolation and extrapolation valid vectors with rbf along the grid
    grid_x_rbf = Rbf(
        valid_boxCentres_grid_X,
        valid_boxCentres_grid_Y,
        valid_boxVectorsX,
        function="linear",
    )
    grid_y_rbf = Rbf(
        valid_boxCentres_grid_X,
        valid_boxCentres_grid_Y,
        valid_boxVectorsY,
        function="linear",
    )

    # extra- and interpolated values
    di_x = grid_x_rbf(boxCentres_grid_X, boxCentres_grid_Y)
    di_y = grid_y_rbf(boxCentres_grid_X, boxCentres_grid_Y)

    # smoothened extra- and intrapolated values with gaussian filter
    di_x_smooth = gaussian_filter1d(di_x, sigma=1)
    di_y_smooth = gaussian_filter1d(di_y, sigma=1)

    # Make a binary mask to only include fascicles within the region
    # between the 2 aponeuroses
    di_x_smooth_masked = np.array(di_x_smooth).reshape(size_x, size_y)
    di_y_smooth_masked = np.array(di_y_smooth).reshape(size_x, size_y)

    # define new mask which contains only the points of the grid which are between the two aponeuroses
    ex_mask = np.zeros((size_x, size_y), dtype=bool)

    # mixture of both conventions!
    for ii in range(size_y):
        coord = boxCentresX[ii]
        ymin = int(ex_y_UA[coord])
        ymax = int(ex_y_LA[coord])

        for jj in range(size_x):
            if boxCentresY[jj] < ymin:
                ex_mask[jj][ii] = False
            elif boxCentresY[jj] > ymax:
                ex_mask[jj][ii] = False
            else:
                ex_mask[jj][ii] = True

    # apply mask to smoothened data, 2D data
    di_x_masked_2 = di_x_smooth_masked * ex_mask.astype(int)
    di_y_masked_2 = di_y_smooth_masked * ex_mask.astype(int)

    # flatten data to 1D, is needed in this format for the quiver plot
    di_x_masked_1 = di_x_masked_2.flatten()
    di_y_masked_1 = di_y_masked_2.flatten()

    # initialize variables to store slope
    slope = np.zeros_like(di_x)
    slope_without_zeros = []

    # calculate slope for the vectors in the region between the aponeuroses
    for i in range(len(di_x)):
        if di_x_masked_1[i] != 0 and di_y_masked_1[i] != 0:
            slope[i] = (-1) * di_y_masked_1[i] / di_x_masked_1[i]
            slope_without_zeros.append(slope[i])

    slope = np.array(slope).reshape(size_x, size_y)

    # get rows which contain information for the region between the two aponeuroses
    slope_non_zero_rows = slope[~np.all(slope == 0, axis=1)]

    # increase size of slope array for plotting
    slope = np.repeat(np.repeat(slope, boxSizePixels, axis=0), boxSizePixels, axis=1)

    # calculate mean and median slope of the complete region between the aponeuroses
    slope_mean = np.mean(slope_without_zeros)
    slope_median = np.median(slope_without_zeros)

    # calculate mean and median angle of the complete region between the aponeuroses
    angle_rad_mean = math.atan(slope_mean)
    angle_deg_mean = math.degrees(angle_rad_mean)
    angle_rad_median = math.atan(slope_median)
    angle_deg_median = math.degrees(angle_rad_median)

    # split image to calculate mean and median angle for different parts
    # Step 1: Split the array in the middle horizontally
    middle_split = np.array_split(slope_non_zero_rows, 2, axis=0)

    # Step 2: Split each of the resulting arrays vertically into 3 parts
    split_arrays = [np.array_split(sub_array, 3, axis=1) for sub_array in middle_split]

    # Step 3: Flatten the list of lists into a single list
    split_arrays = [sub_array for sublist in split_arrays for sub_array in sublist]

    # calculate mean and median angle for each part

    split_angles_deg_mean = []
    split_angles_deg_median = []

    for i in range(len(split_arrays)):
        # apply mask in order that no 0 are in calculation (not part of region between aponeuroses)
        non_zero_mask = split_arrays[i] != 0
        non_zero_elements = split_arrays[i][non_zero_mask]

        mean_non_zero = np.mean(non_zero_elements)
        split_angle_rad_mean = math.atan(mean_non_zero)
        split_angle_deg_mean = math.degrees(split_angle_rad_mean)
        split_angles_deg_mean.append(split_angle_deg_mean)

        median_non_zero = np.median(non_zero_elements)
        split_angle_rad_median = math.atan(median_non_zero)
        split_angle_deg_median = math.degrees(split_angle_rad_median)
        split_angles_deg_median.append(split_angle_deg_median)

    norm = mcolors.Normalize(vmin=np.min(slope), vmax=np.max(slope))
    xplot = 75
    yplot = 400

    # plot heat map of slopes for region between the two aponeuroses
    fig = plt.figure()
    plt.imshow(slope, cmap="viridis", norm=norm, interpolation="none")
    plt.plot(ex_x_LA, ex_y_LA, color="white")
    plt.plot(ex_x_UA, ex_y_UA, color="white")
    plt.colorbar(label="Value")
    plt.title("Matrix Heatmap")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.text(
        xplot,
        yplot,
        (
            "Median pennation angle: "
            + str("%.1f" % split_angles_deg_median[4])
            + " deg"
        ),
        fontsize=10,
        color="white",
    )

    return None, split_angles_deg_median, fig


def doCalculations_curved(
    original_image: np.ndarray,
    img_copy: np.ndarray,
    h: int,
    w: int,
    model_apo,
    model_fasc,
    parameters: dict,
    filter_fasc: bool,
    calib_dist: bool,
    spacing: int,
    approach: str,
):
    """Function to compute muscle architectural parameters accounted for fascicle curvature

    The aponeuroses edges are computed and the fascicle contours are connected to form complete fascicles. Based on three different approaches the fascicle length and pennation angle get calculated. Furthermore, it is possible to calculate an orientation map showing the slope at different points in the region of interest.

    Returns none when not more than one aponeurosis contour or no fascicle contours are
    detected in the image.

    Parameters
    ----------
    original_image : np.ndarray
        Ultrasound image to be analysed
    img_copy : np.ndarray
        A copy of the input image.
    h : int
        Integer variable containing the height of the input image (original image).
    w : int
        Integer variable containing the width of the input image (original image).
    model_apo :
        Contains keras model for prediction of aponeuroses
    model_fasc :
        Contains keras model for prediction of fascicles
    parameters : dict
        Dictionary variable containing analysis parameters.
        These include apo_length_threshold, apo_length_thresh, fasc_cont_thresh, min_width, max_pennation,min_pennation, tolerance, tolerance_to_apo
    filter_fasc : bool
        If True, fascicles will be filtered so that no crossings are included.
        This may reduce number of totally detected fascicles.
    calib_dist : int
        Integer variable containing the distance between the two
        specified point in pixel units. This value was either computed
        automatically or manually. Must be non-negative. If "None", the
        values are outputted in pixel units.
    spacing : {10, 5, 15, 20}
        Integer variable containing the known distance in milimeter
        between the two placed points by the user or the scaling bars
        present in the image. This can be 5, 10, 15 or 20 millimeter.
        Must be non-negative and non-zero.
    approach: str
        Can either be curve_polyfitting, curve_connect_linear, curve_connect_poly or orientation_map.
        curve_polyfitting calculates the fascicle length and pennation angle according to a second order polynomial fitting (see documentation of function curve_polyfitting).
        curve_connect_linear and curve_connect_poly calculate the fascicle length and pennation angle according to a linear connection between the fascicles fascicles (see documentation of function curve_connect).
        orientation_map calculates an orientation map and gives an estimate for the median angle of the image (see documentation of function orientation_map)

    Returns
    -------
    fascicle_length : list
        List variable containing the estimated fascicle lengths
        based on the segmented fascicle fragments in pixel units
        as float. If calib_dist is specified, then the length is computed
        in centimeter.
    pennation_angle : list
        List variable containing the estimated pennation angles
        based on the segmented fascicle fragments and aponeuroses
        as float.
    midthick : float
        Float variable containing the estimated distance
        between the lower and upper aponeurosis in pixel units.
        If calib_dist is specified, then the distance is computed
        in centimeter.
    fig : matplot.figure
        Figure including the input ultrasound image, the segmented aponeuroses and
        the found fascicles extrapolated between the two aponeuroses.

    Notes
    -----
    For more detailed documentation, see the respective functions documentation.

    Example
    ------
    >>> doCalculations_curved(original_image=array([[[160, 160, 160],[159, 159, 159],[158, 158, 158],...[158, 158, 158],[147, 147, 147],[  1,   1,   1]],...,[[  0,   0,   0],[  0,   0,   0],[  0,   0,   0],...,[  4,   4,   4],[  3,   3,   3],[  3,   3,   3]]], dtype=uint8), fas_image = array([[0, 0, 0, ..., 0, 0, 0],[0, 0, 0, ..., 0, 0, 0],[0, 0, 0, ..., 0, 0, 0],...,[0, 0, 0, ..., 0, 0, 0],[0, 0, 0, ..., 0, 0, 0],[0, 0, 0, ..., 0, 0, 0]], dtype=uint8), apo_image = array([[[0, 0, 0],[0, 0, 0],[0, 0, 0],...,[0, 0, 0],[0, 0, 0],[0, 0, 0]]], dtype=uint8), parameters={apo_length_thresh=600, fasc_cont_thresh=5, min_width=60, max_pennation=40,min_pennation=5, tolerance=10, tolerance_to_apo=100}, filter_fascicles=True, calib_dist = None, spacing = 10, approach = "curve_polyfitting")
    """

    start_time = time.time()

    parameters = parameters

    apo_length_tresh = int(parameters["apo_length_thresh"])
    fasc_cont_thresh = int(parameters["fasc_cont_thresh"])
    min_width = int(parameters["min_width"])
    apo_threshold = float(parameters["apo_threshold"])
    fasc_threshold = float(
        parameters["fasc_threshold"]
    )  # TODO why are there no pennation thresholds?

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

    # crop all three images in order that they don't have a frame
    # original_image, fas_image, apo_image = crop(img_copy, fas_image, apo_image) #TODO first image is always misplaced

    # calculations for aponeuroses
    # apo_image_rgb = cv2.cvtColor(apo_image, cv2.COLOR_BGR2RGB)
    # apo_image_gray = cv2.cvtColor(apo_image_rgb, cv2.COLOR_RGB2GRAY)

    width = fas_image.shape[1]

    _, thresh = cv2.threshold(
        apo_image, 0, 255, cv2.THRESH_BINARY  # + cv2.THRESH_OTSU
    )  # TODO Why an additional OTSU? This is not done in the original code
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
        return None, None, None, None

    (contours, _) = sortContours(contours)  # Sort contours from top to bottom

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
        zUA = np.polyfit(upp_x, upp_y_new, 2)
        g = np.poly1d(zUA)
        zLA = np.polyfit(low_x, low_y_new, 2)
        h = np.poly1d(zLA)

        mid = (low_x[-1] - low_x[0]) / 2 + low_x[0]  # Find middle
        x1 = np.linspace(-200, 800, 5000)
        # x1 = np.linspace(
        # low_x[0] - 700, low_x[-1] + 700, 10000
        # )  # Extrapolate polynomial fits to either side of the mid-point
        y_UA = g(x1)
        y_LA = h(x1)

        mid = width / 2
        new_X_UA = np.linspace(
            mid - width, mid + width, 5000
        )  # Extrapolate x,y data using f function
        # new_X_UA = np.linspace(-200, 800, 5000)
        new_Y_UA = g(new_X_UA)
        # new_X_LA = np.linspace(-200, 800, 5000)
        new_X_LA = np.linspace(
            mid - width, mid + width, 5000
        )  # Extrapolate x,y data using f function
        new_Y_LA = h(new_X_LA)

        # calculations for fascicle mask
        # image_rgb = cv2.cvtColor(fas_image, cv2.COLOR_BGR2RGB)
        # image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        # define threshold and find contours around fascicles
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

        # convert contours into a list
        contours = list(contoursF3)

        # Convert each contour to a NumPy array, reshape, and sort
        for i in range(len(contours)):
            contour_array = np.array(contours[i])  # Convert to NumPy array
            if contour_array.shape[1] == 1 and contour_array.shape[2] == 2:
                reshaped_contour = contour_array.reshape(-1, 2)  # Reshape to (58, 2)
                sorted_contour = sorted(
                    reshaped_contour, key=lambda k: (k[0], k[1])
                )  # Sort by x and y
                contours[i] = sorted_contour  # Update the contour in the list
            else:
                print(
                    f"Contour {i} does not have the expected shape: {contour_array.shape}"
                )

        # Now, contours are sorted, and we can sort the list of contours based on the first point
        contours_sorted = sorted(
            contours,
            key=lambda k: (
                (k[0][0], -k[0][1]) if len(k) > 0 else (float("inf"), float("inf"))
            ),
        )

        # calculations depending on chosen approach
        if approach == "curve_polyfitting":
            fascicle_length, pennation_angle, fig = curve_polyfitting(
                contours_sorted,
                new_X_LA,
                new_Y_LA,
                new_X_UA,
                new_Y_UA,
                original_image,
                parameters,
                filter_fasc,
            )
        if approach == "curve_connect_linear" or approach == "curve_connect_poly":
            fascicle_length, pennation_angle, fig = curve_connect(
                contours_sorted,
                new_X_LA,
                new_Y_LA,
                new_X_UA,
                new_Y_UA,
                original_image,
                parameters,
                filter_fasc,
                approach,
            )
        if approach == "orientation_map":
            fascicle_length, pennation_angle, fig = orientation_map(
                fas_image, apo_image, g, h
            )

        try:
            midthick = mindist[0]  # Muscle thickness
        except:
            midthick = mindist

        fascicle_length = np.array(fascicle_length)
        pennation_angle = np.array(pennation_angle)

        if calib_dist:
            fascicle_length = fascicle_length / (calib_dist / int(spacing))
            midthick = midthick / (calib_dist / int(spacing))

        # add median fascicle length, median pennation angle and muscle thickness to the plot
        if approach == "orientation_map":
            xplot = 75
            yplot = 325
            color = "white"
        else:
            xplot = 0
            yplot = 600
            color = "black"
            plt.text(
                xplot,
                yplot,
                (
                    "Median fascicle length: "
                    + str("%.2f" % np.median(fascicle_length))
                    + " mm"
                ),
                fontsize=10,
                color="black",
            )
            plt.text(
                xplot,
                yplot + 50,
                (
                    "Median pennation angle: "
                    + str("%.1f" % np.median(pennation_angle))
                    + " deg"
                ),
                fontsize=10,
                color="black",
            )

        plt.text(
            xplot,
            yplot + 100,
            ("Thickness at centre: " + str("%.1f" % midthick) + " mm"),
            fontsize=10,
            color=color,
        )

        end_time = time.time()
        total_time = end_time - start_time
        print(total_time)

        return (
            fascicle_length,
            pennation_angle,
            midthick,
            fig,
        )  # TODO what about x_low and x_high? Why aren't they returned?

    return None, None, None, None


def fascicle_calculation():

    parameters = dict(
        apo_threshold=0.2,
        fasc_threshold=0.05,
        apo_length_thresh=400,
        fasc_cont_thresh=5,
        min_width=60,
        max_pennation=40,
        min_pennation=5,
        tolerance=10,
        tolerance_to_apo=100,
    )

    height = 512
    width = 512

    apo_modelpath = (
        "C:/Users/carla/Documents/Master_Thesis/Models/model-apo-VGG16-BCE-512.h5"
    )
    fasc_modelpath = (
        "C:/Users/carla/Documents/Master_Thesis/Models/model-fasc-VGG16-BCE-512.h5"
    )

    model_apo = load_model(apo_modelpath, custom_objects={"IoU": IoU})
    model_fasc = load_model(fasc_modelpath, custom_objects={"IoU": IoU})

    # load ultrasound image
    original_image = cv2.imread(
        r"C:\Users\carla\Documents\Master_Thesis\Example_Images\FALLMUD\NeilCronin\images\img_00016.tif",
        cv2.IMREAD_UNCHANGED,
    )

    img_copy = original_image

    original_image = np.reshape(
        original_image, [-1, original_image.shape[0], original_image.shape[1], 3]
    )
    original_image = resize(
        original_image, (1, 512, 512, 3), mode="constant", preserve_range=True
    )
    original_image = original_image / 255.0

    filter_fasc = True
    calib_dist = None
    spacing = 10

    approach = "curve_polyfitting"  # curve_polyfitting, curve_connect_linear, curve_connect_poly, orientation_map

    fascicle_length, pennation_angle, midthick, fig = doCalculations_curved(
        original_image,
        img_copy,
        height,
        width,
        model_apo,
        model_fasc,
        parameters,
        filter_fasc,
        calib_dist,
        spacing,
        approach,
    )

    plt.show()


if __name__ == "__main__":
    fascicle_calculation()
