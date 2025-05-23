"""
Description
-----------
This module contains all additional functions used in the module do_calculations_curved

Functions scope
---------------
adapted_contourEdge
    Function to find only the coordinates representing one edge
    of a contour.
contourEdge
    Function to find only the coordinates representing one edge
    of a contour.
sortContours
    Function to sort detected contours from proximal to distal.
do_curves_intersect
    Function to detect wheter two curves are intersecting or not.
adapted_filter_fascicles
    Filters out fascicles that intersect with other fascicles.
is_point_in_range
    Function to detect wheter a point is between an upper and a lower
    boundary or not.
find_next_fascicle
    Function to find the next fascicle contour.
find_complete_fascicle
    Function to find complete fascicles based on connection of single contours.
crop
    Function to crop the frame around ultrasound images.
"""

import bisect

import cv2
import numpy as np
import pandas as pd
from keras import backend as K
from shapely.geometry import LineString
import tkinter as tk


def adapted_contourEdge(edge: str, contour: list) -> np.ndarray:
    """Function to find only the coordinates representing one edge
    of a contour.

    Either the upper or lower edge of the detected contours is
    calculated. From the contour detected lower in the image,
    the upper edge is searched. From the contour detected
    higher in the image, the lower edge is searched. Allows
    for more points around the end of the contour than contourEdge.

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
    ptsT = sorted(pts, key=lambda k: k[0])

    # Get x and y coordinates from contour

    allx = []
    ally = []
    for a in range(0, len(ptsT)):
        allx.append(ptsT[a][0])
        ally.append(ptsT[a][1])
    # Get rid of doubles
    un = np.unique(allx)

    # Filter x and y coordinates from cont according to selected edge
    leng = len(un) - 1
    x = []
    y = []
    for each in range(
        2, leng - 2
    ):  # range(2, leng - 2):  # Ignore 1st and last 5 points
        indices = [i for i, x in enumerate(allx) if x == un[each]]
        if edge == "T":
            loc = indices[0]
        else:
            loc = indices[-1]
        x.append(ptsT[loc][0])
        y.append(ptsT[loc][1])

    return np.array(x), np.array(y)


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


def do_curves_intersect(curve1: list, curve2: list) -> bool:
    """Function to detect wheter two curves are intersecting or not.

    Parameters
    ----------
    curve1 : list
        List containing (x,y) coordinate pairs representing one curve
    curve2 : list
        List containing (x,y) coordinate pairs representing a second curve

    Returns
    -------
    Bool
        'True' if the curves have an intersection point
        'False' if the curves don't have an intersection point

    Examples
    --------
    >>> do_curves_intersect(curve1=[(98.06, 263.24), (98.26, 263.19), ...],
    curve2=[(63.45, 258.82), (63.65, 258.76), ...])
    """

    line1 = LineString(curve1)
    line2 = LineString(curve2)

    return line1.intersects(line2)


def adapted_filter_fascicles(df: pd.DataFrame, tolerance: int) -> pd.DataFrame:
    """Filters out fascicles that intersect with other fascicles

    This function counts for each fascicle the number of intersections
    with other fascicles and ranks them based on the number of intersections.
    The fascicle with the highest intersection count is excluded.
    This ranking and exclusion process is repeated until there are no more intersections.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing the fascicle data. Expected columns include 'coordsXY'.
    tolerance: integer
        Tolerance to allow intersection points near the aponeuroses

    Returns
    -------
    pd.DataFrame
        A DataFrame with the fascicles that do not intersect with other fascicles.

    Example
    -------
    >>> data = {'coordsXY': [[(78, 268), (78, 266), ...], [(43, 265), (42, 264), ...], ...]}
    >>> tolerance = 100
    >>> adapted_filter_fascicles(data, tolerance)
    """

    df["count"] = 0
    df["intersection"] = [[] for _ in range(len(df))]
    df["keep"] = True

    # detect how many intersection points each fascicle has
    for i in range(len(df)):
        curve1 = df.at[i, "coordsXY"]
        curve1 = curve1[tolerance : len(curve1) - tolerance]
        for j in range(i + 1, len(df)):
            if i != j:
                curve2 = df.at[j, "coordsXY"]
                curve2 = curve2[tolerance : len(curve2) - tolerance]
                if do_curves_intersect(curve1, curve2):
                    df.at[i, "count"] += 1
                    df.at[j, "count"] += 1
                    df.at[i, "intersection"].append(j)
                    df.at[j, "intersection"].append(i)

    while df["count"].max() > 0:

        # determine the maximum amount of intersections
        max_count = df["count"].max()
        max_idx = df.index[df["count"] == max_count].tolist()

        intersection_all = [df.at[idx, "intersection"] for idx in max_idx]
        df.loc[df["count"] == max_count, "keep"] = False

        for i in range(len(max_idx)):
            df.at[max_idx[i], "count"] = np.nan

        for i in range(len(intersection_all)):
            for j in range(len(intersection_all[i])):
                df.at[intersection_all[i][j], "count"] -= 1

    df = df[df["keep"]].drop(columns=["keep"])
    df = df.drop(columns=["count"])
    df = df.drop(columns=["intersection"])

    return df


def is_point_in_range(
    x_point: int, y_point: int, x_poly: list, lb: list, ub: list
) -> bool:
    """Function to detect wheter a point is between an upper and a lower
    boundary or not.

    Parameters
    ----------
    x_point :intger
        Single integer variable representing the x-coordinate of the point to
        be analyzed
    y_point : integer
        Single integer variable representing the y-coordinate of the point to
        be analyzed
    x_poly: list
        List containing all x-coordinates of the range where the point could
        be located
    lb: list
        List containing all y-coordinates of the lower boundary
    ub: list
        List containing all y-coordinates of the upper boundary

    Returns
    -------
    Bool
        'True' if point is within both boundaries
        'False' if point is outside the boundaries

    Examples
    --------
    >>> is_point_in_range(x_point=250, y_point=227,
    x_poly=([-200, ..., 800]), lb=([303.165, ..., 130.044]),
    ub=([323.165, ..., 150.044]))
    """

    # use binary search to find x-point
    idx = bisect.bisect_left(x_poly, x_point)

    # Adjust the index to ensure it's within the valid range
    if idx == 0 or idx == len(x_poly):
        return False

    i = idx - 1

    # Check if y_point is within the bounds for the found interval
    if lb[i] <= y_point <= ub[i]:
        return True

    return False


def find_next_fascicle(
    all_contours: list,
    contours_sorted_x: list,
    contours_sorted_y: list,
    x_current_fascicle: list,
    y_current_fascicle: list,
    x_range: list,
    upper_bound: list,
    lower_bound: list,
    label: dict,
):
    """Function to find the next fascicle contour

    Parameters
    ----------
    all_contours : list
        List containing all (x,y)-coordinates of each detected contour
    contours_sorted_x : list
        List containing all x-coordinates of each detected contour
    contours_sorted_y : list
        List containing all y-coordinates of each detected contour
    x_current_fascicle : list
        List containing the x-coordinates of the currently examined fascicle
    y_current_fascicle : list
        List containing the y-coordinates of the currently examined fascicle
    x_range : list
        List containing all x-coordinates within the range of the extrapolation
    upper_bound : list
        List containing all y-coordinates of the lower boundary
    lower_bound : list
        List containing all y-coordinates of the upper boundary
    label : dictionnary
        Dictionnary containing a label true or false for every fascicle contour,
        true if already used for an extrapolation, false if not

    Returns
    -------
    new_x : list
        List containing the x-coordinates of the currently examined fascicle
        merged with the x-coordinates of the next fascicle contour within the
        boundary if one was found
    new_y : list
        List containing the y-coordinates of the currently examined fascicle
        merged with the y-coordinates of the next fascicle contour within the
        boundary if one was found
    found_fascicle : int
        Integer value of the found fascicle contour, -1 if no contour was found

    Examples
    --------
    >>> find_next_fascicle(all_contours = , contours_sorted_x=[array([ 5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36],dtype=int32),...,array([481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493,494, 495, 496, 497],dtype=int32)]), contours_sorted_y=[array([166, 166, 166, 165, 165, 165, 164, 164, 164, 163, 163, 163, 162, 162, 162, 161, 161, 161, 160, 160, 160, 159, 159, 159, 158, 158, 158, 157, 157, 157, 156, 156], dtype=int32),...,array([76, 76, 76, 76, 75, 75, 75, 74, 74, 74, 74, 73, 73, 73, 72, 72, 72],dtype=int32)], x_current_fascicle=array([ 5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36],dtype=int32), y_current_fascicle=array([166, 166, 166, 165, 165, 165, 164, 164, 164, 163, 163, 163, 162, 162, 162, 161, 161, 161, 160, 160, 160, 159, 159, 159, 158, 158, 158, 157, 157, 157, 156, 156], dtype=int32), x_range=array([-256.   , -255.795, -255.59 , ...,  767.59 ,  767.795,  768.   ]), upper_bound=array([243.137, 243.069, 243.001, ..., -97.372, -97.44 , -97.508]), lower_bound=array([263.137, 263.069, 263.001, ..., -77.372, -77.44 , -77.508]), label={0: True, 1: False, 2: False, 3: False, 4: False, 5: False, 6: False, 7: False, 8: False, 9: False, 10: False, 11: False, 12: False, 13: False})
    """

    found_fascicle = 0

    for i in range(len(all_contours)):
        if len(contours_sorted_x[i]) > 1:
            if (
                contours_sorted_x[i][0] > x_current_fascicle[-1]
                and contours_sorted_y[i][0] < y_current_fascicle[-1]
            ):
                if (
                    is_point_in_range(
                        contours_sorted_x[i][0],
                        contours_sorted_y[i][0],
                        x_range,
                        upper_bound,
                        lower_bound,
                    )
                    and label[i] is False
                ):
                    # print(f"Contour found: {i}")
                    found_fascicle = i
                    break
                else:
                    continue
                    # print("No contour found")

    if found_fascicle > 0:
        new_x = np.append(x_current_fascicle, contours_sorted_x[found_fascicle])
        new_y = np.append(y_current_fascicle, contours_sorted_y[found_fascicle])
    else:
        new_x = x_current_fascicle
        new_y = y_current_fascicle
        found_fascicle = -1

    return new_x, new_y, found_fascicle


def find_complete_fascicle(
    i: int,
    contours_sorted_x: list,
    contours_sorted_y: list,
    contours_sorted: list,
    label: dict,
    mid: int,
    width: int,
    tolerance: int,
    coeff_limit: float,
):
    """Function to find complete fascicles based on connection of single contours.

    The function extrapolates a second order polynomial fit through the first contour. If the coefficients fall outside a specified range, the curve is considered too curved. As a result, a linear fit is calculated and used for subsequent calculations. The next contour is identified if its first point lies within a specified tolerance range in the positive and negative y-direction around the extrapolated fit. If this condition is met, both contours serve as the basis for the next polynomial fit. This process is repeated until no more possible connecting contours are found.

    Parameters
    ----------
    i : int
        Integer value defining the starting contour
    contours_sorted_x : list
        List containing all x-coordinates of each detected contour
    contours_sorted_y : list
        List containing all y-coordinates of each detected contour
    contours_sorted : list
        List containing all (x,y)-coordinates of each detected contour
    label : dictionnary
        Dictionnary containing a label true or false for every fascicle contour,
        true if already used for an extrapolation, false if not
    mid : int
        Integer value defining the middle of the image
    width : int
        Integer value defining the width of the image
    tolerance: int
        Integer value specifing the permissible range in the positive and negative y-direction within which the next contour can be located to still be considered a part of the extrapolated fascicle
    coeff_limit: float
        Value defining the maximum value of the first coefficient in the second polynomial fit

    Returns
    -------
    ex_current_fascicle_x : list
        List containing the x-coordinates of each found and extrapolated fascicle
    ex_current_fascicle_y : list
        List containing the y-coordinates of each found and extrapolated fascicle
    linear_fit : bool
        'True' if extrapolated fit is linear
        'False' if extrapolated fit follows a second order polynomial
    inner_number_contours: list
        List containing the indices of each contour that constitute each fascicle

    Examples
    --------
    >>> find_complete_fascicle(i=0, contours_sorted_x=[array([ 5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36],dtype=int32),...,array([481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497],dtype=int32)]), contours_sorted_y=[array([166, 166, 166, 165, 165, 165, 164, 164, 164, 163, 163, 163, 162, 162, 162, 161, 161, 161, 160, 160, 160, 159, 159, 159, 158, 158, 158, 157, 157, 157, 156, 156],dtype=int32),...,array([76, 76, 76, 76, 75, 75, 75, 74, 74, 74, 74, 73, 73, 73, 72, 72, 72],dtype=int32)], contours_sorted=[[array([ 5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36],dtype=int32),array([166, 166, 166, 165, 165, 165, 164, 164, 164, 163, 163, 163, 162, 162, 162, 161, 161, 161, 160, 160, 160, 159, 159, 159, 158, 158, 158, 157, 157, 157, 156, 156], dtype=int32),...]], label={0: False, 1: False, 2: False, 3: False, 4: False, 5: False, 6: False, 7: False, 8: False, 9: False, 10: False, 11: False, 12: False, 13: False}, mid=256.0, width=512, tolerance=10, coeff_limit=0.000583)
    """

    # get upper edge contour of starting fascicle
    current_fascicle_x = contours_sorted_x[i]
    current_fascicle_y = contours_sorted_y[i]

    # set label to true as fascicle is used
    label[i] = True
    linear_fit = False
    inner_number_contours = []
    inner_number_contours.append(i)

    # calculate second polynomial coefficients
    coefficients = np.polyfit(current_fascicle_x, current_fascicle_y, 2)

    # depending on coefficients edge gets extrapolated as first or second order polynomial
    if (-coeff_limit) < coefficients[0] < coeff_limit:
        g = np.poly1d(coefficients)
        ex_current_fascicle_x = np.linspace(
            mid - width, mid + width, 5000
        )  # Extrapolate x,y data using f function
        ex_current_fascicle_y = g(ex_current_fascicle_x)
        linear_fit = False
    else:
        coefficients = np.polyfit(current_fascicle_x, current_fascicle_y, 1)
        g = np.poly1d(coefficients)
        ex_current_fascicle_x = np.linspace(
            mid - width, mid + width, 5000
        )  # Extrapolate x,y data using f function
        ex_current_fascicle_y = g(ex_current_fascicle_x)
        linear_fit = True

    # compute upper and lower boundary of extrapolation
    upper_bound = ex_current_fascicle_y - tolerance
    lower_bound = ex_current_fascicle_y + tolerance

    # find next fascicle edge within the tolerance, loops as long as a new fascicle edge is found
    # if no new fascicle is found, found_fascicle is set to -1 within function and loop terminates

    found_fascicle = 0

    while found_fascicle >= 0:

        current_fascicle_x, current_fascicle_y, found_fascicle = find_next_fascicle(
            contours_sorted,
            contours_sorted_x,
            contours_sorted_y,
            current_fascicle_x,
            current_fascicle_y,
            ex_current_fascicle_x,
            upper_bound,
            lower_bound,
            label,
        )

        if found_fascicle > 0:
            label[found_fascicle] = True
            inner_number_contours.append(found_fascicle)
        else:
            break

        coefficients = np.polyfit(current_fascicle_x, current_fascicle_y, 2)

        if (-coeff_limit) < coefficients[0] < coeff_limit:
            g = np.poly1d(coefficients)
            ex_current_fascicle_x = np.linspace(
                mid - width, mid + width, 5000
            )  # Extrapolate x,y data using f function
            ex_current_fascicle_y = g(ex_current_fascicle_x)
            linear_fit = False
        else:
            coefficients = np.polyfit(current_fascicle_x, current_fascicle_y, 1)
            g = np.poly1d(coefficients)
            ex_current_fascicle_x = np.linspace(
                mid - width, mid + width, 5000
            )  # Extrapolate x,y data using f function
            ex_current_fascicle_y = g(ex_current_fascicle_x)
            linear_fit = True

        upper_bound = ex_current_fascicle_y - tolerance
        lower_bound = ex_current_fascicle_y + tolerance

    return (
        ex_current_fascicle_x,
        ex_current_fascicle_y,
        linear_fit,
        inner_number_contours,
    )


def find_complete_fascicle_linear(
    i: int,
    contours_sorted_x: list,
    contours_sorted_y: list,
    contours_sorted: list,
    label: dict,
    mid: int,
    width: int,
    tolerance: int,
    coeff_limit: float,
):
    """
    Traces a complete fascicle by iteratively fitting and extrapolating a linear model
    through connected contour segments in an ultrasound image.

    Starting from a seed contour `i`, the function fits a line, extrapolates it over a defined
    width, and collects all contour segments that lie within a vertical tolerance range
    around the extrapolated line. The procedure continues recursively to find connected
    fascicle segments along the extrapolated path.

    Parameters
    ----------
    i : int
        Index of the initial contour used to start the fascicle tracing.
    contours_sorted_x : list of ndarray
        List of arrays containing the x-coordinates of each contour segment.
    contours_sorted_y : list of ndarray
        List of arrays containing the y-coordinates of each contour segment.
    contours_sorted : list of ndarray
        List of full contour arrays (used for proximity checking).
    label : dict
        Dictionary mapping contour indices to a boolean indicating whether they have already been used.
        Will be updated in-place.
    mid : int
        Midpoint x-coordinate used as center of extrapolation range.
    width : int
        Half-width of the extrapolation span in pixels (from `mid - width` to `mid + width`).
    tolerance : int
        Vertical distance in pixels allowed between the extrapolated line and candidate contour segments.
    coeff_limit : float
        Currently unused; placeholder for limiting slope or intercept values during fitting.

    Returns
    -------
    ex_current_fascicle_x : ndarray
        Extrapolated x-coordinates of the complete fascicle.
    ex_current_fascicle_y : ndarray
        Corresponding y-coordinates from the final linear model fit.
    linear_fit : bool
        Flag indicating whether a linear fit was successfully computed (always `True` in current logic).
    inner_number_contours : list of int
        List of contour indices that belong to the detected fascicle.

    Notes
    -----
    - This function modifies the `label` dictionary in-place to prevent reprocessing contours.
    - The input `coeff_limit` is accepted but currently not used for filtering.

    Examples
    --------
    >>> ex_x, ex_y, is_linear, used_contours = find_complete_fascicle_linear(
    ...     i=0,
    ...     contours_sorted_x=contour_x_list,
    ...     contours_sorted_y=contour_y_list,
    ...     contours_sorted=contour_list,
    ...     label={j: False for j in range(len(contour_x_list))},
    ...     mid=256,
    ...     width=100,
    ...     tolerance=5,
    ...     coeff_limit=0.5
    ... )
    >>> plt.plot(ex_x, ex_y)
    >>> print(f"Used {len(used_contours)} contour segments.")
    """

    # get upper edge contour of starting fascicle
    current_fascicle_x = contours_sorted_x[i]
    current_fascicle_y = contours_sorted_y[i]

    # set label to true as fascicle is used
    label[i] = True
    linear_fit = False
    inner_number_contours = []
    inner_number_contours.append(i)

    # calculate linear coefficients
    coefficients = np.polyfit(current_fascicle_x, current_fascicle_y, 1)
    g = np.poly1d(coefficients)
    ex_current_fascicle_x = np.linspace(
        mid - width, mid + width, 5000
    )  # Extrapolate x,y data using f function
    ex_current_fascicle_y = g(ex_current_fascicle_x)
    linear_fit = True

    # compute upper and lower boundary of extrapolation
    upper_bound = ex_current_fascicle_y - tolerance
    lower_bound = ex_current_fascicle_y + tolerance

    # find next fascicle edge within the tolerance, loops as long as a new fascicle edge is found
    # if no new fascicle is found, found_fascicle is set to -1 within function and loop terminates

    found_fascicle = 0

    while found_fascicle >= 0:

        current_fascicle_x, current_fascicle_y, found_fascicle = find_next_fascicle(
            contours_sorted,
            contours_sorted_x,
            contours_sorted_y,
            current_fascicle_x,
            current_fascicle_y,
            ex_current_fascicle_x,
            upper_bound,
            lower_bound,
            label,
        )

        if found_fascicle > 0:
            label[found_fascicle] = True
            inner_number_contours.append(found_fascicle)
        else:
            break

        coefficients = np.polyfit(current_fascicle_x, current_fascicle_y, 1)
        g = np.poly1d(coefficients)
        ex_current_fascicle_x = np.linspace(
            mid - width, mid + width, 5000
        )  # Extrapolate x,y data using f function
        ex_current_fascicle_y = g(ex_current_fascicle_x)
        linear_fit = True

        upper_bound = ex_current_fascicle_y - tolerance
        lower_bound = ex_current_fascicle_y + tolerance

    return (
        ex_current_fascicle_x,
        ex_current_fascicle_y,
        linear_fit,
        inner_number_contours,
    )


def crop(original_image: list, image_fas: list, image_apo: list):
    """Function to crop the frame around ultrasound images

    Additionally crops the fascicle and aponeuroses images in order that all three images have the same size

    Parameters
    ----------
    original_image : list
        Image of the original ultrasound image
    image_fas : list
        Binary image of the fascicles within the original image
    image_apo: list
        Binary image of the aponeuroses within the original image

    Returns
    -------
    cropped_US : list
        Image of the original ultrasound image without frame around it
    cropped_fas : list
        Cropped binary image of fascicles within the original image
    cropped_apo : list
        Cropped binary image of the aponeuroses within the original image

    Examples
    --------
    >>> crop(original_image=array([[[160, 160, 160],[159, 159, 159],[158, 158, 158],...[158, 158, 158],[147, 147, 147],[  1,   1,   1]],...,[[  0,   0,   0],[  0,   0,   0],[  0,   0,   0],...,[  4,   4,   4],[  3,   3,   3],[  3,   3,   3]]], dtype=uint8), image_fas = array([[0, 0, 0, ..., 0, 0, 0],[0, 0, 0, ..., 0, 0, 0],[0, 0, 0, ..., 0, 0, 0],...,[0, 0, 0, ..., 0, 0, 0],[0, 0, 0, ..., 0, 0, 0],[0, 0, 0, ..., 0, 0, 0]], dtype=uint8), image_apo = array([[[0, 0, 0],[0, 0, 0],[0, 0, 0],...,[0, 0, 0],[0, 0, 0],[0, 0, 0]]], dtype=uint8))
    """

    # define mask, pixel value has to be higher than 10
    mask = np.array((original_image > 10).astype("f4"))

    # find contours
    cnts, _ = cv2.findContours(
        cv2.cvtColor((mask * 255).astype("u1"), cv2.COLOR_BGR2GRAY),
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE,
    )[-2:]

    # define contour with the biggest area
    c = max(cnts, key=cv2.contourArea)

    # get starting point and width, height of biggest area as a rectangle
    x, y, w, h = cv2.boundingRect(c)

    # crop original, fascicle and aponeuroses images
    cropped_US = np.array(original_image[y : y + h, x : x + w])
    cropped_fas = np.array(image_fas[y : y + h, x : x + w])
    cropped_apo = np.array(image_apo[y : y + h, x : x + w])

    return cropped_US, cropped_fas, cropped_apo
