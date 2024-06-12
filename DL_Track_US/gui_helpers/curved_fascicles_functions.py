import bisect

import cv2
import numpy as np
import pandas as pd
from shapely.geometry import LineString


def adapted_contourEdge(edge: str, contour: list) -> np.ndarray:
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
    #>>> contourEdge(edge="T", contour=[[[195 104]] ... [[196 104]]])
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

    """
    found_fascicle = 0

    for i in range(len(all_contours)):
        if len(contours_sorted_x[i]) > 0:
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


def crop(original_image, image_fas, image_apo):
    """Function to crop the frame around ultrasound images

    Parameters
    ----------
    original_image : list
        List containing (x,y) coordinate pairs representing one curve
    image_fas : list
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
    croped_US = np.array(original_image[y : y + h, x : x + w])
    croped_fas = np.array(image_fas[y : y + h, x : x + w])
    croped_apo = np.array(image_apo[y : y + h, x : x + w])

    return croped_US, croped_fas, croped_apo
