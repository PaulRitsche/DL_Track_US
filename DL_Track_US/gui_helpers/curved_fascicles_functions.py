import bisect

import cv2
import numpy as np


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
    for each in range(2, leng - 2):  # Ignore 1st and last 5 points
        indices = [i for i, x in enumerate(allx) if x == un[each]]
        if edge == "T":
            loc = indices[0]
        else:
            loc = indices[-1]
        x.append(ptsT[loc][0])
        y.append(ptsT[loc][1])

    return np.array(x), np.array(y)


def is_point_in_range(x_point, y_point, x_poly, lb, ub):  # x_point at first place
    for i in range(len(x_poly) - 1):
        if x_poly[i] <= x_point <= x_poly[i + 1]:
            if lb[i] <= y_point <= ub[i]:
                return True
    return False


def is_point_in_range_2(x_point, y_point, x_poly, lb, ub):
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
    all_contours,
    contours_sorted_x,
    contours_sorted_y,
    x_current_fascicle,
    y_current_fascicle,
    x_range,
    upper_bound,
    lower_bound,
    label,
):

    found_fascicle = 0

    for i in range(len(all_contours)):
        if len(contours_sorted_x[i]) > 0:
            if (
                contours_sorted_x[i][0] > x_current_fascicle[-1]
                and contours_sorted_y[i][0] < y_current_fascicle[-1]
            ):
                if (
                    is_point_in_range_2(
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
