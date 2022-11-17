"""
Description
-----------
This module contains functions to automatically or manually
scale images.
The scope of the automatic method is limited to scaling bars being
present in the right side of the image. The scope of the manual method
is not limited to specific scaling types in images. However, the distance
between two selected points in the image required for the scaling must be known.

Functions scope
---------------
mclick
    Instance method to detect mouse click coordinates in image.
calibrateDistanceManually
    Function to manually calibrate an image to convert measurements
    in pixel units to centimeters.
calibrateDistanceStatic
    Function to calibrate an image to convert measurements
    in pixel units to centimeters.
"""
import math
import tkinter as tk
from sys import platform

import cv2
import numpy as np

# global variable to store mouse clicks
mlocs = []


def mclick(event, x_val, y_val, flags, param):
    """
    Instance method to detect mouse click coordinates in image.

    This instance is used when the image to be analyzed should be
    cropped. Upon clicking the mouse button, the coordinates
    of the cursor position are stored in the instance attribute
    self.mlocs.

    Parameters
    ----------
    event
        Event flag specified as Cv2 mouse event left mouse button down.
    x_val
        Value of x-coordinate of mouse event to be recorded.
    y_val
        Value of y-coordinate of mouse event to be recorded.
    flags
        Specific condition whenever a mouse event occurs. This
        is not used here but needs to be specified as input
        parameter.
    param
        User input data. This is not required here but needs to
        be specified as input parameter.
    """
    # Define global variable for functions to access
    global mlocs

    # if the left mouse button was clicked, record the (x, y) coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        mlocs.append(y_val)
        mlocs.append(x_val)


def calibrateDistanceManually(img: np.ndarray, spacing: int):
    """
    Function to manually calibrate an image to convert measurements
    in pixel units to centimeters.

    The function calculates the distance in pixel units between two
    points on the input image. The points are determined by clicks of
    the user. The distance (in milimeters) is determined by the value
    contained in the spacing variable. Then the ratio of pixel / centimeter
    is calculated. To get the distance, the euclidean distance between the
    two points is calculated.

    Parameters
    ----------
    img : np.ndarray
        Input image to be analysed as a numpy array. The image must
        be loaded prior to calibration, specifying a path
        is not valid.
    spacing : int
        Integer variable containing the known distance in milimeter
        between the two placed points by the user. This can be 5, 10,
        15 or 20 milimeter.

    Returns
    -------
        calib_dist : int
            Integer variable containing the distance between the two
            specified point in pixel units.
        scale_statement : str
            String variable containing a statement how many milimeter
            correspond to how many pixels.

    Examples
    --------
    >>> calibrateDistanceManually(img=([[[[0.22414216 0.19730392 0.22414216] ... [0.2509804  0.2509804  0.2509804 ]]]), 5)
    99, 5 mm corresponds to 99 pixels
    """
    # Check platform for imshow and if MacOS, break
    if platform == "darwin":
        tk.messagebox.showerror(
            "Information",
            "Manual scaling not available on MacOS"
            + "\n Contine with 'No Scaling' Scaing Type.",
        )
        return None, None

    img2 = np.uint8(img)

    # display the image and wait for a keypress
    cv2.imshow("image", img2)
    cv2.setMouseCallback("image", mclick)
    key = cv2.waitKey(0)

    # if the 'q' key is pressed, break from the loop
    if key == ord("q"):
        cv2.destroyAllWindows()

    global mlocs

    calib_dist = np.abs(
        math.sqrt((mlocs[3] - mlocs[1]) ** 2 + (mlocs[2] - mlocs[0]) ** 2)
    )
    mlocs = []
    # calculate calib_dist for 10mm
    if spacing == 5:
        calib_dist = calib_dist * 2
    if spacing == 15:
        calib_dist = calib_dist * (2 / 3)
    if spacing == 20:
        calib_dist = calib_dist / 2

    # print(str(spacing) + ' mm corresponds to ' + str(calib_dist) + ' pixels')
    scale_statement = "10 mm corresponds to " + str(calib_dist) + " pixels"

    return calib_dist, scale_statement


def calibrateDistanceStatic(img: np.ndarray, spacing: str):
    """
    Function to calibrate an image to convert measurements
    in pixel units to centimeter.

    The function calculates the distance in pixel units between two
    scaling bars on the input image. The bars should be positioned on the
    right side of image. The distance (in milimeter) between two bars must
    be specified by the spacing variable. It is the known distance between two
    bars in milimeter. Then the ratio of pixel / centimeter is calculated.
    To get the distance, the median distance between two detected bars
    is calculated.

    Parameters
    ----------
    img : np.ndarray
        Input image to be analysed as a numpy array. The image must
        be loaded prior to calibration, specifying a path
        is not valid.
    spacing : int
        Integer variable containing the known distance in milimeter
        between the two scaling bars. This can be 5, 10,
        15 or 20 milimeter.

    Returns
    -------
    calib_dist : int
        Integer variable containing the distance between the two
        specified point in pixel units.
    scale_statement : str
        String variable containing a statement how many milimeter
        correspond to how many pixels.

    Examples
    --------
    >>> calibrateDistanceStatic(img=([[[[0.22414216 0.19730392 0.22414216] ... [0.2509804  0.2509804  0.2509804 ]]]), 5)
    99, 5 mm corresponds to 99 pixels
    """
    try:
        # crop right border of image
        height = img.shape[0]
        width = img.shape[1]
        imgscale = img[
            int(height * 0.4) : (height), (width - int(width * 0.15)) : width
        ]

        # search for rows with white pixels, calculate median of distance
        calib_dist = np.max(np.diff(np.argwhere(imgscale.max(axis=1) > 150), axis=0))

        # return none if distance too small
        if int(calib_dist) < 1:
            return None, None

        # create scale statement
        scale_statement = f"{spacing} mm corresponds to {calib_dist} pixels"

        return calib_dist, scale_statement

    # Handle error occuring when no bright pixels detected on right side of image
    except ValueError:

        return None, None
