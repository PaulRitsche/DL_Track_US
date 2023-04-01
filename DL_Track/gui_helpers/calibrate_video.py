"""
Description
-----------
This module contains functions to manually scale video frames.
The scope of the manual method is not limited to specific scaling
types in images. However, the distance between two selected points
in the image required for the scaling must be known.
We assume that the scaling does not change throughout the video, which is why
only the first frame is used for calibration.

Functions scope
---------------
mclick
    Instance method to detect mouse click coordinates in image.
calibrateDistanceManually
    Function to manually calibrate an image to convert measurements
    in pixel units to centimeters.

See Also
--------
calibrate.py
"""
import tkinter as tk
from sys import platform

import cv2
import numpy as np

# global variable to store mouse clicks
mlocs = []


def mclick(event, x_val, y_val, flags, param):
    """Instance method to detect mouse click coordinates in image.

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


def calibrateDistanceManually(cap, spacing: int):
    """Function to manually calibrate an image to convert measurements
    in pixel units to centimeters.

    The function calculates the distance in pixel units between two
    points on the input image. The points are determined by clicks of
    the user. The distance (in milimeter) is determined by the value
    contained in the spacing variable. Then the ratio of pixel / centimeter
    is calculated. To get the distance, the euclidean distance between the
    two points is calculated.

    Parameters
    ----------
    cap : cv2.VideoCapture
        Object that contains the video in a np.ndarrray format.
        In this way, seperate frames can be accessed.
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
    >>> calibrateDistanceManually(cap=VideoCapture 000002A261ADC590, 5)
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

    # Get global variable
    global mlocs
    mlocs = []

    # take only first frame
    _, frame = cap.read()

    # display the image and select points
    cv2.imshow("image", frame)
    cv2.setMouseCallback("image", mclick)
    key = cv2.waitKey(0)

    # if the 'q' key is pressed, break from the loop
    if key == ord("q"):
        cv2.destroyAllWindows()

    # Caclulate absolute distance between points
    calib_dist = np.abs(np.abs(mlocs[0] - mlocs[1]))

    # Empty coordinate list again
    mlocs = []

    # calculate distance for 10mm
    if spacing == 5:
        calib_dist = calib_dist * 2
    if spacing == 15:
        calib_dist = calib_dist * (2 / 3)
    if spacing == 20:
        calib_dist = calib_dist / 2

    # Formulate scale statement
    scale_statement = "10 mm corresponds to " + str(calib_dist) + " pixels"

    return calib_dist, scale_statement
