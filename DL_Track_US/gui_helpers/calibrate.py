"""
Description
-----------
This module contains functions to automatically scale images.
The scope of the automatic method is limited to scaling bars being
present in the right side of the image. However, the distance
between two selected points in the image required for the scaling must be
known.

Functions scope
---------------
calibrateDistanceStatic
    Function to calibrate an image to convert measurements
    in pixel units to centimeters.
"""

import numpy as np


def calibrateDistanceStatic(img: np.ndarray, spacing: str):
    """Function to calibrate an image to convert measurements
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
    spacing : {10, 5, 15, 20}
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

    # Handle error occuring when no bright pixels detected on right side
    # of image
    except ValueError:

        return None, None
