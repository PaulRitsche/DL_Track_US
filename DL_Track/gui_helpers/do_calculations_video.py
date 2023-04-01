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

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from scipy.signal import savgol_filter
from skimage.morphology import skeletonize
from skimage.transform import resize
from tensorflow.keras.utils import img_to_array

from DL_Track.gui_helpers.calculate_architecture import IoU
from DL_Track.gui_helpers.do_calculations import contourEdge, sortContours

plt.style.use("ggplot")


def doCalculationsVideo(
    vid_len: int,
    cap,
    vid_out,
    flip: str,
    apo_modelpath: str,
    fasc_modelpath: str,
    calib_dist: int,
    dic: dict,
    step: int,
    gui,
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
    flip : str
        String variable defining wheter an image should be flipped.
        This can be "no_flip" (video is not flipped) or "flipe"
        (video is flipped).
    apo_modelpath : str
        String variable containing the absolute path to the aponeurosis
        neural network.
    fasc_modelpath : str
        String variable containing the absolute path to the fascicle
        neural network.
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
    gui : tk.TK
        A tkinter.TK class instance that represents a GUI. By passing this
        argument, interaction with the GUI is possible i.e., stopping
        the calculation process after each image.

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
                        gui=<__main__.DLTrack object at 0x000002BFA7528190>)
    [array([60.5451731 , 58.86892027, 64.16011534, 55.46192704, 63.40711356]), ..., array([64.90849385, 60.31621836])]
    [[19.124207107383114, 19.409753216521565, 18.05706763600641, 20.54453899050867, 17.808652286488794], ..., [17.26241882195032, 16.284803480359543]]
    [[148, 5, 111, 28, -164], [356, 15, 105, -296], [357, 44, -254], [182, 41, -233], [40, 167, 42, -170], [369, 145, 57, -139], [376, 431, 32], [350, 0]]
    [[725, 568, 725, 556, 444], [926, 572, 516, 508], [971, 565, 502], [739, 578, 474], [554, 766, 603, 475], [1049, 755, 567, 430], [954, 934, 568], [968, 574]]
    [23.484416057267826, 22.465452189555794, 21.646971767045816, 21.602856412413924, 21.501286239714894, 21.331137350026623, 21.02446763240188, 21.250352548097883]
    """
    try:

        # Check if analysis parameters are postive
        for _, value in dic.items():
            if float(value) <= 0:
                tk.messagebox.showerror(
                    "Information",
                    "Analysis paremters must be non-zero" + " and non-negative"
                )
                gui.should_stop = False
                gui.is_running = False
                gui.do_break()
                return

        # Get variables from dictionary
        fasc_cont_thresh = int(dic["fasc_cont_thresh"])
        min_width = int(dic["min_width"])
        max_pennation = int(dic["max_pennation"])
        min_pennation = int(dic["min_pennation"])
        apo_threshold = float(dic["apo_treshold"])
        fasc_threshold = float(dic["fasc_threshold"])

        # Define empty lists for parameter storing
        fasc_l_all = []
        pennation_all = []
        x_lows_all = []
        x_highs_all = []
        thickness_all = []

        # load the aponeurosis model
        model_apo = load_model(apo_modelpath, custom_objects={"IoU": IoU})
        # load the fascicle model
        model_fasc = load_model(fasc_modelpath, custom_objects={"IoU": IoU})

        # Loop through each frame of the video
        for a in range(0, vid_len - 1, step):

            if gui.should_stop:
                # there was an input to stop the calculations
                break

            # Reshape, resize and normalize each frame.
            _, frame = cap.read()
            img = img_to_array(frame)
            if flip == "flip":
                img = np.fliplr(img)
            img_orig = img  # Make a copy
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h = img.shape[0]
            w = img.shape[1]
            img = np.reshape(img, [-1, h, w, 3])
            img = resize(img, (1, 512, 512, 3), mode="constant",
                         preserve_range=True)
            img = img / 255.0

            # Predict aponeurosis
            pred_apo = model_apo.predict(img)
            # Employ threshold for segmentation
            pred_apo_t = (pred_apo > apo_threshold).astype(np.uint8)

            # Predict fascicle segments
            pred_fasc = model_fasc.predict(img)
            # Employ threshold for segmentation
            pred_fasc_t = (pred_fasc > fasc_threshold).astype(np.uint8)

            # Resize and reshape predictions for further usage
            img = resize(img, (1, h, w, 1))
            img = np.reshape(img, (h, w))
            pred_apo = resize(pred_apo, (1, h, w, 1))
            pred_apo = np.reshape(pred_apo, (h, w))
            pred_apo_t = resize(pred_apo_t, (1, h, w, 1))
            pred_apo_t = np.reshape(pred_apo_t, (h, w))
            pred_fasc = resize(pred_fasc, (1, h, w, 1))
            pred_fasc = np.reshape(pred_fasc, (h, w))
            pred_fasc_t = resize(pred_fasc_t, (1, h, w, 1))
            pred_fasc_t = np.reshape(pred_fasc_t, (h, w))

            # Aponuerosis calculation PArt

            # Compute the contours to identify aponeuroses
            _, thresh = cv2.threshold(pred_apo_t, 0, 255, cv2.THRESH_BINARY)
            thresh = thresh.astype("uint8")
            # Find contours in thresholded image
            contours, hierarchy = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )

            # include contours of long length
            contours = [i for i in contours if len(i) > 600]
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
            for countU in range(0, len(contours_re2) - 1):
                if (
                    xs1[countU + 1] > xs2[countU]
                ):  # Check if x of contour2 is higher than x of contour 1
                    y1 = ys2[countU]
                    y2 = ys1[countU + 1]
                    if y1 - 10 <= y2 <= y1 + 10:
                        m = np.vstack((contours_re2[countU],
                                       contours_re2[countU + 1]))
                        cv2.drawContours(maskT, [m], 0, 255, -1)
                countU += 1

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
            mask_apoE = np.zeros(thresh.shape, np.uint8)

            # Select only long contours and draw to mask
            contoursE = [i for i in contoursE if len(i) > 600]
            for contour in contoursE:
                cv2.drawContours(mask_apoE, [contour], 0, 255, -1)

            # Sort contours again from top to bottom
            contoursE, _ = sortContours(contoursE)

            # Continue only when 2 or more aponeuroses were detected
            if len(contoursE) >= 2:

                # Get the x,y coordinates of the upper/lower edge of the 2
                # aponeuroses
                upp_x, upp_y = contourEdge("B", contoursE[0])
                if contoursE[1][0, 0, 1] > (contoursE[0][0, 0, 1] + min_width):
                    low_x, low_y = contourEdge("T", contoursE[1])
                else:
                    low_x, low_y = contourEdge("T", contoursE[2])

                # Filter data one-dimensionally to extend the data
                upp_y_new = savgol_filter(upp_y, 81, 2)
                low_y_new = savgol_filter(low_y, 81, 2)

                # Make a binary mask
                ex_mask = np.zeros(thresh.shape, np.uint8)
                ex_1 = 0
                ex_2 = np.minimum(len(low_x), len(upp_x))
                for ii in range(ex_1, ex_2):
                    ymin = int(np.floor(upp_y_new[ii]))
                    ymax = int(np.ceil(low_y_new[ii]))

                    ex_mask[:ymin, ii] = 0
                    ex_mask[ymax:, ii] = 0
                    ex_mask[ymin:ymax, ii] = 255

                # Calculate slope of central portion of each aponeurosis
                # & use this to compute muscle thickness
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
                            (upp_x[upp_ind], upp_y_new[upp_ind]),
                            (low_x[val], low_y_new[val]),
                        )
                        if dist < mindist:
                            mindist = dist

                # Add aponeuroses to a mask for display
                imgT = np.zeros((h, w, 3), np.uint8)

                # Compute functions to approximate the shape of the aponeuroses
                zUA = np.polyfit(upp_x, upp_y_new, 2)  # 2nd order polynomial
                g = np.poly1d(zUA)
                zLA = np.polyfit(low_x, low_y_new, 2)
                h = np.poly1d(zLA)

                mid = (low_x[-1] - low_x[0]) / 2 + low_x[
                    0
                ]  # Find middle of the aponeurosis
                x1 = np.linspace(
                    low_x[0] - 700, low_x[-1] + 700, 10000
                )  # Extrapolate polynomial fits to either side
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

                # Fascicle calculation part

                # Compute contours to identify fascicles / fascicle orientation
                _, threshF = cv2.threshold(pred_fasc_t, 0, 255,
                                           cv2.THRESH_BINARY)
                threshF = threshF.astype("uint8")
                contoursF, hierarchy = cv2.findContours(
                    threshF, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                # Remove any contours that are very small
                maskF = np.zeros(threshF.shape, np.uint8)
                for contour in contoursF:
                    if len(contour) > fasc_cont_thresh:
                        cv2.drawContours(maskF, [contour], 0, 255, -1)

                # Only include fascicles within the region of the 2 aponeuroses
                mask_Fi = maskF & ex_mask
                contoursF2, hierarchy = cv2.findContours(
                    mask_Fi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )

                # Only include fascicles longer then specified length
                contoursF3 = []
                for contour in contoursF2:
                    if len(contour) > fasc_cont_thresh:
                        contoursF3.append(contour)

                # Define lists to store analysis parameters
                xs = []
                ys = []
                fas_ext = []
                fasc_l = []
                pennation = []
                x_low1 = []
                x_high1 = []

                # Loop through facicle contours to compute fascicle
                for cnt in contoursF3:
                    x, y = contourEdge("B", cnt)
                    if len(x) == 0:
                        continue
                    z = np.polyfit(np.array(x), np.array(y), 1)
                    f = np.poly1d(z)
                    newX = np.linspace(
                        -400, w + 400, 5000
                    )  # Extrapolate x,y data using f function
                    newY = f(newX)

                    # Find intersection between each fascicle and the
                    # aponeuroses.
                    diffU = newY - new_Y_UA  # Find intersections
                    locU = np.where(diffU == min(diffU, key=abs))[0]
                    diffL = newY - new_Y_LA
                    locL = np.where(diffL == min(diffL, key=abs))[0]

                    coordsX = newX[int(locL): int(locU)]
                    coordsY = newY[int(locL): int(locU)]

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
                    Apoangle = 90 + abs(Apoangle)

                    # Don't include fascicles that are completely outside of
                    # the field of view or those that don't pass through
                    # central 1/3 of the image
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
                                (newX[locU] - newX[locL]) ** 2
                                + (y_UA[locU] - y_LA[locL]) ** 2
                            )
                            fasc_l.append(length1[0])  # Calculate FL
                            pennation.append(Apoangle - FascAng)
                            x_low1.append(coordsX[0].astype("int32"))
                            x_high1.append(coordsX[-1].astype("int32"))
                            coords = np.array(
                                list(
                                    zip(
                                        coordsX.astype("int32"),
                                        coordsY.astype("int32")
                                    )
                                )
                            )
                            cv2.polylines(imgT, [coords], False, (20, 15, 200),
                                          3)

                # Store the results for each frame and normalise using scale
                # factor (if calibration was done above)
                try:
                    midthick = mindist[0]  # Muscle thickness
                except:
                    midthick = mindist

                if calib_dist:
                    fasc_l = fasc_l / (calib_dist / 10)
                    midthick = midthick / (calib_dist / 10)

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

            fasc_l_all.append(fasc_l)
            pennation_all.append(pennation)
            x_lows_all.append(x_low1)
            x_highs_all.append(x_high1)
            thickness_all.append(midthick)

            # Display each processed frame
            img_orig[mask_apoE > 0] = (235, 25, 42)

            comb = cv2.addWeighted(img_orig.astype(np.uint8), 1, imgT, 0.8, 0)
            vid_out.write(comb)  # Write each image to video file
            cv2.putText(
                comb,
                ("Frame: " + str(a + 1) + " of " + str(vid_len)),
                (125, 350),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (249, 249, 249),
            )
            cv2.putText(
                comb,
                (
                    "Pennation angle: "
                    + str("%.1f" % np.median(pennation_all[-1]))
                    + " deg"
                ),
                (125, 410),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (249, 249, 249),
            )
            if calib_dist:
                cv2.putText(
                    comb,
                    (
                        "Fascicle length: "
                        + str("%.2f" % np.median(fasc_l_all[-1]) + " mm")
                    ),
                    (125, 380),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1,
                    (249, 249, 249),
                )
                cv2.putText(
                    comb,
                    ("Thickness at centre: " + str("%.1f" % thickness_all[-1]) +
                     " mm"),
                    (125, 440),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1,
                    (249, 249, 249),
                )
            else:
                cv2.putText(
                    comb,
                    (
                        "Fascicle length: "
                        + str("%.2f" % np.median(fasc_l_all[-1]) + " px")
                    ),
                    (125, 380),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1,
                    (249, 249, 249),
                )
                cv2.putText(
                    comb,
                    ("Thickness at centre: " + str("%.1f" % thickness_all[-1])
                     + " px"),
                    (125, 440),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1,
                    (249, 249, 249),
                )

            # Check platform for imshow
            # Windows
            if platform in ("win32", "linux"):
                cv2.imshow("Analysed image", comb)
            # MacOS
            elif platform == "darwin":
                print("Analysed image cannot be displayed on MacOS.")

            # Press 'q' to stop the analysis
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

        # Release video and close all analysis windows when analysis finished
        cap.release()
        vid_out.release()
        cv2.destroyAllWindows()

        return fasc_l_all, pennation_all, x_lows_all, x_highs_all, thickness_all

    # Check if model path is correct
    except OSError:
        tk.messagebox.showerror("Information",
                                "Apo/Fasc model path is incorrect.")
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
