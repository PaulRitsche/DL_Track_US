import math
import time

import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy
import numpy as np
import orientationpy
import pandas as pd
from curved_fascicles_functions import (
    adapted_contourEdge,
    adapted_filter_fascicles,
    crop,
    do_curves_intersect,
    find_next_fascicle,
)
from do_calculations import contourEdge, sortContours
from scipy.interpolate import Rbf
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from skimage.morphology import skeletonize

# load fascicle mask
fas_image = cv2.imread(
    r"C:\Users\carla\Documents\Master_Thesis\Example_Images\FALLMUD\NeilCronin\fascicle_masks\img_00049.tif",
    cv2.IMREAD_UNCHANGED,
)
# load aponeurosis mask
apo_image = cv2.imread(
    r"C:\Users\carla\Documents\Master_Thesis\Example_Images\FALLMUD\NeilCronin\aponeurosis_masks\img_00049.jpg",
    cv2.IMREAD_UNCHANGED,
)
# load ultrasound image
original_image = cv2.imread(
    r"C:\Users\carla\Documents\Master_Thesis\Example_Images\FALLMUD\NeilCronin\images\img_00049.tif",
    cv2.IMREAD_UNCHANGED,
)

# crop all three images in order that they don't have a frame
original_image, fas_image, apo_image = crop(original_image, fas_image, apo_image)

parameters = dict(
    apo_length_thresh=600,
    fasc_cont_thresh=40,
    min_width=60,
    max_pennation=40,
    min_pennation=5,
    tolerance=10,
    tolerance_to_apo=100,
    coeff_limit=0.000583,
)

filter_fasc = True

approach = 3


def doCalculations_curved(
    original_image, fas_image, apo_image, parameters, filter_fasc, approach
):

    parameters = parameters

    apo_length_tresh = int(parameters["apo_length_thresh"])
    fasc_cont_thresh = int(parameters["fasc_cont_thresh"])
    min_width = int(parameters["min_width"])

    # calculations for aponeuroses
    apo_image_rgb = cv2.cvtColor(apo_image, cv2.COLOR_BGR2RGB)
    apo_image_gray = cv2.cvtColor(apo_image_rgb, cv2.COLOR_RGB2GRAY)
    width = fas_image.shape[1]

    _, thresh = cv2.threshold(
        apo_image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
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
    # if len(contours) < 1:
    # return None

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
        image_rgb = cv2.cvtColor(fas_image, cv2.COLOR_BGR2RGB)
        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        # define threshold and find contours around fascicles
        _, threshF = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY)
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

        if approach == 1:
            Curved_Approach_1(
                contours_sorted,
                new_X_LA,
                new_Y_LA,
                new_X_UA,
                new_Y_UA,
                original_image,
                parameters,
                filter_fasc,
            )
        if approach == 2 or approach == 3:
            Curved_Approach_2_3(
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
        if approach == 4:
            Orientation_map(original_image, fas_image, apo_image, g, h)


def Curved_Approach_1(
    contours_sorted,
    ex_x_LA,
    ex_y_LA,
    ex_x_UA,
    ex_y_UA,
    original_image,
    parameters,
    filter_fasc,
):

    start_time = time.time()

    # Set parameters
    tolerance = int(parameters["tolerance"])
    tolerance_to_apo = int(parameters["tolerance_to_apo"])
    coeff_limit = float(parameters["coeff_limit"])
    max_pennation = int(parameters["max_pennation"])
    min_pennation = int(parameters["min_pennation"])

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
            if 0 < coefficients[0] < coeff_limit:
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

                current_fascicle_x, current_fascicle_y, found_fascicle = (
                    find_next_fascicle(
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
                )

                if found_fascicle > 0:
                    label[found_fascicle] = True
                    inner_number_contours.append(found_fascicle)
                else:
                    break

                coefficients = np.polyfit(current_fascicle_x, current_fascicle_y, 2)

                if 0 < coefficients[0] < coeff_limit:
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

    end_time = time.time()
    total_time = end_time - start_time

    median_length = data["fascicle_length"].median()
    mean_length = data["fascicle_length"].mean()
    median_angle = data["pennation_angle"].median()
    mean_angle = data["pennation_angle"].mean()

    print(total_time)
    print(data)
    print(median_length, mean_length, median_angle, mean_angle)

    # plot filtered curves between detected fascicles between the two aponeuroses

    colormap = plt.get_cmap("rainbow", len(all_coordsX))
    number_contours = list(data["number_contours"])  # contours after filtering

    for i in range(len(all_coordsX)):

        color = colormap(i)
        x = all_coordsX[i]
        y = all_coordsY[i]

        if len(number_contours[i]) == 1:
            a = contours_sorted_x[number_contours[i][0]][0]
            b = contours_sorted_x[number_contours[i][0]][-1]
            x_before_a = x[x <= a]
            y_before_a = y[x <= a]
            x_after_b = x[x >= b]
            y_after_b = y[x >= b]

            plt.figure(1)
            plt.plot(x_before_a, y_before_a, color=color, alpha=0.4)
            plt.plot(x_after_b, y_after_b, color=color, alpha=0.4)
            plt.plot(
                contours_sorted_x[number_contours[i][0]],
                contours_sorted_y[number_contours[i][0]],
                color="gold",
                alpha=0.6,
            )

        if len(number_contours[i]) == 2:
            a = contours_sorted_x[number_contours[i][0]][0]
            b = contours_sorted_x[number_contours[i][0]][-1]
            c = contours_sorted_x[number_contours[i][1]][0]
            d = contours_sorted_x[number_contours[i][1]][-1]
            x_before_a = x[x <= a]
            y_before_a = y[x <= a]
            x_b_to_c = x[(x >= b) & (x <= c)]
            y_b_to_c = y[(x >= b) & (x <= c)]
            x_after_d = x[x >= d]
            y_after_d = y[x >= d]

            plt.figure(1)
            plt.plot(x_before_a, y_before_a, color=color, alpha=0.4)
            plt.plot(x_b_to_c, y_b_to_c, color=color, alpha=0.4)
            plt.plot(x_after_d, y_after_d, color=color, alpha=0.4)
            plt.plot(
                contours_sorted_x[number_contours[i][0]],
                contours_sorted_y[number_contours[i][0]],
                color="gold",
                alpha=0.6,
            )
            plt.plot(
                contours_sorted_x[number_contours[i][1]],
                contours_sorted_y[number_contours[i][1]],
                color="gold",
                alpha=0.6,
            )

        if len(number_contours[i]) == 3:
            a = contours_sorted_x[number_contours[i][0]][0]
            b = contours_sorted_x[number_contours[i][0]][-1]
            c = contours_sorted_x[number_contours[i][1]][0]
            d = contours_sorted_x[number_contours[i][1]][-1]
            e = contours_sorted_x[number_contours[i][2]][0]
            f = contours_sorted_x[number_contours[i][2]][-1]
            x_before_a = x[x <= a]
            y_before_a = y[x <= a]
            x_b_to_c = x[(x >= b) & (x <= c)]
            y_b_to_c = y[(x >= b) & (x <= c)]
            x_d_to_e = x[(x >= d) & (x <= e)]
            y_d_to_e = y[(x >= d) & (x <= e)]
            x_after_f = x[x >= f]
            y_after_f = y[x >= f]

            plt.figure(1)
            plt.plot(x_before_a, y_before_a, color=color, alpha=0.4)
            plt.plot(x_b_to_c, y_b_to_c, color=color, alpha=0.4)
            plt.plot(x_d_to_e, y_d_to_e, color=color, alpha=0.4)
            plt.plot(x_after_f, y_after_f, color=color, alpha=0.4)
            plt.plot(
                contours_sorted_x[number_contours[i][0]],
                contours_sorted_y[number_contours[i][0]],
                color="gold",
                alpha=0.6,
            )
            plt.plot(
                contours_sorted_x[number_contours[i][1]],
                contours_sorted_y[number_contours[i][1]],
                color="gold",
                alpha=0.6,
            )
            plt.plot(
                contours_sorted_x[number_contours[i][2]],
                contours_sorted_y[number_contours[i][2]],
                color="gold",
                alpha=0.6,
            )

        if len(number_contours[i]) == 4:
            a = contours_sorted_x[number_contours[i][0]][0]
            b = contours_sorted_x[number_contours[i][0]][-1]
            c = contours_sorted_x[number_contours[i][1]][0]
            d = contours_sorted_x[number_contours[i][1]][-1]
            e = contours_sorted_x[number_contours[i][2]][0]
            f = contours_sorted_x[number_contours[i][2]][-1]
            g = contours_sorted_x[number_contours[i][3]][0]
            h = contours_sorted_x[number_contours[i][3]][-1]
            x_before_a = x[x <= a]
            y_before_a = y[x <= a]
            x_b_to_c = x[(x >= b) & (x <= c)]
            y_b_to_c = y[(x >= b) & (x <= c)]
            x_d_to_e = x[(x >= d) & (x <= e)]
            y_d_to_e = y[(x >= d) & (x <= e)]
            x_f_to_g = x[(x >= f) & (x <= g)]
            y_f_to_g = y[(x >= f) & (x <= g)]
            x_after_h = x[x >= h]
            y_after_h = y[x >= h]

            plt.figure(1)
            plt.plot(x_before_a, y_before_a, color=color, alpha=0.4)
            plt.plot(x_b_to_c, y_b_to_c, color=color, alpha=0.4)
            plt.plot(x_d_to_e, y_d_to_e, color=color, alpha=0.4)
            plt.plot(x_f_to_g, y_f_to_g, color=color, alpha=0.4)
            plt.plot(x_after_h, y_after_h, color=color, alpha=0.4)
            plt.plot(
                contours_sorted_x[number_contours[i][0]],
                contours_sorted_y[number_contours[i][0]],
                color="gold",
                alpha=0.6,
            )
            plt.plot(
                contours_sorted_x[number_contours[i][1]],
                contours_sorted_y[number_contours[i][1]],
                color="gold",
                alpha=0.6,
            )
            plt.plot(
                contours_sorted_x[number_contours[i][2]],
                contours_sorted_y[number_contours[i][2]],
                color="gold",
                alpha=0.6,
            )
            plt.plot(
                contours_sorted_x[number_contours[i][3]],
                contours_sorted_y[number_contours[i][3]],
                color="gold",
                alpha=0.6,
            )

        if len(number_contours[i]) == 5:
            a = contours_sorted_x[number_contours[i][0]][0]
            b = contours_sorted_x[number_contours[i][0]][-1]
            c = contours_sorted_x[number_contours[i][1]][0]
            d = contours_sorted_x[number_contours[i][1]][-1]
            e = contours_sorted_x[number_contours[i][2]][0]
            f = contours_sorted_x[number_contours[i][2]][-1]
            g = contours_sorted_x[number_contours[i][3]][0]
            h = contours_sorted_x[number_contours[i][3]][-1]
            m = contours_sorted_x[number_contours[i][3]][0]
            n = contours_sorted_x[number_contours[i][3]][-1]
            x_before_a = x[x <= a]
            y_before_a = y[x <= a]
            x_b_to_c = x[(x >= b) & (x <= c)]
            y_b_to_c = y[(x >= b) & (x <= c)]
            x_d_to_e = x[(x >= d) & (x <= e)]
            y_d_to_e = y[(x >= d) & (x <= e)]
            x_f_to_g = x[(x >= f) & (x <= g)]
            y_f_to_g = y[(x >= f) & (x <= g)]
            x_h_to_m = x[(x >= h) & (x <= m)]
            y_h_to_m = y[(x >= h) & (x <= m)]
            x_after_n = x[x >= n]
            y_after_n = y[x >= n]

            plt.figure(1)
            plt.plot(x_before_a, y_before_a, color=color, alpha=0.4)
            plt.plot(x_b_to_c, y_b_to_c, color=color, alpha=0.4)
            plt.plot(x_d_to_e, y_d_to_e, color=color, alpha=0.4)
            plt.plot(x_f_to_g, y_f_to_g, color=color, alpha=0.4)
            plt.plot(x_h_to_m, y_h_to_m, color=color, alpha=0.4)
            plt.plot(x_after_n, y_after_n, color=color, alpha=0.4)
            plt.plot(
                contours_sorted_x[number_contours[i][0]],
                contours_sorted_y[number_contours[i][0]],
                color="gold",
                alpha=0.6,
            )
            plt.plot(
                contours_sorted_x[number_contours[i][1]],
                contours_sorted_y[number_contours[i][1]],
                color="gold",
                alpha=0.6,
            )
            plt.plot(
                contours_sorted_x[number_contours[i][2]],
                contours_sorted_y[number_contours[i][2]],
                color="gold",
                alpha=0.6,
            )
            plt.plot(
                contours_sorted_x[number_contours[i][3]],
                contours_sorted_y[number_contours[i][3]],
                color="gold",
                alpha=0.6,
            )
            plt.plot(
                contours_sorted_x[number_contours[i][4]],
                contours_sorted_y[number_contours[i][4]],
                color="gold",
                alpha=0.6,
            )

        if len(number_contours[i]) > 5:
            print(">=6 contours detected")

    plt.figure(1)
    plt.imshow(original_image)
    plt.plot(ex_x_LA, ex_y_LA, color="blue", alpha=0.5)
    plt.plot(ex_x_UA, ex_y_UA, color="blue", alpha=0.5)

    plt.show()


def Curved_Approach_2_3(
    contours_sorted,
    ex_x_LA,
    ex_y_LA,
    ex_x_UA,
    ex_y_UA,
    original_image,
    parameters,
    filter_fasc,
    approach,
):

    start_time = time.time()

    # set parameteres
    tolerance = int(parameters["tolerance"])
    tolerance_to_apo = int(parameters["tolerance_to_apo"])
    coeff_limit = float(parameters["coeff_limit"])
    max_pennation = int(parameters["max_pennation"])
    min_pennation = int(parameters["min_pennation"])

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
            if 0 < coefficients[0] < coeff_limit:
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

                current_fascicle_x, current_fascicle_y, found_fascicle = (
                    find_next_fascicle(
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
                )

                if found_fascicle > 0:
                    label[found_fascicle] = True
                    inner_number_contours.append(found_fascicle)
                else:
                    break

                coefficients = np.polyfit(current_fascicle_x, current_fascicle_y, 2)

                if 0 < coefficients[0] < coeff_limit:
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

        if approach == 2:
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

        if approach == 3:
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

    end_time = time.time()
    total_time = end_time - start_time

    median_length = data["fascicle_length"].median()
    mean_length = data["fascicle_length"].mean()
    median_angle = data["pennation_angle"].median()
    mean_angle = data["pennation_angle"].mean()

    print(total_time)
    print(data)
    print(median_length, mean_length, median_angle, mean_angle)

    colormap = plt.get_cmap("rainbow", len(all_coordsX))

    plt.figure(1)
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

    plt.show()


def Orientation_map(original_image, fas_image, apo_image, g, h):

    start_time = time.time()

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
        numpy.arange(orientationsBoxes["theta"].shape[0]) * boxSizePixels
        + boxSizePixels // 2
    )
    boxCentresX = (
        numpy.arange(orientationsBoxes["theta"].shape[1]) * boxSizePixels
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

    end_time = time.time()
    total_time = end_time - start_time

    print(total_time)
    print(f"Mean angle in degrees: {angle_deg_mean}")
    print(f"Median angle in degrees: {angle_deg_median}")
    print(f"Mean slope: {slope_mean}")
    print(f"Median slope: {slope_median}")
    print(f"Mean angles in degree for 6 parts: {split_angles_deg_mean}")
    print(f"Median angles in degree for 6 parts: {split_angles_deg_median}")

    # figure 1: plot smoothened inter- and extrapolation vectors only for the region between the two aponeuroses
    plt.figure(1)
    plt.imshow(original_image, cmap="Greys_r", vmin=0)
    plt.plot(ex_x_UA, ex_y_UA, color="white")
    plt.plot(ex_x_LA, ex_y_LA, color="white")
    plt.title("smoothened linear interpolation and extrapolation with Rbf")
    plt.quiver(
        boxCentres_grid_X,
        boxCentres_grid_Y,
        di_x_masked_1,
        di_y_masked_1,
        angles="xy",
        scale=0.2,
        scale_units="xy",
        # scale=energyBoxes.ravel(),
        color="r",
        headwidth=0,
        headlength=0,
        headaxislength=1,
    )

    norm = mcolors.Normalize(vmin=np.min(slope), vmax=np.max(slope))

    # figure 2: plot heat map of slopes for region between the two aponeuroses
    plt.figure(2)
    plt.imshow(slope, cmap="viridis", norm=norm, interpolation="none")
    plt.plot(ex_x_LA, ex_y_LA, color="white")
    plt.plot(ex_x_UA, ex_y_UA, color="white")
    plt.colorbar(label="Value")
    plt.title("Matrix Heatmap")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.show()


doCalculations_curved(
    original_image, fas_image, apo_image, parameters, filter_fasc, approach
)
