"""Approach 1"""

import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from curved_fascicles_functions import (
    adapted_contourEdge,
    adapted_filter_fascicles,
    crop,
    do_curves_intersect,
    find_next_fascicle,
)
from curved_fascicles_prep import apo_to_contour, fascicle_to_contour

# load fascicle mask
image = cv2.imread(
    r"C:\Users\carla\Documents\Master_Thesis\Example_Images\FALLMUD\NeilCronin\fascicle_masks\img_00068.tif",
    cv2.IMREAD_UNCHANGED,
)
# load aponeurosis mask
apo_image = cv2.imread(
    r"C:\Users\carla\Documents\Master_Thesis\Example_Images\FALLMUD\NeilCronin\aponeurosis_masks\img_00068.jpg",
    cv2.IMREAD_UNCHANGED,
)
# load ultrasound image
original_image = cv2.imread(
    r"C:\Users\carla\Documents\Master_Thesis\Example_Images\FALLMUD\NeilCronin\images\img_00068.tif",
    cv2.IMREAD_UNCHANGED,
)

# crop all three images in order that they don't have a frame
original_image, image, apo_image = crop(original_image, image, apo_image)

# get sorted fascicle contours
image_gray, contours_sorted = fascicle_to_contour(image)

# get extrapolation of aponeuroses
apo_image_gray, ex_x_LA, ex_y_LA, ex_x_UA, ex_y_UA = apo_to_contour(apo_image)

#### Start of independent function  ####
start_time = time.time()

# Set parameters
tolerance = 10
tolerance_to_apo = 100
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
data = adapted_filter_fascicles(fascicle_data, tolerance_to_apo)

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
