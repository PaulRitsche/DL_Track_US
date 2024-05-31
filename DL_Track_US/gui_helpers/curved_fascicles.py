"""This is a new file"""

import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from curved_fascicles_functions import (
    adapted_contourEdge,
    adapted_filter_fascicles,
    adapted_filter_fascicles_fast,
    find_next_fascicle,
)
from curved_fascicles_prep import apo_to_contour, fascicle_to_contour

# load image as gray scale image
image = cv2.imread(
    r"C:\Users\carla\Documents\Master_Thesis\Example_Images\FALLMUD\NeilCronin\fascicle_masks\img_00020.tif",
    cv2.IMREAD_UNCHANGED,
)
apo_image = cv2.imread(
    r"C:\Users\carla\Documents\Master_Thesis\Example_Images\FALLMUD\NeilCronin\aponeurosis_masks\img_00020.jpg",
    cv2.IMREAD_UNCHANGED,
)
original_image = cv2.imread(
    r"C:\Users\carla\Documents\Master_Thesis\Example_Images\FALLMUD\NeilCronin\images\img_00020.tif",
    cv2.IMREAD_UNCHANGED,
)


# get sorted fascicle contours
image_gray, contoursF, contours_sorted = fascicle_to_contour(image)

# get extrapolation of aponeuroses
apo_image_gray, ex_x_LA, ex_y_LA, ex_x_UA, ex_y_UA = apo_to_contour(apo_image)

# plot contours
plt.figure(1)
contour_image = cv2.cvtColor(
    image_gray, cv2.COLOR_GRAY2BGR
)  # Convert to BGR for visualization
cv2.drawContours(contour_image, contoursF, -1, (0, 255, 0), 2)  # Draw contours in green
plt.imshow(contour_image)

start_time = time.time()

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
tolerance = 10
all_fascicles_x = []
all_fascicles_y = []

fascicle_data = pd.DataFrame(
    columns=["number_contours", "linear_fit", "coordsX", "coordsY", "coordsXY"]
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
        if 0 < coefficients[0] < 0.000583:
            g = np.poly1d(coefficients)
            ex_current_fascicle_x = np.linspace(
                -200, 800, 5000
            )  # Extrapolate x,y data using f function
            ex_current_fascicle_y = g(ex_current_fascicle_x)
            linear_fit = False
        else:
            coefficients = np.polyfit(current_fascicle_x, current_fascicle_y, 1)
            g = np.poly1d(coefficients)
            ex_current_fascicle_x = np.linspace(
                -200, 800, 5000
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

            if 0 < coefficients[0] < 0.000583:
                g = np.poly1d(coefficients)
                ex_current_fascicle_x = np.linspace(
                    -200, 800, 5000
                )  # Extrapolate x,y data using f function
                ex_current_fascicle_y = g(ex_current_fascicle_x)
                linear_fit = False
            else:
                coefficients = np.polyfit(current_fascicle_x, current_fascicle_y, 1)
                g = np.poly1d(coefficients)
                ex_current_fascicle_x = np.linspace(
                    -200, 800, 5000
                )  # Extrapolate x,y data using f function
                ex_current_fascicle_y = g(ex_current_fascicle_x)
                linear_fit = True

            upper_bound = ex_current_fascicle_y - tolerance
            lower_bound = ex_current_fascicle_y + tolerance

        # Find intersection between fascicle and aponeuroses
        diffU = ex_current_fascicle_y - ex_y_UA
        locU = np.where(diffU == min(diffU, key=abs))[0]
        diffL = ex_current_fascicle_y - ex_y_LA
        locL = np.where(diffL == min(diffL, key=abs))[0]

        # if min(diffU, key=abs) > 80:
        # continue

        # Get coordinates of fascicle between the two aponeuroses
        coordsX = ex_current_fascicle_x[int(locL) : int(locU)]
        coordsY = ex_current_fascicle_y[int(locL) : int(locU)]
        coordsXY = list(zip(coordsX, coordsY))

        all_fascicles_x.append(ex_current_fascicle_x)
        all_fascicles_y.append(ex_current_fascicle_y)
        coefficient_label.append(linear_fit)
        number_contours.append(inner_number_contours)

        fascicle_data_temp = pd.DataFrame(
            {
                "number_contours": [inner_number_contours],
                "linear_fit": linear_fit,
                "coordsX": [coordsX],
                "coordsY": [coordsY],
                "coordsXY": [coordsXY],
            }
        )

        fascicle_data = pd.concat(
            [fascicle_data, fascicle_data_temp], ignore_index=True
        )

tolerance_to_apo = 100
# data = adapted_filter_fascicles(fascicle_data, tolerance_to_apo)
data = adapted_filter_fascicles_fast(fascicle_data, tolerance_to_apo)

# data = filter_fascicles(fascicle_data)
print(data)

end_time = time.time()
total_time = end_time - start_time
print(total_time)

# plot extrapolated fascicles
plt.figure(2)
plt.imshow(apo_image_gray, cmap="gray", alpha=0.5)
plt.imshow(contour_image, alpha=0.5)
for i in range(len(all_fascicles_x)):
    # if coefficient_label[i] is False:
    plt.plot(all_fascicles_x[i], all_fascicles_y[i])
plt.plot(ex_x_LA, ex_y_LA)
plt.plot(ex_x_UA, ex_y_UA)
# plt.show()

for i in range(len(all_fascicles_x)):

    x = all_fascicles_x[i]
    y = all_fascicles_y[i]

    if len(number_contours[i]) == 1:
        a = contours_sorted_x[number_contours[i][0]][0]
        b = contours_sorted_x[number_contours[i][0]][-1]
        x_before_a = x[x <= a]
        y_before_a = y[x <= a]
        x_after_b = x[x >= b]
        y_after_b = y[x >= b]

        plt.figure(3)
        plt.plot(x_before_a, y_before_a, color="red", alpha=0.4)
        plt.plot(x_after_b, y_after_b, color="red", alpha=0.4)

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

        plt.figure(3)
        plt.plot(x_before_a, y_before_a, color="red", alpha=0.4)
        plt.plot(x_b_to_c, y_b_to_c, color="red", alpha=0.4)
        plt.plot(x_after_d, y_after_d, color="red", alpha=0.4)

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

        plt.figure(3)
        plt.plot(x_before_a, y_before_a, color="red", alpha=0.4)
        plt.plot(x_b_to_c, y_b_to_c, color="red", alpha=0.4)
        plt.plot(x_d_to_e, y_d_to_e, color="red", alpha=0.4)
        plt.plot(x_after_f, y_after_f, color="red", alpha=0.4)

    if len(number_contours[i]) > 3:
        print(">=4 contours detected")

plt.figure(3)
plt.imshow(original_image)
plt.plot(ex_x_LA, ex_y_LA, color="blue", alpha=0.5)
plt.plot(ex_x_UA, ex_y_UA, color="blue", alpha=0.5)

# plot extrapolated fascicles and aponeuroses together with original image
plt.figure(4)
plt.imshow(original_image)
for i in range(len(all_fascicles_x)):
    plt.plot(all_fascicles_x[i], all_fascicles_y[i], color="red", alpha=0.4)
plt.plot(ex_x_LA, ex_y_LA, color="blue", alpha=0.5)
plt.plot(ex_x_UA, ex_y_UA, color="blue", alpha=0.5)


# plot filtered fascicles
plt.figure(5)
plt.imshow(original_image)
for row in data.iterrows():
    plt.plot(row[1]["coordsX"], row[1]["coordsY"], color="red", alpha=0.4)
plt.plot(ex_x_LA, ex_y_LA, color="blue", alpha=0.5)
plt.plot(ex_x_UA, ex_y_UA, color="blue", alpha=0.5)

plt.show()
