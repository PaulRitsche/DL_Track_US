"""Approach 2"""

import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from curved_fascicles_functions import (
    adapted_contourEdge,
    adapted_filter_fascicles,
    find_next_fascicle,
)
from curved_fascicles_prep import apo_to_contour, fascicle_to_contour
from matplotlib.patches import Rectangle

# load image as gray scale image
image = cv2.imread(
    r"C:\Users\carla\Documents\Master_Thesis\Example_Images\FALLMUD\NeilCronin\fascicle_masks\img_00004.tif",
    cv2.IMREAD_UNCHANGED,
)
apo_image = cv2.imread(
    r"C:\Users\carla\Documents\Master_Thesis\Example_Images\FALLMUD\NeilCronin\aponeurosis_masks\img_00004.jpg",
    cv2.IMREAD_UNCHANGED,
)
original_image = cv2.imread(
    r"C:\Users\carla\Documents\Master_Thesis\Example_Images\FALLMUD\NeilCronin\images\img_00004.tif",
    cv2.IMREAD_UNCHANGED,
)

# get sorted fascicle contours
image_gray, contoursF, contours_sorted = fascicle_to_contour(image)

# get extrapolation of aponeuroses
apo_image_gray, ex_x_LA, ex_y_LA, ex_x_UA, ex_y_UA = apo_to_contour(apo_image)

# get contours around detected fascicles
contour_image = cv2.cvtColor(
    image_gray, cv2.COLOR_GRAY2BGR
)  # Convert to BGR for visualization
cv2.drawContours(contour_image, contoursF, -1, (0, 255, 0), 2)  # Draw contours in green

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
    columns=[
        "number_contours",
        "linear_fit",
        "coordsX",
        "coordsY",
        "coordsX_combined",
        "coordsY_combined",
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
                # "coordsXY": None
            }
        )

        fascicle_data = pd.concat(
            [fascicle_data, fascicle_data_temp], ignore_index=True
        )

number_contours = list(fascicle_data["number_contours"])

for i in range(len(number_contours)):

    coefficients = np.polyfit(
        contours_sorted_x[number_contours[i][0]],
        contours_sorted_y[number_contours[i][0]],
        1,
    )
    g = np.poly1d(coefficients)
    ex_current_fascicle_x = np.linspace(
        -200, contours_sorted_x[number_contours[i][0]][0]
    )
    ex_current_fascicle_y = g(ex_current_fascicle_x)

    coordsX = [list(ex_current_fascicle_x)]
    coordsY = [list(ex_current_fascicle_y)]

    coordsX.append(contours_sorted_x[number_contours[i][0]])
    coordsY.append(contours_sorted_y[number_contours[i][0]])

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

    coefficients = np.polyfit(
        contours_sorted_x[number_contours[i][-1]],
        contours_sorted_y[number_contours[i][-1]],
        1,
    )
    g = np.poly1d(coefficients)
    ex_current_fascicle_x_2 = np.linspace(
        contours_sorted_x[number_contours[i][-1]][-1], 800
    )
    ex_current_fascicle_y_2 = g(ex_current_fascicle_x_2)

    coordsX.append(list(ex_current_fascicle_x_2))
    coordsY.append(list(ex_current_fascicle_y_2))

    coordsX_combined = []
    coordsX_combined = [item for sublist in coordsX for item in sublist]

    coordsY_combined = []
    coordsY_combined = [item for sublist in coordsY for item in sublist]

    # diffU = coordsX_combined - ex_y_UA
    # locU = np.where(diffU == min(diffU, key=abs))[0]
    # diffL = coordsY_combined - ex_y_LA
    # locL = np.where(diffL == min(diffL, key=abs))[0]

    # coordsX_combined = coordsX_combined[int(locL) : int(locU)]
    # coordsY_combined = coordsY_combined[int(locL) : int(locU)]
    # coordsXY = list(zip(coordsX_combined, coordsY_combined))

    fascicle_data.at[i, "coordsX"] = coordsX
    fascicle_data.at[i, "coordsY"] = coordsY
    fascicle_data.at[i, "coordsX_combined"] = coordsX_combined
    fascicle_data.at[i, "coordsY_combined"] = coordsY_combined
    # fascicle_data.at[i, "coordsXY"] = coordsXY

print(fascicle_data)

plt.figure(3)
plt.imshow(contour_image)
# plt.imshow(original_image)
for row in fascicle_data.iterrows():
    plt.plot(
        row[1]["coordsX_combined"], row[1]["coordsY_combined"], color="red", alpha=0.4
    )
plt.plot(ex_x_LA, ex_y_LA)
plt.plot(ex_x_UA, ex_y_UA)


plt.figure(1)
plt.imshow(apo_image_gray, cmap="gray", alpha=0.5)
plt.imshow(contour_image, alpha=0.5)
for i in range(len(all_fascicles_x)):
    # if coefficient_label[i] is False:
    plt.plot(all_fascicles_x[i], all_fascicles_y[i])
plt.plot(ex_x_LA, ex_y_LA)
plt.plot(ex_x_UA, ex_y_UA)

plt.show()
