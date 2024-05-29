"""This is a new file"""

import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from curved_fascicles_functions import adapted_contourEdge, find_next_fascicle
from curved_fascicles_prep import apo_to_contour, fascicle_to_contour

# load image as gray scale image
image = cv2.imread(
    r"C:\Users\carla\Documents\Master_Thesis\Example_Images\FALLMUD\NeilCronin\fascicle_masks\img_00012.tif",
    cv2.IMREAD_UNCHANGED,
)
apo_image = cv2.imread(
    r"C:\Users\carla\Documents\Master_Thesis\Example_Images\FALLMUD\NeilCronin\aponeurosis_masks\img_00012.jpg",
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
tolerance = 10
all_fascicles_x = []
all_fascicles_y = []

# calculate merged fascicle edges
for i in range(len(contours_sorted)):

    if label[i] is False and len(contours_sorted_x[i]) > 1:
        # get upper edge contour of starting fascicle
        current_fascicle_x = contours_sorted_x[i]
        current_fascicle_y = contours_sorted_y[i]

        print("new loop")

        # set label to true as fascicle is used
        label[i] = True
        linear_fit = False

        # calculate second polynomial coefficients
        coefficients = np.polyfit(current_fascicle_x, current_fascicle_y, 2)

        # depending on coefficients edge gets extrapolated as first or second order polynomial
        if 0 < coefficients[0] < 0.000583:  # -0.000327 < coefficients[0] < 0.000583:
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

            print(current_fascicle_x)
            print(len(current_fascicle_x))

            if found_fascicle > 0:
                label[found_fascicle] = True
            else:
                break

            coefficients = np.polyfit(current_fascicle_x, current_fascicle_y, 2)

            if (
                0 < coefficients[0] < 0.000583
            ):  # -0.000327 < coefficients[0] < 0.000583: #and len(current_fascicle_x)>40:
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

end_time = time.time()
total_time = end_time - start_time
print(total_time)

print(coefficient_label)

# plot extrapolated fascicles
plt.figure(2)
plt.imshow(apo_image_gray, cmap="gray", alpha=0.5)
plt.imshow(contour_image, alpha=0.5)
for i in range(len(all_fascicles_x)):
    # if coefficient_label[i] is False:
    plt.plot(all_fascicles_x[i], all_fascicles_y[i])
plt.plot(ex_x_LA, ex_y_LA)
plt.plot(ex_x_UA, ex_y_UA)
plt.show()
