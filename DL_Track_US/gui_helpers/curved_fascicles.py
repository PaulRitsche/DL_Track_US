"""This is a new file"""

import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from curved_fascicles_functions import (
    contourEdge,
    fascicle_to_contour,
    find_next_fascicle,
)

# load image as gray scale image
image = cv2.imread(
    r"C:\Users\carla\Documents\Master_Thesis\Example_Images\FALLMUD\NeilCronin\fascicle_masks\img_00014.tif",
    cv2.IMREAD_UNCHANGED,
)

# get sorted fascicle contours
image_gray, contoursF, contours_sorted = fascicle_to_contour(image)

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
    contours_sorted[i][0], contours_sorted[i][1] = contourEdge("B", contours_sorted[i])
    contours_sorted_x.append(contours_sorted[i][0])
    contours_sorted_y.append(contours_sorted[i][1])

# initialize some important variables
label = {x: False for x in range(len(contours_sorted))}
tolerance = 6
all_fascicles_x = []
all_fascicles_y = []

# calculate merged fascicle edges
for i in range(len(contours_sorted)):

    if label[i] is False:
        # get upper edge contour of starting fascicle
        current_fascicle_x = contours_sorted_x[i]
        current_fascicle_y = contours_sorted_y[i]

        # set label to true as fascicle is used
        label[i] = True

        # calculate second polynomial coefficients
        coefficients = np.polyfit(current_fascicle_x, current_fascicle_y, 2)

        # depending on coefficients edge gets extrapolated as first or second order polynomial
        if -0.000327 < coefficients[0] < 0.000583:
            g = np.poly1d(coefficients)
            ex_current_fascicle_x = np.linspace(
                0, 512, 5000
            )  # Extrapolate x,y data using f function
            ex_current_fascicle_y = g(ex_current_fascicle_x)
            print(coefficients)
            print(coefficients[0])
        else:
            coefficients = np.polyfit(current_fascicle_x, current_fascicle_y, 1)
            g = np.poly1d(coefficients)
            ex_current_fascicle_x = np.linspace(
                0, 512, 5000
            )  # Extrapolate x,y data using f function
            ex_current_fascicle_y = g(ex_current_fascicle_x)
            print(coefficients)
            print(coefficients[0])

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
            )

            if found_fascicle > 0:
                label[found_fascicle] = True
            else:
                break

            coefficients = np.polyfit(current_fascicle_x, current_fascicle_y, 2)
            g = np.poly1d(coefficients)
            ex_current_fascicle_x = np.linspace(
                0, 512, 5000
            )  # Extrapolate x,y data using f function
            ex_current_fascicle_y = g(ex_current_fascicle_x)

            upper_bound = ex_current_fascicle_y - tolerance
            lower_bound = ex_current_fascicle_y + tolerance

        all_fascicles_x.append(ex_current_fascicle_x)
        all_fascicles_y.append(ex_current_fascicle_y)

end_time = time.time()
total_time = end_time - start_time
print(total_time)

# plot extrapolated fascicles
plt.figure(2)
plt.imshow(contour_image)
for i in range(len(all_fascicles_x)):
    plt.plot(all_fascicles_x[i], all_fascicles_y[i])
plt.show()
