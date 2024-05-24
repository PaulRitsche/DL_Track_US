"""This is a new file"""

import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from curved_fascicles_functions import contourEdge, find_next_fascicle

# load image as gray scale image
image = cv2.imread(
    r"C:\Users\carla\Documents\Master_Thesis\Example_Images\FALLMUD\NeilCronin\fascicle_masks\img_00001.tif",
    cv2.IMREAD_UNCHANGED,
)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

# define threshold and find contours around fascicles
_, threshF = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY)
threshF = threshF.astype("uint8")
contoursF, hierarchy = cv2.findContours(
    threshF, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

# plot contours
plt.figure(1)
contour_image = cv2.cvtColor(
    image_gray, cv2.COLOR_GRAY2BGR
)  # Convert to BGR for visualization
cv2.drawContours(contour_image, contoursF, -1, (0, 255, 0), 2)  # Draw contours in green
plt.imshow(contour_image)

start_time = time.time()

# convert contours into a list
contours = list(contoursF)

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
        print(f"Contour {i} does not have the expected shape: {contour_array.shape}")

# Now, contours are sorted, and we can sort the list of contours based on the first point
contours_sorted = sorted(
    contours,
    key=lambda k: (k[0][0], -k[0][1]) if len(k) > 0 else (float("inf"), float("inf")),
)

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
plt.imshow(image_rgb)
for i in range(len(all_fascicles_x)):
    plt.plot(all_fascicles_x[i], all_fascicles_y[i])
plt.show()
