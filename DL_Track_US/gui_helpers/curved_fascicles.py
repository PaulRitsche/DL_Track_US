"""This is a new file"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from curved_fascicles_functions import contourEdge, find_next_fascicle

# load image as gray scale image
image = cv2.imread(
    r"C:\Users\carla\Documents\Master_Thesis\Example_Images\FALLMUD\NeilCronin\fascicle_masks\img_00026.tif",
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
plt.axis("off")  # Hide the axis
# plt.show()

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
    key=lambda k: (k[0][0], k[0][1]) if len(k) > 0 else (float("inf"), float("inf")),
)

# get upper contours of each contour
contours_sorted_x = []
contours_sorted_y = []
for i in range(len(contours_sorted)):
    contours_sorted[i][0], contours_sorted[i][1] = contourEdge("B", contours_sorted[i])
    contours_sorted_x.append(contours_sorted[i][0])
    contours_sorted_y.append(contours_sorted[i][1])

# get contour of the first fascicle (the most left and lowest fascicle contour)
current_fascicle_x = contours_sorted_x[1]
current_fascicle_y = contours_sorted_y[1]

# initialize label as false for each fascicle within the contours and set label of first fascicle contour as true as this one is already in use
label = {x: False for x in range(len(contours_sorted))}
label[0] = True

# calculate second polynomial fit and extrapolate function for first fascicle
coefficients = np.polyfit(current_fascicle_x, current_fascicle_y, 1)
g = np.poly1d(coefficients)
ex_current_fascicle_x = np.linspace(
    0, 512, 5000
)  # Extrapolate x,y data using f function
ex_current_fascicle_y = g(ex_current_fascicle_x)
print(coefficients)

# find next fascicle
tolerance = 10
upper_bound = ex_current_fascicle_y - tolerance
lower_bound = ex_current_fascicle_y + tolerance

# plot
plt.figure(2)
plt.imshow(image_rgb)
plt.plot(ex_current_fascicle_x, ex_current_fascicle_y)
plt.plot(ex_current_fascicle_x, upper_bound)
plt.plot(ex_current_fascicle_x, lower_bound)
# plt.show()

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

    coefficients = np.polyfit(current_fascicle_x, current_fascicle_y, 2)
    g = np.poly1d(coefficients)
    ex_current_fascicle_x = np.linspace(
        0, 512, 5000
    )  # Extrapolate x,y data using f function
    ex_current_fascicle_y = g(ex_current_fascicle_x)

    # plot
    plt.figure()
    plt.imshow(image_rgb)
    plt.plot(ex_current_fascicle_x, ex_current_fascicle_y)

    upper_bound = ex_current_fascicle_y - tolerance
    lower_bound = ex_current_fascicle_y + tolerance


# new_x, new_y, found_fascicle = find_next_fascicle(
# contours_sorted,
# contours_sorted_x,
# contours_sorted_y,
# new_x,
# new_y,
# new_x_first_fascicle,
# upper_bound,
# lower_bound,
# )

# label[found_fascicle] = True

# coefficients = np.polyfit(new_x, new_y, 2)
# g = np.poly1d(coefficients)
# new_x_first_fascicle = np.linspace(
# 0, 512, 5000
# )  # Extrapolate x,y data using f function
# new_y_first_fascicle = g(new_x_first_fascicle)
# print(coefficients)

print(label)
# plot
# plt.figure(4)
# plt.imshow(image_rgb)
# plt.plot(new_x_first_fascicle, new_y_first_fascicle)
plt.show()
