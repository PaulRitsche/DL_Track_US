"""This is a new file"""

import os

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

# get upper contour of first fascicle (the most left one)
first_fascicle = contours_sorted[0]
x_first_fascicle, y_first_fascicle = contourEdge("B", first_fascicle)

# get upper contours of each contour
contours_sorted_x = []
contours_sorted_y = []
for i in range(len(contours_sorted)):
    contours_sorted[i][0], contours_sorted[i][1] = contourEdge("B", contours_sorted[i])
    contours_sorted_x.append(contours_sorted[i][0])
    contours_sorted_y.append(contours_sorted[i][1])

# calculate second polynomial fit and extrapolate function for first fascicle
coefficients = np.polyfit(x_first_fascicle, y_first_fascicle, 2)
g = np.poly1d(coefficients)
new_x_first_fascicle = np.linspace(
    0, 512, 5000
)  # Extrapolate x,y data using f function
new_y_first_fascicle = g(new_x_first_fascicle)
print(coefficients)

# find next fascicle
tolerance = 10
upper_bound = new_y_first_fascicle - tolerance
lower_bound = new_y_first_fascicle + tolerance

# plot
plt.figure(2)
plt.imshow(image_rgb)
plt.plot(new_x_first_fascicle, new_y_first_fascicle)
plt.plot(new_x_first_fascicle, upper_bound)
plt.plot(new_x_first_fascicle, lower_bound)
# plt.show()

new_x, new_y = find_next_fascicle(
    contours_sorted,
    contours_sorted_x,
    contours_sorted_y,
    x_first_fascicle,
    y_first_fascicle,
    new_x_first_fascicle,
    upper_bound,
    lower_bound,
)

coefficients = np.polyfit(new_x, new_y, 2)
g = np.poly1d(coefficients)
new_x_first_fascicle = np.linspace(
    0, 512, 5000
)  # Extrapolate x,y data using f function
new_y_first_fascicle = g(new_x_first_fascicle)
print(coefficients)

# plot
plt.figure(3)
plt.imshow(image_rgb)
plt.plot(new_x_first_fascicle, new_y_first_fascicle)

upper_bound = new_y_first_fascicle - tolerance
lower_bound = new_y_first_fascicle + tolerance
found_fascicle = 0

new_x, new_y = find_next_fascicle(
    contours_sorted,
    contours_sorted_x,
    contours_sorted_y,
    new_x,
    new_y,
    new_x_first_fascicle,
    upper_bound,
    lower_bound,
)

coefficients = np.polyfit(new_x, new_y, 2)
g = np.poly1d(coefficients)
new_x_first_fascicle = np.linspace(
    0, 512, 5000
)  # Extrapolate x,y data using f function
new_y_first_fascicle = g(new_x_first_fascicle)
print(coefficients)

# plot
plt.figure(4)
plt.imshow(image_rgb)
plt.plot(new_x_first_fascicle, new_y_first_fascicle)
plt.show()
