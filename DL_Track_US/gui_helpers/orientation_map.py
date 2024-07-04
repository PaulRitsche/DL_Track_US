import math
import time

import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy
import numpy as np
import orientationpy
from curved_fascicles_prep import apo_to_contour_orientation_map
from scipy.interpolate import Rbf
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

# load image as gray scale image
image = cv2.imread(
    r"C:\Users\carla\Documents\Master_Thesis\Example_Images\FALLMUD\NeilCronin\fascicle_masks\img_00027.tif",
    cv2.IMREAD_UNCHANGED,
)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

apo_image = cv2.imread(
    r"C:\Users\carla\Documents\Master_Thesis\Example_Images\FALLMUD\NeilCronin\aponeurosis_masks\img_00027.jpg",
    cv2.IMREAD_UNCHANGED,
)
apo_image_gray, ex_x_LA, ex_y_LA, ex_x_UA, ex_y_UA = apo_to_contour_orientation_map(
    apo_image
)
width = apo_image_gray.shape[1]

start_time = time.time()

for n, mode in enumerate(["finite_difference", "gaussian", "splines"]):
    Gy, Gx = orientationpy.computeGradient(image_gray, mode=mode)

structureTensor = orientationpy.computeStructureTensor([Gy, Gx], sigma=2)

orientations = orientationpy.computeOrientation(
    structureTensor, computeEnergy=True, computeCoherency=True
)

# The coherency measures how strongly aligned the image is locally
orientations["coherency"][image == 0] = 0

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

valid_points = np.array(list(zip(valid_boxCentres_grid_X, valid_boxCentres_grid_Y)))

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

window_length = 5  # Length of the filter window (must be odd)
polyorder = 2  # Order of the polynomial fit

# smoothened extra- and interpolated values with savgol filter
# di_x_smooth = savgol_filter(di_x, window_length=window_length, polyorder=polyorder)
# di_y_smooth = savgol_filter(di_y, window_length=window_length, polyorder=polyorder)

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
# split_arrays = np.array_split(slope_non_zero_rows, 3, axis=0) # uncomment for horizontal split into three parts, comment step 2 and 3 if no vertical split is needed

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

# figure 1: plot orientation vectors calculated from orientationpy
plt.figure(1)
plt.title("Local orientation vector in boxes")
plt.imshow(image_gray, cmap="Greys_r", vmin=0)

# Warning, matplotlib is XY convention, not YX!
plt.quiver(
    boxCentresX,
    boxCentresY,
    boxVectorsYX[1],
    boxVectorsYX[0],
    angles="xy",
    scale=0.2,
    scale_units="xy",
    # scale=energyBoxes.ravel(),
    color="r",
    headwidth=0,
    headlength=0,
    headaxislength=1,
)

# figure 2: plot inter- and extrapolated vectors for complete image
plt.figure(2)
plt.imshow(image_gray, cmap="Greys_r", vmin=0)
plt.plot(ex_x_UA, ex_y_UA, color="white")
plt.plot(ex_x_LA, ex_y_LA, color="white")
plt.title("linear interpolation and extrapolation with Rbf")
plt.quiver(
    boxCentres_grid_X,
    boxCentres_grid_Y,
    di_x,
    di_y,
    angles="xy",
    scale=0.2,
    scale_units="xy",
    # scale=energyBoxes.ravel(),
    color="r",
    headwidth=0,
    headlength=0,
    headaxislength=1,
)

# figure 3: plot smoothened inter- and extrapolated vectors for complete image
plt.figure(3)
plt.imshow(image_gray, cmap="Greys_r", vmin=0)
plt.plot(ex_x_UA, ex_y_UA, color="white")
plt.plot(ex_x_LA, ex_y_LA, color="white")
plt.title("smoothened linear interpolation and extrapolation with Rbf")
plt.quiver(
    boxCentres_grid_X,
    boxCentres_grid_Y,
    di_x_smooth,
    di_y_smooth,
    angles="xy",
    scale=0.2,
    scale_units="xy",
    # scale=energyBoxes.ravel(),
    color="r",
    headwidth=0,
    headlength=0,
    headaxislength=1,
)

# figure 4: plot smoothened inter- and extrapolation vectors only for the region between the two aponeuroses
plt.figure(4)
plt.imshow(image_gray, cmap="Greys_r", vmin=0)
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

# figure 5: plot heat map of slopes for region between the two aponeuroses
plt.figure(5)
plt.imshow(slope, cmap="viridis", norm=norm, interpolation="none")
plt.plot(ex_x_LA, ex_y_LA, color="white")
plt.plot(ex_x_UA, ex_y_UA, color="white")
plt.colorbar(label="Value")
plt.title("Matrix Heatmap")
plt.xlabel("Column")
plt.ylabel("Row")
plt.show()
