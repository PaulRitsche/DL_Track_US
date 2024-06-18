import cv2
import matplotlib
import matplotlib.colors
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy
import numpy as np
import orientationpy
import tifffile
from curved_fascicles_prep import apo_to_contour
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, Rbf, griddata
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

# load image as gray scale image
image = cv2.imread(
    r"C:\Users\carla\Documents\Master_Thesis\Example_Images\FALLMUD\NeilCronin\fascicle_masks\img_00014.tif",
    cv2.IMREAD_UNCHANGED,
)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

apo_image = cv2.imread(
    r"C:\Users\carla\Documents\Master_Thesis\Example_Images\FALLMUD\NeilCronin\aponeurosis_masks\img_00014.jpg",
    cv2.IMREAD_UNCHANGED,
)
apo_image_gray, ex_x_LA, ex_y_LA, ex_x_UA, ex_y_UA = apo_to_contour(apo_image)

# plt.figure(1)
for n, mode in enumerate(["finite_difference", "gaussian", "splines"]):
    Gy, Gx = orientationpy.computeGradient(image_gray, mode=mode)
    # plt.subplot(2, 3, n + 1)
    # plt.title(f"{mode}-Gy")
    # plt.imshow(Gy, cmap="coolwarm", vmin=-64, vmax=64)

    # plt.subplot(2, 3, 3 + n + 1)
    # plt.title(f"{mode}-Gx")
    # plt.imshow(Gx, cmap="coolwarm", vmin=-64, vmax=64)

structureTensor = orientationpy.computeStructureTensor([Gy, Gx], sigma=2)

orientations = orientationpy.computeOrientation(
    structureTensor, computeEnergy=True, computeCoherency=True
)

# plt.figure(2, figsize=(10, 4))

# The energy represents how strong the orientation signal is
# plt.subplot(1, 2, 1)
# plt.imshow(orientations["energy"] / orientations["energy"].max(), vmin=0, vmax=1)
# plt.colorbar(shrink=0.7)
# plt.title("Energy Normalised")

# The coherency measures how strongly aligned the image is locally
orientations["coherency"][image == 0] = 0

# plt.subplot(1, 2, 2)
# plt.imshow(orientations["coherency"], vmin=0, vmax=1)
# plt.title("Coherency")
# plt.colorbar(shrink=0.7)
# plt.tight_layout()

# plt.figure(3)
# try:
# plt.suptitle("Overlay with orientation")
# plt.title(
# "Greyscale image with HSV orientations overlaid\nwith transparency as coherency"
# )
# plt.imshow(image_gray, cmap="Greys_r", vmin=0)
# plt.imshow(
# orientations["theta"],
# cmap="hsv",
# alpha=orientations["coherency"] / (2 * orientations["coherency"].max()),
# vmin=-90,
# vmax=90,
# )

# plt.colorbar(shrink=0.7)
# except:
# print("Didn't manage to make the plot :(")

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

boxCentres_grid = np.meshgrid(boxCentresX, boxCentresY)
boxCentres_grid_X = boxCentres_grid[0]
boxCentres_grid_Y = boxCentres_grid[1]
boxCentres_grid_X = [item for sublist in boxCentres_grid_X for item in sublist]
boxCentres_grid_Y = [item for sublist in boxCentres_grid_Y for item in sublist]

size_x = boxCentresX.shape[0]
size_y = boxCentresY.shape[0]

# Compute X and Y components of the vector
boxVectorsYX = orientationpy.anglesToVectors(orientationsBoxes)

# Vectors with low energy reset
boxVectorsYX[:, orientationsBoxes["energy"] < 0.05] = 0.0

boxVectorsYX[:, orientationsBoxes["theta"] > 50] = 0.0
# boxVectorsYX[:, orientationsBoxes["theta"] < 10] = 0.0

plt.figure(4)
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
# plt.show()

# Show image
# plt.figure(5)
# plt.imshow(image_gray, cmap="Greys_r")
# plt.suptitle("Original image")
# plt.show()

# boxVectorsYX = [list(zip(x, y)) for x, y in zip(boxVectorsYX[0], boxVectorsYX[1])]
boxVectorsX = [item for sublist in boxVectorsYX[1] for item in sublist]
boxVectorsY = [item for sublist in boxVectorsYX[0] for item in sublist]

mask = (np.array(boxVectorsX) != 0) & (np.array(boxVectorsY) != 0)

valid_boxCentres_grid_X = np.array(boxCentres_grid_X)[mask]
valid_boxCentres_grid_Y = np.array(boxCentres_grid_Y)[mask]
valid_boxVectorsX = np.array(boxVectorsX)[mask]
valid_boxVectorsY = np.array(boxVectorsY)[mask]

valid_points = np.array(list(zip(valid_boxCentres_grid_X, valid_boxCentres_grid_Y)))

# interpolation with griddata, methods: linear, cubic, nearest
grid_x = griddata(
    valid_points,
    valid_boxVectorsX,
    (boxCentres_grid_X, boxCentres_grid_Y),
    method="linear",
)
grid_y = griddata(
    valid_points,
    valid_boxVectorsY,
    (boxCentres_grid_X, boxCentres_grid_Y),
    method="linear",
)

# Plotting the interpolated x- and y-component as an arrow
# plt.figure(6)
# plt.imshow(image_gray, cmap="Greys_r", vmin=0)
# plt.title("linear interpolation with griddata")
# plt.quiver(
# boxCentres_grid_X,
# boxCentres_grid_Y,
# grid_x,
# grid_y,
# angles="xy",
# scale=0.2,
# scale_units="xy",
## scale=energyBoxes.ravel(),
# color="r",
# headwidth=0,
# headlength=0,
# headaxislength=1,
# )

# interpolation and extrapolation with rbf
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

di_x = grid_x_rbf(boxCentres_grid_X, boxCentres_grid_Y)
di_y = grid_y_rbf(boxCentres_grid_X, boxCentres_grid_Y)

window_length = 5  # Length of the filter window (must be odd)
polyorder = 2  # Order of the polynomial fit

# di_x_smooth = savgol_filter(di_x, window_length=window_length, polyorder=polyorder)
# di_y_smooth = savgol_filter(di_y, window_length=window_length, polyorder=polyorder)

di_x_smooth = gaussian_filter1d(di_x, sigma=1)
di_y_smooth = gaussian_filter1d(di_y, sigma=1)

plt.figure(7)
plt.imshow(image_gray, cmap="Greys_r", vmin=0)
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

plt.figure(8)
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

# interpolation with NearestNDInterpolator, nearest neighbour method
nearest_inter_x = NearestNDInterpolator(valid_points, valid_boxVectorsX)
nearest_inter_y = NearestNDInterpolator(valid_points, valid_boxVectorsY)

zi_nearest_x = nearest_inter_x(boxCentres_grid_X, boxCentres_grid_Y)
zi_nearest_y = nearest_inter_y(boxCentres_grid_X, boxCentres_grid_Y)

# zi_nearest_x_smooth = savgol_filter(
# zi_nearest_x, window_length=window_length, polyorder=polyorder
# )
# zi_nearest_y_smooth = savgol_filter(
# zi_nearest_y, window_length=window_length, polyorder=polyorder
# )

zi_nearest_x_smooth = gaussian_filter1d(zi_nearest_x, sigma=1)
zi_nearest_y_smooth = gaussian_filter1d(zi_nearest_y, sigma=1)

plt.figure(9)
plt.imshow(image_gray, cmap="Greys_r", vmin=0)
plt.plot(ex_x_LA, ex_y_LA, color="white")
plt.plot(ex_x_UA, ex_y_UA, color="white")
plt.title(
    "nearest neighbour interpolation and extrapolation with NearestNDInterpolator"
)
plt.quiver(
    boxCentres_grid_X,
    boxCentres_grid_Y,
    zi_nearest_x,
    zi_nearest_y,
    angles="xy",
    scale=0.2,
    scale_units="xy",
    # scale=energyBoxes.ravel(),
    color="r",
    headwidth=0,
    headlength=0,
    headaxislength=1,
)

plt.figure(10)
plt.imshow(image_gray, cmap="Greys_r", vmin=0)
plt.plot(ex_x_LA, ex_y_LA, color="white")
plt.plot(ex_x_UA, ex_y_UA, color="white")
plt.title(
    "nearest neighbour interpolation and extrapolation with NearestNDInterpolator"
)
plt.quiver(
    boxCentres_grid_X,
    boxCentres_grid_Y,
    zi_nearest_x_smooth,
    zi_nearest_y_smooth,
    angles="xy",
    scale=0.2,
    scale_units="xy",
    # scale=energyBoxes.ravel(),
    color="r",
    headwidth=0,
    headlength=0,
    headaxislength=1,
)

# interpolation with LinearNDInterpolator, method: linear
linear_inter_x = LinearNDInterpolator(
    valid_points, valid_boxVectorsX
)  # fill_value=np.nan-> define which values should be filled to points outside
linear_inter_y = LinearNDInterpolator(valid_points, valid_boxVectorsY)

zi_linear_x = linear_inter_x(boxCentres_grid_X, boxCentres_grid_Y)
zi_linear_y = linear_inter_y(boxCentres_grid_X, boxCentres_grid_Y)

# plt.figure(11)
# plt.imshow(image_gray, cmap="Greys_r", vmin=0)
# plt.title("linear interpolation with LinearNDInterpolator")
# plt.quiver(
# boxCentres_grid_X,
# boxCentres_grid_Y,
# zi_linear_x,
# zi_linear_y,
# angles="xy",
# scale=0.2,
# scale_units="xy",
## scale=energyBoxes.ravel(),
# color="r",
# headwidth=0,
# headlength=0,
# headaxislength=1,
# )

# get slope
slope = [((-1) * di_y_smooth[i]) / di_x_smooth[i] for i in range(len(di_x))]
slope = np.array(slope).reshape(size_x, size_y)
slope = np.repeat(np.repeat(slope, boxSizePixels, axis=0), boxSizePixels, axis=1)

norm = mcolors.Normalize(vmin=np.min(slope), vmax=np.max(slope))

plt.figure(12)
plt.imshow(slope, cmap="viridis", norm=norm, interpolation="none")
plt.plot(ex_x_LA, ex_y_LA, color="white")
plt.plot(ex_x_UA, ex_y_UA, color="white")
plt.colorbar(label="Value")
plt.title("Matrix Heatmap")
plt.xlabel("Column")
plt.ylabel("Row")
plt.show()
