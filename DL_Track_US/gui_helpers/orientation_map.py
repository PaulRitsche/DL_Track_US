import cv2
import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy
import orientationpy
import tifffile

# load image as gray scale image
image = cv2.imread(
    r"C:\Users\carla\Documents\Master_Thesis\Example_Images\FALLMUD\NeilCronin\fascicle_masks\img_00014.tif",
    cv2.IMREAD_UNCHANGED,
)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

plt.figure(1)
for n, mode in enumerate(["finite_difference", "gaussian", "splines"]):
    Gy, Gx = orientationpy.computeGradient(image_gray, mode=mode)
    plt.subplot(2, 3, n + 1)
    plt.title(f"{mode}-Gy")
    plt.imshow(Gy, cmap="coolwarm", vmin=-64, vmax=64)

    plt.subplot(2, 3, 3 + n + 1)
    plt.title(f"{mode}-Gx")
    plt.imshow(Gx, cmap="coolwarm", vmin=-64, vmax=64)

structureTensor = orientationpy.computeStructureTensor([Gy, Gx], sigma=2)

orientations = orientationpy.computeOrientation(
    structureTensor, computeEnergy=True, computeCoherency=True
)

plt.figure(2, figsize=(10, 4))

# The energy represents how strong the orientation signal is
plt.subplot(1, 2, 1)
plt.imshow(orientations["energy"] / orientations["energy"].max(), vmin=0, vmax=1)
plt.colorbar(shrink=0.7)
plt.title("Energy Normalised")

# The coherency measures how strongly aligned the image is locally
orientations["coherency"][image == 0] = 0

plt.subplot(1, 2, 2)
plt.imshow(orientations["coherency"], vmin=0, vmax=1)
plt.title("Coherency")
plt.colorbar(shrink=0.7)
plt.tight_layout()

plt.figure(3)
try:
    plt.suptitle("Overlay with orientation")
    plt.title(
        "Greyscale image with HSV orientations overlaid\nwith transparency as coherency"
    )
    plt.imshow(image_gray, cmap="Greys_r", vmin=0)
    plt.imshow(
        orientations["theta"],
        cmap="hsv",
        alpha=orientations["coherency"] / (2 * orientations["coherency"].max()),
        vmin=-90,
        vmax=90,
    )

    plt.colorbar(shrink=0.7)
except:
    print("Didn't manage to make the plot :(")

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

# Compute X and Y components of the vector
boxVectorsYX = orientationpy.anglesToVectors(orientationsBoxes)

# Vectors with low energy reset
boxVectorsYX[:, orientationsBoxes["energy"] < 0.05] = 0.0

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
plt.figure(5)
plt.imshow(image_gray, cmap="Greys_r")
plt.suptitle("Original image")
plt.show()
