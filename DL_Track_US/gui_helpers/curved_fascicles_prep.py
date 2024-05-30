import cv2
import numpy as np
from do_calculations import contourEdge, sortContours
from scipy.signal import savgol_filter
from skimage.morphology import skeletonize


def fascicle_to_contour(image):

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # define threshold and find contours around fascicles
    _, threshF = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY)
    threshF = threshF.astype("uint8")
    contoursF, hierarchy = cv2.findContours(
        threshF, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

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
            print(
                f"Contour {i} does not have the expected shape: {contour_array.shape}"
            )

    # Now, contours are sorted, and we can sort the list of contours based on the first point
    contours_sorted = sorted(
        contours,
        key=lambda k: (
            (k[0][0], -k[0][1]) if len(k) > 0 else (float("inf"), float("inf"))
        ),
    )

    return image_gray, contoursF, contours_sorted


def apo_to_contour(image):

    apo_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    apo_image_gray = cv2.cvtColor(apo_image_rgb, cv2.COLOR_RGB2GRAY)

    _, thresh = cv2.threshold(
        apo_image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    thresh = thresh.astype("uint8")
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    contours_re = []
    for contour in contours:  # Remove any contours that are very small
        if len(contour) > 600:
            contours_re.append(contour)
    contours = contours_re

    (contours, _) = sortContours(contours)

    contours_re2 = []
    for contour in contours:
        #     cv2.drawContours(mask_apo,[contour],0,255,-1)
        pts = list(contour)
        ptsT = sorted(
            pts, key=lambda k: [k[0][0], k[0][1]]
        )  # Sort each contour based on x values
        allx = []
        ally = []
        for a in range(0, len(ptsT)):
            allx.append(ptsT[a][0, 0])
            ally.append(ptsT[a][0, 1])
        app = np.array(list(zip(allx, ally)))
        contours_re2.append(app)

    # Merge nearby contours
    # countU = 0
    xs1 = []
    xs2 = []
    ys1 = []
    ys2 = []
    maskT = np.zeros(thresh.shape, np.uint8)
    for cnt in contours_re2:
        ys1.append(cnt[0][1])
        ys2.append(cnt[-1][1])
        xs1.append(cnt[0][0])
        xs2.append(cnt[-1][0])
        cv2.drawContours(maskT, [cnt], 0, 255, -1)

    for countU in range(0, len(contours_re2) - 1):
        if (
            xs1[countU + 1] > xs2[countU]
        ):  # Check if x of contour2 is higher than x of contour 1
            y1 = ys2[countU]
            y2 = ys1[countU + 1]
            if y1 - 10 <= y2 <= y1 + 10:
                m = np.vstack((contours_re2[countU], contours_re2[countU + 1]))
                cv2.drawContours(maskT, [m], 0, 255, -1)
        countU += 1

    maskT[maskT > 0] = 1
    skeleton = skeletonize(maskT).astype(np.uint8)
    kernel = np.ones((3, 7), np.uint8)
    dilate = cv2.dilate(skeleton, kernel, iterations=15)
    erode = cv2.erode(dilate, kernel, iterations=10)

    contoursE, hierarchy = cv2.findContours(
        erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    mask_apoE = np.zeros(thresh.shape, np.uint8)

    contoursE = [
        i for i in contoursE if len(i) > 600
    ]  # Remove any contours that are very small

    for contour in contoursE:
        cv2.drawContours(mask_apoE, [contour], 0, 255, -1)
    contoursE, _ = sortContours(contoursE)

    # Only continues beyond this point if 2 aponeuroses can be detected
    if len(contoursE) >= 2:
        # Get the x,y coordinates of the upper/lower edge of the 2 aponeuroses
        upp_x, upp_y = contourEdge("B", contoursE[0])

        if contoursE[1][0, 0, 1] > contoursE[0][0, 0, 1] + 60:
            low_x, low_y = contourEdge("T", contoursE[1])
        else:
            low_x, low_y = contourEdge("T", contoursE[2])

        upp_y_new = savgol_filter(upp_y, 81, 2)  # window size 51, polynomial 3
        low_y_new = savgol_filter(low_y, 81, 2)

    # Compute functions to approximate the shape of the aponeuroses
    zUA = np.polyfit(upp_x, upp_y_new, 2)
    g = np.poly1d(zUA)
    zLA = np.polyfit(low_x, low_y_new, 2)
    h = np.poly1d(zLA)

    mid = (low_x[-1] - low_x[0]) / 2 + low_x[0]  # Find middle
    x1 = np.linspace(-200, 800, 5000)
    # x1 = np.linspace(
    # low_x[0] - 700, low_x[-1] + 700, 10000
    # )  # Extrapolate polynomial fits to either side of the mid-point
    y_UA = g(x1)
    y_LA = h(x1)

    # new_X_UA = np.linspace(
    # mid - 700, mid + 700, 5000
    # )  # Extrapolate x,y data using f function
    new_X_UA = np.linspace(-200, 800, 5000)
    new_Y_UA = g(new_X_UA)
    new_X_LA = np.linspace(-200, 800, 5000)
    # new_X_LA = np.linspace(
    # mid - 700, mid + 700, 5000
    # )  # Extrapolate x,y data using f function
    new_Y_LA = h(new_X_LA)

    return apo_image_rgb, new_X_LA, new_Y_LA, new_X_UA, new_Y_UA
