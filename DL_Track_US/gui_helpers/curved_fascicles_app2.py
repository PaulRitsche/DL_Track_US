"""Approach 2"""

import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from DL_Track_US.gui_helpers.curved_fascicles_functions import (
    adapted_contourEdge,
    adapted_filter_fascicles,
    crop,
    do_curves_intersect,
    find_next_fascicle,
)
from curved_fascicles_prep import apo_to_contour, fascicle_to_contour

# load image as gray scale image
image = cv2.imread(
    r"C:\Users\carla\Documents\Master_Thesis\Example_Images\FALLMUD\NeilCronin\fascicle_masks\img_00029.tif",
    cv2.IMREAD_UNCHANGED,
)
apo_image = cv2.imread(
    r"C:\Users\carla\Documents\Master_Thesis\Example_Images\FALLMUD\NeilCronin\aponeurosis_masks\img_00029.jpg",
    cv2.IMREAD_UNCHANGED,
)
original_image = cv2.imread(
    r"C:\Users\carla\Documents\Master_Thesis\Example_Images\FALLMUD\NeilCronin\images\img_00029.tif",
    cv2.IMREAD_UNCHANGED,
)

# original_image, image, apo_image = crop(original_image, image, apo_image)

# get sorted fascicle contours
image_gray, contours_sorted = fascicle_to_contour(image)

# # get extrapolation of aponeuroses
# apo_image_gray, ex_x_LA, ex_y_LA, ex_x_UA, ex_y_UA = apo_to_contour(apo_image)

#### start of independent function ####
start_time = time.time()

# set parameteres
tolerance = 10
tolerance_to_apo = 100
coeff_limit = 0.000583

# # get upper edge of each contour
# contours_sorted_x = []
# contours_sorted_y = []
# for i in range(len(contours_sorted)):
#     contours_sorted[i][0], contours_sorted[i][1] = adapted_contourEdge(
#         "B", contours_sorted[i]
#     )
#     contours_sorted_x.append(contours_sorted[i][0])
#     contours_sorted_y.append(contours_sorted[i][1])

# initialize some important variables
label = {x: False for x in range(len(contours_sorted))}
coefficient_label = []
number_contours = []
all_fascicles_x = []
all_fascicles_y = []
width = original_image.shape[1]
mid = width / 2
LA_curve = list(zip(ex_x_LA, ex_y_LA))
UA_curve = list(zip(ex_x_UA, ex_y_UA))

fascicle_data = pd.DataFrame(
    columns=[
        "number_contours",
        "linear_fit",
        "coordsX",
        "coordsY",
        "coordsX_combined",
        "coordsY_combined",
        "coordsXY",
        "locU",
        "locL",
    ]
)

# # calculate merged fascicle edges
# for i in range(len(contours_sorted)):

#     if label[i] is False and len(contours_sorted_x[i]) > 1:
#         # get upper edge contour of starting fascicle
#         current_fascicle_x = contours_sorted_x[i]
#         current_fascicle_y = contours_sorted_y[i]

#         # set label to true as fascicle is used
#         label[i] = True
#         linear_fit = False
#         inner_number_contours = []
#         inner_number_contours.append(i)

#         # calculate second polynomial coefficients
#         coefficients = np.polyfit(current_fascicle_x, current_fascicle_y, 2)

        # depending on coefficients edge gets extrapolated as first or second order polynomial
        if 0 < coefficients[0] < coeff_limit:
            g = np.poly1d(coefficients)
            ex_current_fascicle_x = np.linspace(
                mid - width, mid + width, 5000
            )  # Extrapolate x,y data using f function
            ex_current_fascicle_y = g(ex_current_fascicle_x)
            linear_fit = False
        else:
            coefficients = np.polyfit(current_fascicle_x, current_fascicle_y, 1)
            g = np.poly1d(coefficients)
            ex_current_fascicle_x = np.linspace(
                mid - width, mid + width, 5000
            )  # Extrapolate x,y data using f function
            ex_current_fascicle_y = g(ex_current_fascicle_x)
            linear_fit = True

#         # compute upper and lower boundary of extrapolation
#         upper_bound = ex_current_fascicle_y - tolerance
#         lower_bound = ex_current_fascicle_y + tolerance

#         # find next fascicle edge within the tolerance, loops as long as a new fascicle edge is found
#         # if no new fascicle is found, found_fascicle is set to -1 within function and loop terminates

#         found_fascicle = 0

#         while found_fascicle >= 0:

#             current_fascicle_x, current_fascicle_y, found_fascicle = find_next_fascicle(
#                 contours_sorted,
#                 contours_sorted_x,
#                 contours_sorted_y,
#                 current_fascicle_x,
#                 current_fascicle_y,
#                 ex_current_fascicle_x,
#                 upper_bound,
#                 lower_bound,
#                 label,
#             )

#             if found_fascicle > 0:
#                 label[found_fascicle] = True
#                 inner_number_contours.append(found_fascicle)
#             else:
#                 break

#             coefficients = np.polyfit(current_fascicle_x, current_fascicle_y, 2)

            if 0 < coefficients[0] < coeff_limit:
                g = np.poly1d(coefficients)
                ex_current_fascicle_x = np.linspace(
                    mid - width, mid + width, 5000
                )  # Extrapolate x,y data using f function
                ex_current_fascicle_y = g(ex_current_fascicle_x)
                linear_fit = False
            else:
                coefficients = np.polyfit(current_fascicle_x, current_fascicle_y, 1)
                g = np.poly1d(coefficients)
                ex_current_fascicle_x = np.linspace(
                    mid - width, mid + width, 5000
                )  # Extrapolate x,y data using f function
                ex_current_fascicle_y = g(ex_current_fascicle_x)
                linear_fit = True

#             upper_bound = ex_current_fascicle_y - tolerance
#             lower_bound = ex_current_fascicle_y + tolerance

#         all_fascicles_x.append(ex_current_fascicle_x)
#         all_fascicles_y.append(ex_current_fascicle_y)
#         coefficient_label.append(linear_fit)
#         number_contours.append(inner_number_contours)

#         fascicle_data_temp = pd.DataFrame(
#             {
#                 "number_contours": [inner_number_contours],
#                 "linear_fit": linear_fit,
#                 "coordsX": None,
#                 "coordsY": None,
#                 "coordsX_combined": None,
#                 "coordsY_combined": None,
#                 "coordsXY": None,
#                 "locU": None,
#                 "locL": None,
#             }
#         )

#         fascicle_data = pd.concat(
#             [fascicle_data, fascicle_data_temp], ignore_index=True
#         )

# number_contours = list(fascicle_data["number_contours"])

# for i in range(len(number_contours)):

#     # calculate linear fit through first contour of fascicle, extrapolate over the complete image and compute intersection point with lower aponeurosis
#     coefficients = np.polyfit(
#         contours_sorted_x[number_contours[i][0]],
#         contours_sorted_y[number_contours[i][0]],
#         1,
#     )
#     g = np.poly1d(coefficients)
#     ex_current_fascicle_x = np.linspace(mid - width, mid + width, 5000)
#     ex_current_fascicle_y = g(ex_current_fascicle_x)

#     fas_LA_curve = list(zip(ex_current_fascicle_x, ex_current_fascicle_y))
#     fas_LA_intersection = do_curves_intersect(LA_curve, fas_LA_curve)

#     # calculate intersection point with lower aponeurosis
#     diffL = ex_current_fascicle_y - ex_y_LA
#     locL = np.where(diffL == min(diffL, key=abs))[0]

#     # find index of first item of first contour
#     first_item = contours_sorted_x[number_contours[i][0]][0]
#     differences = np.abs(ex_current_fascicle_x - first_item)
#     index_first_item = np.argmin(differences)

#     # get extrapolation from the intersection with the lower aponeurosis to the beginning of the first fascicle
#     ex_current_fascicle_x = ex_current_fascicle_x[int(locL) : index_first_item]
#     ex_current_fascicle_y = ex_current_fascicle_y[int(locL) : index_first_item]

#     # convert to list, want list of sections in one list (list in list)
#     coordsX = [list(ex_current_fascicle_x)]
#     coordsY = [list(ex_current_fascicle_y)]

#     # append first contour to list
#     coordsX.append(contours_sorted_x[number_contours[i][0]])
#     coordsY.append(contours_sorted_y[number_contours[i][0]])

#     # append gap between contours and following contours to the list
#     if len(number_contours[i]) > 1:
#         for j in range(len(number_contours[i]) - 1):
#             end_x = contours_sorted_x[number_contours[i][j]][-1]
#             end_y = contours_sorted_y[number_contours[i][j]][-1]
#             start_x = contours_sorted_x[number_contours[i][j + 1]][0]
#             start_y = contours_sorted_y[number_contours[i][j + 1]][0]
#             coordsX.append([end_x, start_x])
#             coordsY.append([end_y, start_y])
#             coordsX.append(contours_sorted_x[number_contours[i][j + 1]])
#             coordsY.append(contours_sorted_y[number_contours[i][j + 1]])

#     # calculate linear fit for last contour, extrapolate over complete image to get intersection point with upper aponeurosis
#     coefficients = np.polyfit(
#         contours_sorted_x[number_contours[i][-1]],
#         contours_sorted_y[number_contours[i][-1]],
#         1,
#     )
#     g = np.poly1d(coefficients)
#     ex_current_fascicle_x_2 = np.linspace(mid - width, mid + width, 5000)
#     ex_current_fascicle_y_2 = g(ex_current_fascicle_x_2)

#     fas_UA_curve = list(zip(ex_current_fascicle_x_2, ex_current_fascicle_y_2))
#     fas_UA_intersection = do_curves_intersect(UA_curve, fas_UA_curve)

#     # calulate intersection point with upper aponeurosis
#     diffU = ex_current_fascicle_y_2 - ex_y_UA
#     locU = np.where(diffU == min(diffU, key=abs))[0]

#     # find index of last item of last contour
#     last_item = contours_sorted_x[number_contours[i][-1]][-1]
#     differences_2 = np.abs(ex_current_fascicle_x_2 - last_item)
#     index_last_item = np.argmin(differences_2)

#     # get extrapolation from the end of the last fascicle to the upper aponeurosis
#     ex_current_fascicle_x_2 = ex_current_fascicle_x_2[index_last_item : int(locU)]
#     ex_current_fascicle_y_2 = ex_current_fascicle_y_2[index_last_item : int(locU)]

#     # append to list
#     coordsX.append(list(ex_current_fascicle_x_2))
#     coordsY.append(list(ex_current_fascicle_y_2))

#     # get new list in which the different lists are not separated
#     coordsX_combined = []
#     coordsX_combined = [item for sublist in coordsX for item in sublist]

#     coordsY_combined = []
#     coordsY_combined = [item for sublist in coordsY for item in sublist]

#     coordsXY = list(zip(coordsX_combined, coordsY_combined))

#     fascicle_data.at[i, "coordsX"] = (
#         coordsX  # x-coordinates of all sections as list in list
#     )
#     fascicle_data.at[i, "coordsY"] = (
#         coordsY  # y-coordinates of all sections as list in list
#     )
#     fascicle_data.at[i, "coordsX_combined"] = (
#         coordsX_combined  # x-coordinates of all sections as one list
#     )
#     fascicle_data.at[i, "coordsY_combined"] = (
#         coordsY_combined  # y-coordinates of all sections as one list
#     )
#     fascicle_data.at[i, "coordsXY"] = (
#         coordsXY  # x- and y-coordinates of all sections as one list
#     )
#     fascicle_data.at[i, "locU"] = locU
#     fascicle_data.at[i, "locL"] = locL
#     fascicle_data.at[i, "intersection_LA"] = fas_LA_intersection
#     fascicle_data.at[i, "intersection_UA"] = fas_UA_intersection

# fascicle_data = fascicle_data[fascicle_data["intersection_LA"]].drop(
#     columns="intersection_LA"
# )  # .reset_index()
# fascicle_data = fascicle_data[fascicle_data["intersection_UA"]].drop(
#     columns="intersection_UA"
# )  # .reset_index()
# fascicle_data = fascicle_data.reset_index(drop=True)

# filter overlapping fascicles
data = adapted_filter_fascicles(fascicle_data, tolerance_to_apo)

# all_coordsX = list(data["coordsX"])
# all_coordsY = list(data["coordsY"])
# all_locU = list(data["locU"])
# all_locL = list(data["locL"])
# data["fascicle_length"] = np.nan
# data["pennation_angle"] = np.nan

# for i in range(len(all_coordsX)):

    # calculate length of fascicle
    curve_length_total = 0

#     for j in range(len(all_coordsX[i])):

#         x = all_coordsX[i][j]
#         y = all_coordsY[i][j]

#         dx = np.diff(x)
#         dy = np.diff(y)

#         segment_lengths = np.sqrt(dx**2 + dy**2)
#         curve_length = np.sum(segment_lengths)
#         curve_length_total += curve_length

    # calculate pennation angle
    apoangle = np.arctan(
        (ex_y_LA[all_locL[i]] - ex_y_LA[all_locL[i] + 50])
        / (ex_x_LA[all_locL[i] + 50] - ex_x_LA[all_locL[i]])
    ) * (180 / np.pi)
    fasangle = np.arctan(
        (all_coordsY[i][0][0] - all_coordsY[i][0][-1])
        / (all_coordsX[i][0][-1] - all_coordsX[i][0][0])
    ) * (180 / np.pi)
    penangle = fasangle - apoangle

#     data.iloc[i, data.columns.get_loc("pennation_angle")] = penangle
#     data.iloc[i, data.columns.get_loc("fascicle_length")] = curve_length_total

end_time = time.time()
total_time = end_time - start_time

# median_length = data["fascicle_length"].median()
# mean_length = data["fascicle_length"].mean()
# median_angle = data["pennation_angle"].median()
# mean_angle = data["pennation_angle"].mean()

print(total_time)
print(data)
print(median_length, mean_length, median_angle, mean_angle)

# colormap = plt.get_cmap("rainbow", len(all_coordsX))

plt.figure(1)
plt.imshow(original_image)
for i in range(len(all_coordsX)):
    color = colormap(i)
    for j in range(len(all_coordsX[i])):
        if j == 0:
            plt.plot(all_coordsX[i][j], all_coordsY[i][j], color=color, alpha=0.4)
        if j % 2 == 1:
            plt.plot(all_coordsX[i][j], all_coordsY[i][j], color="gold", alpha=0.6)
        else:
            plt.plot(all_coordsX[i][j], all_coordsY[i][j], color=color, alpha=0.4)
plt.plot(ex_x_LA, ex_y_LA, color="blue", alpha=0.5)
plt.plot(ex_x_UA, ex_y_UA, color="blue", alpha=0.5)

# plt.show()
