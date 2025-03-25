"""
Description
-----------
This module contains functions to automatically or manually
analyse muscle architecture in longitudinal ultrasonography images
of human lower limb muscles.
The scope of the automatic method is limited to the vastus
lateralis, tibialis anterior, gastrocnemius medialis and soleus muscles
due to training data availability.
The scope of the manual method is not limited to specific muscles.
The module was specifically designed to be executed from a GUI.
When used from the GUI, the module saves the analysis results in a .xlsx file
to a given directory. The user needs to provide paths to the image, model,
and flipflag file directories.


Functions scope
---------------
importAndReshapeImage
    Function to import and reshape an image. Moreover, based upon
    user specification the image might be flipped.
importImageManual
     Function to import an image.
importFlipFlagsList
     Function to retrieve flip values from a .txt file.
compileSaveResults
     Function to save the analysis results to a .xlsx file.
IoU
    Function to compute the intersection over union score (IoU),
    a measure of prediction accuracy. This is sometimes also called Jaccard
    score.
calculateBatch
    Function to calculate muscle architecture in longitudinal ultrasonography
    images of human lower limb muscles. The values computed are fascicle
    length (FL), pennation angle (PA), and muscle thickness (MT).
calculateBatchManual
    Function used for manual calculation of fascicle length, muscle thickness
    and pennation angles in longitudinal ultrasonography images of human lower
    limb muscles.

Notes
-----
Additional information and usage exaples can be found at the respective
functions documentations.
"""

import glob
import os
import time
import tkinter as tk
import warnings

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

# Carla imports
from keras import backend as K
from keras.models import load_model
from matplotlib.backends.backend_pdf import PdfPages
from pandas import ExcelWriter
from skimage.transform import resize
from tensorflow.keras.utils import img_to_array


# original imports
from DL_Track_US.gui_helpers.calibrate import (
    calibrateDistanceStatic,
)
from DL_Track_US.gui_helpers.do_calculations import doCalculations
from DL_Track_US.gui_helpers.manual_tracing import ManualAnalysis
from DL_Track_US.gui_helpers.do_calculations_curved import doCalculations_curved
from DL_Track_US.gui_helpers.filter_data import hampelFilterList

# plt.style.use("ggplot")
# plt.switch_backend("Agg")

# Ensure eager execution is enabled
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

# remove interactive plotting
plt.ioff()

# set matplotlib style
# plt.rcParams["figure.figsize"] = [7.00, 3.50]
# plt.rcParams["figure.autolayout"] = True


class ImageProcessor:
    def __init__(
        self, model_apo_path, model_fasc_path, preprocess_function, batch_size=1
    ):
        self.preprocess_function = preprocess_function
        self.batch_size = batch_size
        self.model_apo = tf.keras.models.load_model(
            model_apo_path, custom_objects={"IoU": self.get_iou}
        )
        self.model_fasc = tf.keras.models.load_model(
            model_fasc_path, custom_objects={"IoU": self.get_iou}
        )

    def get_iou(self, y_true, y_pred, smooth: int = 1):
        # Calculate Intersection
        intersect = K.sum(K.abs(y_true * y_pred), axis=-1)
        # Calculate Union
        union = K.sum(y_true, -1) + K.sum(y_pred, -1) - intersect
        # Caclulate iou
        iou = (intersect + smooth) / (union + smooth)

        return iou

    # TODO do all the predictions here and then import the do_calculation functions directly.
    # Maybe even loop trough the images then.
    # So far, this function is unused.
    def process_images(self, image_list):
        preprocessed_images = [self.preprocess_function(image) for image in image_list]
        preprocessed_images = np.stack(preprocessed_images, axis=0)

        results_apo = []
        results_fasc = []
        for i in range(0, len(preprocessed_images), self.batch_size):
            batch = preprocessed_images[i : i + self.batch_size]
            output_apo = self.model_apo.predict(batch)
            output_fasc = self.model_fasc.predict(batch)
            results_apo.extend(output_apo)
            results_fasc.extend(output_fasc)

        return results_apo, results_fasc


# TODO include all image pre-processing in this function
def preprocess_function(image):
    """Function to preprocess an image.

    Parameters
    ----------
    image : np.ndarray
        Image to be preprocessed.

    Returns
    -------
    np.ndarray
        Preprocessed image.
    """
    return image / 255


def importAndReshapeImage(path_to_image: str, flip: int):
    """Function to import and reshape an image. Moreover, based upon
    user specification the image might be flipped.

    Usually flipping is only required when the imported image is of
    a specific structure that is incompatible with the trained models
    provided here

    Parameters
    ----------
    path_to_image : str
         String variable containing the imagepath. This should be an
         absolute path.
    flip : {0, 1}
        Integer value defining wheter an image should be flipped.
        This can be 0 (image is not flipped) or 1 (image is flipped).

    Returns
    -------
    img : np.ndarray
        The loaded images is converted to a np.nadarray. This is done
        using the img_to_array kears functionality. The input image is
        futhter flipped (if selected), resized, respahed and normalized.
    img_copy : np.ndarray
        A copy of the input image.
    non_flipped_img : np.ndarray
        A copy of the input image. This copy is made prior to image
        flipping.
    height : int
        Integer value containing the image height of the input image.
    width : int
        Integer value containing the image width of the input image.
    filename : str
        String value containing the name of the input image, not the
        entire path.

    Examples
    --------
    >>> importAndReshapeImage(path_to_image="C:/Users/Dokuments/images/img1.tif", flip=0)
    [[[[0.10113753 0.09391343 0.09030136] ... [0 0 0]]], [[[28 26 25] ... [ 0  0  0]]], [[[28 26 25] ... [ 0  0  0]]], 512, 512, img1.tif
    """
    # Define the image to analyse here and load image
    filename = os.path.splitext(os.path.basename(path_to_image))[0]
    img = cv2.imread(path_to_image, 1)

    # Save nonflipped img
    non_flipped_img = img

    # Flip if specified
    if flip == 1:
        img = np.fliplr(img)

    # Copying the image here is used for plotting later
    img_copy = img

    # Turn image to array
    img = img_to_array(img)

    # Reshape, resize and normalize image
    height = img.shape[0]
    width = img.shape[1]
    img = np.reshape(img, [-1, img.shape[0], img.shape[1], 3])
    img = resize(img, (1, 512, 512, 3), mode="constant", preserve_range=True)
    img = img / 255.0  # Now used in preprocess image

    return img, img_copy, non_flipped_img, height, width, filename


def importImageManual(path_to_image: str, flip: int):
    """Function to import an image.

    This function is used when manual analysis of the
    image is selected in the GUI. For manual analysis,
    it is not necessary to resize, reshape and normalize the image.
    The image may be flipped.

    Parameters
    ----------
    path_to_image : str
         String variable containing the imagepath. This should be an
         absolute path.
    flip : {0, 1}
        Integer value defining wheter an image should be flipped.
        This can be 0 (image is not flipped) or 1 (image is flipped).

    Returns
    -------
    img : np.ndarray
        The loaded images as a np.nadarray in grayscale.
    filename : str
        String value containing the name of the input image, not the
        entire path.

    Examples
    --------
    >>> importImageManual(path_to_image="C:/Desktop/Test/Img1.tif", flip=0)
    [[[28 26 25] [28 26 25] [28 26 25] ... [[ 0  0  0] [ 0  0  0] [ 0  0  0]]],
    Img1.tif
    """
    # Load image
    filename = os.path.splitext(os.path.basename(path_to_image))[0]
    img = cv2.imread(path_to_image, 0)

    # Flip image if required
    if flip == "flip":
        img = cv2.flip(img, 1)

    return img, filename


def getFlipFlagsList(flip_flag_path: str, gui) -> list:
    """Function to retrieve flip values from a .txt file.

    The flip flags decide wether an image should be flipped or not.
    The flags can be 0 (image not flipped) or 1 (image is flipped).
    The flags must be specified in the .txt file and can either be
    on a seperate line for each image or on a seperate line for each folder.
    The amount of flip flags must equal the amount of images analyzed.

    Parameters
    ----------
    flip_flag_path : str
        String variabel containing the absolute path to the flip flag
        .txt file containing the flip flags.
    gui : tk.Tk
        The GUI object.

    Returns
    -------
    flip_flags : list
        A list variable containing all flip flags included in the
        specified .txt file

    Example
    -------
    >>> getFlipFlagsList(flip_flag_path="C:/Desktop/Test/FlipFlags/flags.txt")
    [1, 0, 1, 0, 1, 1, 1]
    """
    # Specify empty list
    flip_flags = []

    try:
        # open .txt file
        file = open(flip_flag_path, "r")
        for line in file:
            for digit in line:
                if digit.isdigit():
                    flip_flags.append(digit)

        return flip_flags

    except UnicodeDecodeError:
        tk.messagebox.showerror(
            "Information", "Location of flipflag file is incorrect."
        )
        gui.should_stop = False
        gui.is_running = False
        gui.do_break()
        return

    except FileNotFoundError:
        tk.messagebox.showerror(
            "Information", "Location of flipflag file is incorrect."
        )
        gui.should_stop = False
        gui.is_running = False
        gui.do_break()
        return


def exportToExcel(
    path: str,
    fasc_l_all: list,
    pennation_all: list,
    x_lows_all: list,
    x_highs_all: list,
    thickness_all: list,
    filename: str = "Results",
    filtered_fasc=None,
    filtered_pennation=None,
):
    """
    Saves raw and optionally filtered analysis results to an Excel file, including an index column for each frame.

    Parameters
    ----------
    path : str
        Path where the Excel file will be saved.
    filename : str
        Name of the Excel file.
    fasc_l_all : list of lists
        Raw fascicle length data.
    pennation_all : list of lists
        Raw pennation angle data.
    x_lows_all : list of lists
        X-coordinates of lower aponeurosis intersections.
    x_highs_all : list of lists
        X-coordinates of upper aponeurosis intersections.
    thickness_all : list of lists
        Muscle thickness measurements.
    filtered_fasc : list of lists, optional
        Filtered fascicle length data.
    filtered_pennation : list of lists, optional
        Filtered pennation angle data.
    """

    def create_array(data):
        """Creates a NumPy array with NaNs for padding."""
        return np.full((len(data), len(max(data, key=len, default=[]))), np.nan)

    # Convert lists to NumPy arrays
    fl_raw = create_array(fasc_l_all)
    pe_raw = create_array(pennation_all)
    xl = create_array(x_lows_all)
    xh = create_array(x_highs_all)

    # Fill arrays with data
    for i, j in enumerate(fasc_l_all):
        fl_raw[i, : len(j)] = j
    for i, j in enumerate(pennation_all):
        pe_raw[i, : len(j)] = j
    for i, j in enumerate(x_lows_all):
        xl[i, : len(j)] = j
    for i, j in enumerate(x_highs_all):
        xh[i, : len(j)] = j

    # Create DataFrames with an index column
    df1 = pd.DataFrame(data=fl_raw, dtype=float)
    df2 = pd.DataFrame(data=pe_raw, dtype=float)
    df3 = pd.DataFrame(data=xl, dtype=float)
    df4 = pd.DataFrame(data=xh, dtype=float)
    df5 = pd.DataFrame(data=thickness_all, dtype=float)

    # Write to Excel
    writer = ExcelWriter(f"{path}/{filename}.xlsx")

    df1.to_excel(writer, sheet_name="Fasc_length_raw", index=True)
    df2.to_excel(writer, sheet_name="Pennation_raw", index=True)
    df3.to_excel(writer, sheet_name="X_low", index=True)
    df4.to_excel(writer, sheet_name="X_high", index=True)
    df5.to_excel(writer, sheet_name="Thickness", index=True)

    # Add filtered fascicle length (if provided)
    if filtered_fasc is not None:
        fl_filtered = create_array(filtered_fasc)
        for i, j in enumerate(filtered_fasc):
            fl_filtered[i, : len(j)] = j
        df6 = pd.DataFrame(data=fl_filtered, dtype=float)
        df6.to_excel(writer, sheet_name="Fasc_length_filtered", index=True)

        # add overall filtereed median
        df61 = pd.DataFrame(
            hampelFilterList(df6.median(axis=1, skipna=True))["filtered"]
        )
        df61.to_excel(writer, sheet_name="Fasc_length_filtered_median", index=True)

    # Add filtered pennation angle (if provided)
    if filtered_pennation is not None:
        pe_filtered = create_array(filtered_pennation)
        for i, j in enumerate(filtered_pennation):
            pe_filtered[i, : len(j)] = j
        df7 = pd.DataFrame(data=pe_filtered, dtype=float)
        df7.to_excel(writer, sheet_name="Pennation_filtered", index=True)

        df71 = pd.DataFrame(
            hampelFilterList(df7.median(axis=1, skipna=True))["filtered"]
        )
        df71.to_excel(writer, sheet_name="pennation_filtered_median", index=True)

    writer.close()


def compileSaveResults(rootpath: str, dataframe: pd.DataFrame) -> None:
    """Function to save the analysis results to a .xlsx file.

    A pd.DataFrame object must be inputted. The results
    inculded in the dataframe are saved to an .xlsx file.
    The .xlsx file is saved to the specified rootpath.

    Parameters
    ----------
    rootpath : str
        String variable containing the path to where the .xlsx file
        should be saved.
    dataframe : pd.DataFrame
        Pandas dataframe variable containing the image analysis results
        for every image anlyzed.

    Examples
    --------
    >>> saveResults(img_path = "C:/Users/admin/Dokuments/images",
                    dataframe = [['File',"image1"],['FL', 12],
                                 ['PA', 17], ...])
    """
    # Make filepath
    excelpath = rootpath + "/Results.xlsx"

    # Check if file already existing and write .xlsx
    if os.path.exists(excelpath):
        with pd.ExcelWriter(excelpath, mode="a", if_sheet_exists="replace") as writer:
            data = dataframe
            data.to_excel(writer, sheet_name="Results")
    else:
        with pd.ExcelWriter(excelpath, mode="w") as writer:
            data = dataframe
            data.to_excel(writer, sheet_name="Results")


def IoU(y_true, y_pred, smooth: int = 1) -> float:
    """Function to compute the intersection of union score (IoU),
    a measure of prediction accuracy. This is sometimes also called Jaccard
    score.

    The IoU can be used as a loss metric during binary segmentation when
    convolutional neural networks are applied. The IoU is calculated for both,
    the training and validation set.

    Parameters
    ----------
    y_true : tf.Tensor
        True positive image segmentation label predefined by the user.
        This is the mask that is provided prior to model training.
    y_pred : tf.Tensor
        Predicted image segmentation by the network.
    smooth : int, default = 1
        Smoothing operator applied during final calculation of
        IoU. Must be non-negative and non-zero.

    Returns
    -------
    iou : tf.Tensor
        IoU representation in the same shape as y_true, y_pred.

    Notes
    -----
    The IoU is usually calculated as IoU = intersection / union.
    The intersection is calculated as the overlap of y_true and
    y_pred, whereas the union is the sum of y_true and y_pred.

    Examples
    --------
    >>> IoU(y_true=Tensor("IteratorGetNext:1", shape=(1, 512, 512, 1),
             dtype=float32),
             y_pred=Tensor("VGG16_U-Net/conv2d_8/Sigmoid:0",
             shape=(1, 512, 512, 1), dtype=float32),
             smooth=1)
    Tensor("truediv:0", shape=(1, 512, 512), dtype=float32)
    """
    # Calculate Intersection
    intersect = K.sum(K.abs(y_true * y_pred), axis=-1)
    # Calculate Union
    union = K.sum(y_true, -1) + K.sum(y_pred, -1) - intersect
    # Caclulate iou
    iou = (intersect + smooth) / (union + smooth)

    return iou


def calculateBatch(
    rootpath: str,
    apo_modelpath: str,
    fasc_modelpath: str,
    flip_file_path: str,
    file_type: str,
    scaling: str,
    spacing: int,
    filter_fasc: bool,
    settings: dict,
    gui,
    image_frame=None,
) -> None:
    """Function to calculate muscle architecture in longitudinal
    ultrasonography images of human lower limb muscles. The values
    computed are fascicle length (FL), pennation angle (PA),
    and muscle thickness (MT).

    The scope of this function is limited. Images of the vastus lateralis,
    tibialis anterior soleus and gastrocnemius  muscles can be analyzed.
    This is due to the limited amount of training data for our convolutional
    neural networks. This functions makes extensive use
    of several other functions and was designed to be executed from a GUI.

    Parameters
    ----------
    rootpath : str
        String variable containing the path to the folder where all images
        to be analyzed are saved.
    apo_modelpath : str
        String variable containing the absolute path to the aponeurosis
        neural network.
    fasc_modelpath : str
        String variable containing the absolute path to the fascicle
        neural network.
    flip_flag_path : str
        String variabel containing the absolute path to the flip flag
        .txt file containing the flip flags. Flipping is necessary as the
        models were trained on images of with specific fascicle orientation.
    filetype : str
        String variable containg the respective type of the images.
        This is needed to select only the relevant image files
        in the root directory.
    scaling : {"bar", "manual", "No scaling"}
        String variabel determining the image scaling method.
        There are three types of scaling available:
        - scaling = "manual" (user must scale images manually)
        - sclaing = "bar" (image are scaled automatically. This is done by
          detecting scaling bars on the right side of the image.)
        - scaling = "No scaling" (image is not scaled.)
        Scaling is necessary to compute measurements in centimeter,
        if "no scaling" is chosen, the results are in pixel units.
    spacing : {10, 5, 15, 20}
        Distance (in milimeter) between two scaling bars in the image.
        This is needed to compute the pixel/cm ratio and therefore report
        the results in centimeter rather than pixel units.
    filter_fasc : bool
        If True, fascicles will be filtered so that no crossings are included.
        This may reduce number of totally detected fascicles.
    settings : dict
        Dictionary containing the analysis settings of the GUI.
    gui : tk.TK
        A tkinter.TK class instance that represents a GUI. By passing this
        argument, interaction with the GUI is possible i.e., stopping
        the calculation process after each image.

    See Also
    --------
    do_calculations.py for exact description of muscle architecture parameter
    calculation.

    Notes
    -----
    For specific explanations of the included functions see the respective
    function docstrings in this module. To see an examplenary PDF output
    and .xlsx file take at look at the examples provided in the "examples"
    directory.
    This function is called by the GUI. Note that the functioned was
    specifically designed to be called from the GUI. Thus, tk.messagebox
    will pop up when errors are raised even if the GUI is not started.

    Examples
    --------
    >>> calculateBatch(rootpath="C:/Users/admin/Dokuments/images",
                       apo_modelpath="C:/Users/admin/Dokuments/models/apo_model.h5",
                       fasc_modelpath="C:/Users/admin/Dokuments/models/apo_model.h5",
                       flip_flag_path="C:/Users/admin/Dokuments/flip_flags.txt",
                       filetype="/**/*.tif, scaline="bar", spacing=10, filter_fasc=False,
                       settings=settings,
                       gui=<__main__.DL_Track_US object at 0x000002BFA7528190>)
    """
    # Get list of files
    list_of_files = glob.glob(rootpath + file_type, recursive=True)

    # Check validity of flipflag path and rais execption
    flip_flags = getFlipFlagsList(flip_file_path, gui)

    # Check analysis parameters for positive values
    for _, value in settings.items():
        # Ensure value is not empty or zero
        if isinstance(value, str):
            if not value.strip():  # Check if string is empty or whitespace
                tk.messagebox.showerror(
                    "Information",
                    "Analysis parameters must be non-zero and non-negative",
                )
                gui.should_stop = False
                gui.is_running = False
                gui.do_break()
                return
        else:
            if float(value) <= 0:  # Check if numeric value is <= 0
                tk.messagebox.showerror(
                    "Information",
                    "Analysis parameters must be non-zero and non-negative",
                )
                gui.should_stop = False
                gui.is_running = False
                gui.do_break()
                return

    # Get fascicle calcilation approach
    fasc_calculation_approach = settings["fascicle_calculation_method"]

    # Define list for failed files
    failed_files = []

    # Define lists for paramters of all images
    fascicles_all = []
    pennation_all = []
    x_low_all = []
    x_high_all = []
    thickness_all = []

    # Define count for index in .xlsx file
    count = 0

    # Open PDF for image segmentation saving
    with PdfPages(rootpath + "/ResultImages.pdf") as pdf:

        # Only continue with equal amount of images/flip flags
        if len(list_of_files) == len(flip_flags):

            image_processor = ImageProcessor(
                apo_modelpath, fasc_modelpath, preprocess_function
            )
            # Exceptions raised during the analysis process.
            try:
                start_time = time.time()

                # Loop through files
                for imagepath in list_of_files:

                    # Get start time
                    start_time_img = time.time()

                    # Check if gui signed to stop
                    if gui.should_stop:
                        # there was an input to stop the calculations
                        break

                    # Get individual flipflag
                    flip = flip_flags.pop(0)

                    # Load image
                    img, img_copy, nonflipped_img, height, width, filename = (
                        importAndReshapeImage(imagepath, int(flip))
                    )

                    # Determine scaling type and continue analysis
                    if scaling == "Bar":
                        calibrate_fn = calibrateDistanceStatic
                        # Find length of the scaling line
                        calib_dist, scale_statement = calibrate_fn(
                            nonflipped_img, spacing
                        )

                        # Append warning to failed images when no error was
                        # found
                        if calib_dist is None:
                            fail = f"Scalingbars not found in {imagepath}"
                            failed_files.append(fail)
                            warnings.warn("Image fails with StaticScalingError")
                            continue

                    # Manual scaling here because only applied once on first image
                    if scaling == "Manual":
                        calib_dist = gui.calib_dist
                        scale_statement = f"10 mm corresponds to {calib_dist} pixels"

                    # No sclaing option
                    else:
                        calib_dist = None
                        scale_statement = ""

                    if fasc_calculation_approach == "linear_extrapolation":

                        # Continue with analysis and predict apos and fasicles
                        fasc_l, pennation, x_low, x_high, midthick, fig = (
                            doCalculations(
                                original_image=img,
                                img_copy=img_copy,
                                h=height,
                                w=width,
                                calib_dist=calib_dist,
                                spacing=spacing,
                                model_apo=image_processor.model_apo,
                                model_fasc=image_processor.model_fasc,
                                dictionary=settings,
                                filter_fasc=filter_fasc,
                                image_callback=image_frame,
                            )
                        )

                    elif fasc_calculation_approach == "curve_polyfitting":
                        fasc_l, pennation, midthick, x_low, x_high, fig = (
                            doCalculations_curved(
                                original_image=img,
                                img_copy=img_copy,
                                h=height,
                                w=width,
                                model_apo=image_processor.model_apo,
                                model_fasc=image_processor.model_fasc,
                                dic=settings,
                                filter_fasc=filter_fasc,
                                calib_dist=calib_dist,
                                spacing=spacing,
                                approach="curve_polyfitting",
                                image_callback=image_frame,
                            )
                        )

                    elif fasc_calculation_approach == "curve_connect_linear":
                        fasc_l, pennation, midthick, x_low, x_high, fig = (
                            doCalculations_curved(
                                original_image=img,
                                img_copy=img_copy,
                                h=height,
                                w=width,
                                model_apo=image_processor.model_apo,
                                model_fasc=image_processor.model_fasc,
                                dic=settings,
                                filter_fasc=filter_fasc,
                                calib_dist=calib_dist,
                                spacing=spacing,
                                approach="curve_connect_linear",
                                image_callback=image_frame,
                            )
                        )

                    elif fasc_calculation_approach == "curve_connect_poly":
                        fasc_l, pennation, midthick, x_low, x_high, fig = (
                            doCalculations_curved(
                                original_image=img,
                                img_copy=img_copy,
                                h=height,
                                w=width,
                                model_apo=image_processor.model_apo,
                                model_fasc=image_processor.model_fasc,
                                dic=settings,
                                filter_fasc=filter_fasc,
                                calib_dist=calib_dist,
                                spacing=spacing,
                                approach="curve_connect_poly",
                                image_callback=image_frame,
                            )
                        )

                    elif fasc_calculation_approach == "orientation_map":
                        fasc_l, pennation, midthick, x_low, x_high, fig = (
                            doCalculations_curved(
                                original_image=img,
                                img_copy=img_copy,
                                h=height,
                                w=width,
                                model_apo=image_processor.model_apo,
                                model_fasc=image_processor.model_fasc,
                                dic=settings,
                                filter_fasc=filter_fasc,
                                calib_dist=calib_dist,
                                spacing=spacing,
                                approach="orientation_map",
                                image_callback=image_frame,
                            )
                        )
                        x_low = None
                        x_high = None

                    # Append warning to failes files when no aponeurosis was
                    # found and continue analysis
                    if fasc_l is None:
                        fail = f"No two aponeuroses found in {imagepath}"
                        failed_files.append(fail)
                        continue

                    # Sort parameters
                    # Creating a DataFrame from the lists
                    df = pd.DataFrame(
                        {
                            "Fascicles": fasc_l,
                            "Pennation": pennation,
                            "X_low": x_low,
                            "X_high": x_high,
                        }
                    )

                    # Sorting the DataFrame according to X_low
                    df_sorted = df.sort_values(by="X_low")

                    # Append parameters to overall list
                    fascicles_all.append(df_sorted["Fascicles"].tolist())
                    pennation_all.append(df_sorted["Pennation"].tolist())
                    x_low_all.append(df_sorted["X_low"].tolist())
                    x_high_all.append(df_sorted["X_high"].tolist())
                    thickness_all.append(midthick)

                    # Save figures of fascicles and apos to PDF
                    fig.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

                    # Get time duration of analysis of single image
                    duration_img = time.time() - start_time_img
                    print(f"duration single image: {duration_img}")

                    # Increase index count
                    count += 1

                # Get time of total analysis
                duration = time.time() - start_time
                print(f"duration total analysis: {duration}")

                # Apply Hampel Filter to the results to avoid outliers
                # This is done here to loop over the final dataframe
                fasc_l_all_filtered = [
                    hampelFilterList(fasc_l, win_size=4, num_dev=1)["filtered"]
                    for fasc_l in fascicles_all
                ]
                pennation_all_filtered = [
                    hampelFilterList(pennation, win_size=4, num_dev=1)["filtered"]
                    for pennation in pennation_all
                ]

                # Save results as.xlsx file
                exportToExcel(
                    rootpath,
                    fascicles_all,
                    pennation_all,
                    x_low_all,
                    x_high_all,
                    thickness_all,
                    filtered_fasc=fasc_l_all_filtered,
                    filtered_pennation=pennation_all_filtered,
                )

            except FileNotFoundError:
                tk.messagebox.showerror("Information", "Input directory is incorrect.")
                gui.should_stop = False
                gui.is_running = False
                gui.do_break()
                return

            except ValueError:
                failed_files.append(fail)
                warnings.warn("No aponeuroses found in image.")

            except PermissionError:
                tk.messagebox.showerror(
                    "Information", "Close results file berfore ongoing analysis."
                )
                gui.should_stop = False
                gui.is_running = False
                gui.do_break()
                return

            # Subsequent to analysis of all images, results are saved and
            # the GUI is stopped
            finally:

                # Write failed images in file
                if len(failed_files) >= 1:
                    file = open(rootpath + "/failed_images.txt", "w")
                    for fail in failed_files:
                        file.write(fail + "\n")
                    file.close()

                # Clean up
                gui.should_stop = False
                gui.is_running = False

        # Check if images are contained in specified directory
        elif len(list_of_files) == 0:
            tk.messagebox.showerror(
                "Information",
                "No images files found."
                + "\nCheck specified image type or input directory",
            )
            gui.do_break()
            gui.should_stop = False
            gui.is_running = False

        # Filpflage != number of images
        else:
            tk.messagebox.showerror(
                "Information", "Number of flipflags must match number of images."
            )
            gui.should_stop = False
            gui.is_running = False


def calculateBatchManual(rootpath: str, filetype: str, gui):
    """Function used for manual calculation of fascicle length,
    muscle thickness and pennation angles in longitudinal
    ultrasonography images of human lower limb muscles.

    This function is not restricted to any specific muscles. However,
    its use is restricted to a specific method for assessing muscle
    thickness fascicle length and pennation angles.

    - Muscle thickness:
                       Exactly one segment reaching from the superficial to the
                       deep aponeuroses of the muscle must be drawn.
                       If multiple measurement are drawn, these are averaged.
                       Drawing can be started by clickling the left mouse
                       button and keeping it pressed until it is not further
                       required to draw the line (i.e., the other aponeurosis
                       border is reached). Only the respective y-coordinates
                       of the points where the cursor was clicked and released
                       are considered for calculation of muscle thickness.
    - Fascicle length:
                      Exactly three segments along the fascicleof the muscle
                      must be drawn. If multiple fascicle are drawn, their
                      lengths are averaged. Drawing can be started by clickling
                      the left mouse button and keeping it pressed until one
                      segment is finished (mostly where fascicle curvature
                      occurs the other aponeurosis border is reached). Using
                      the euclidean distance, the total fascicle length is
                      computed as a sum of the segments.
    - Pennation angle:
                      Exactly two segments, one along the fascicle
                      orientation, the other along the aponeurosis orientation
                      must be drawn. The line along the aponeurosis must be
                      started where the line along the fascicle ends. If
                      multiple angle are drawn, they are averaged. Drawing can
                      be started by clickling the left mouse button and keeping
                      it pressed until it is not further required to draw the
                      line (i.e., the aponeurosis border is reached by the
                      fascicle). The angle is calculated using the arc-tan
                      function.
    In order to scale the frame, it is required to draw a line of length 10
    milimeter somewhere in the image. The line can be drawn in the same
    fashion as for example the muscle thickness. Here however, the euclidean
    distance is used to calculate the pixel / centimeter ratio. This has to
    be done for every frame.

    We also provide the functionality to extent the muscle aponeuroses to more
    easily extrapolate fascicles. The lines can be drawn in the same fashion as
    for example the muscle thickness.

    Parameters
    ----------
    rootpath : str
        String variable containing the path to the folder where all images
        to be analyzed are saved.
    gui : tk.TK
        A tkinter.TK class instance that represents a GUI. By passing this
        argument, interaction with the GUI is possible i.e., stopping
        the calculation process after each image.

    Notes
    -----
    For specific explanations of the included functions see the respective
    function docstrings in this module. The function outputs an .xlsx
    file in rootpath containing the (averaged) analysis results for each
    image.

    Examples
    --------
    >>> calculateBatchManual(rootpath="C:/Users/admin/Dokuments/images",
                             filetype="/**/*.tif,
                             gui")
    """
    list_of_files = glob.glob(rootpath + filetype, recursive=True)

    try:
        analysis = ManualAnalysis(list_of_files, rootpath)
        analysis.calculateBatchManual()

    except IndexError:
        tk.messagebox.showerror(
            "Information", "No image files founds" + "\nEnter correct file type"
        )
        gui.do_break()
        gui.should_stop = False
        gui.is_running = False

    finally:
        # clean up
        gui.should_stop = False
        gui.is_running = False
