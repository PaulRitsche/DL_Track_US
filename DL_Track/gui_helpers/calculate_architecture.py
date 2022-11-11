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
    a measure of prediction accuracy. This is sometimes also called Jaccard score.
calculateBatch
    Function to calculate muscle architecture in longitudinal ultrasonography images
    of human lower limb muscles. The values computed are fascicle length (FL),
    pennation angle (PA), and muscle thickness (MT).
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import backend as K
from keras.models import load_model
from matplotlib.backends.backend_pdf import PdfPages
from skimage.transform import resize
from tensorflow.keras.utils import img_to_array

from DL_Track.gui_helpers.calibrate import (
    calibrateDistanceManually,
    calibrateDistanceStatic,
)
from DL_Track.gui_helpers.do_calculations import doCalculations
from DL_Track.gui_helpers.manual_tracing import ManualAnalysis

plt.style.use("ggplot")
plt.switch_backend("agg")


def importAndReshapeImage(path_to_image: str, flip: int):
    """
    Function to import and reshape an image. Moreover, based upon
    user specification the image might be flipped.

    Usually flipping is only required when the imported image is of
    a specific structure that is incompatible with the trained models
    provided here

    Parameters
    ----------
    path_to_image : str
         String variable containing the imagepath. This should be an
         absolute path.
    flip : int
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
    img = np.reshape(img, [-1, height, width, 3])
    img = resize(img, (1, 512, 512, 3), mode="constant", preserve_range=True)
    img = img / 255.0

    return img, img_copy, non_flipped_img, height, width, filename


def importImageManual(path_to_image: str, flip: int):
    """
    Function to import an image.

    This function is used when manual analysis of the
    image is selected in the GUI. For manual analysis,
    it is not necessary to resize, reshape and normalize the image.
    The image may be flipped.

    Parameters
    ----------
    path_to_image : str
         String variable containing the imagepath. This should be an
         absolute path.
    flip : int
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
    [[[28 26 25] [28 26 25] [28 26 25] ... [[ 0  0  0] [ 0  0  0] [ 0  0  0]]], Img1.tif
    """
    # Load image
    filename = os.path.splitext(os.path.basename(path_to_image))[0]
    img = cv2.imread(path_to_image, 0)

    # Flip image if required
    if flip == "flip":
        img = cv2.flip(img, 1)

    return img, filename


def getFlipFlagsList(flip_flag_path: str) -> list:
    """
    Function to retrieve flip values from a .txt file.

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
    # open .txt file
    file = open(flip_flag_path, "r")
    for line in file:
        for digit in line:
            if digit.isdigit():
                flip_flags.append(digit)

    return flip_flags


def compileSaveResults(rootpath: str, dataframe: pd.DataFrame) -> None:
    """
    Function to save the analysis results to a .xlsx file.

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
    """
    Function to compute the intersection of union score (IoU),
    a measure of prediction accuracy. This is sometimes also called Jaccard score.

    The IoU can be used as a loss metric during binary segmentation when
    convolutional neural networks are applied. The IoU is calculated for both, the
    training and validation set.

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
    >>> IoU(y_true=Tensor("IteratorGetNext:1", shape=(1, 512, 512, 1), dtype=float32),
             y_pred=Tensor("VGG16_U-Net/conv2d_8/Sigmoid:0", shape=(1, 512, 512, 1), dtype=float32),
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
    apo_treshold: float,
    fasc_threshold: float,
    fasc_cont_thresh: int,
    min_width: int,
    min_pennation: int,
    max_pennation: int,
    gui,
) -> None:
    """
    Function to calculate muscle architecture in longitudinal ultrasonography images
    of human lower limb muscles. The values computed are fascicle length (FL),
    pennation angle (PA), and muscle thickness (MT).

    The scope of this function is limited. Images of the vastus lateralis, tibialis anterior
    soleus and gastrocnemius  muscles can be analyzed. This is due to the limited amount of
    training data for our convolutional neural networks. This functions makes extensive use
    of several other functions and was designed to be executed from a GUI.

    Parameters
    ----------
    rootpath : str
        String variable containing the path to the folder where all images
        to be analyzed are saved.
    apo_modelpath : str
        String variable containing the absolute path to the aponeurosis neural network.
    fasc_modelpath : str
        String variable containing the absolute path to the fascicle neural network.
    flip_flag_path : str
        String variabel containing the absolute path to the flip flag
        .txt file containing the flip flags. Flipping is necessary as the models were trained
        on images of with specific fascicle orientation.
    filetype : str
        String variable containg the respective type of the images.
        This is needed to select only the relevant image files
        in the root directory.
    scaling : str
        String variabel determining the image scaling method.
        There are three types of scaling available:
        - scaling = "manual" (user must scale images manually)
        - sclaing = "bar" (image are scaled automatically. This is done by
          detecting scaling bars on the right side of the image.)
        - scaling = "No scaling" (image is not scaled.)
        Scaling is necessary to compute measurements in centimeter,
        if "no scaling" is chosen, the results are in pixel units.
    spacing : int
        Distance (in milimeter) between two scaling bars in the image.
        This is needed to compute the pixel/cm ratio and therefore report
        the results in centimeter rather than pixel units.
    apo_threshold : float
        Float variable containing the threshold applied to predicted aponeurosis
        pixels by our neural networks. By varying this threshold, different
        structures will be classified as aponeurosis as the threshold for classifying
        a pixel as aponeurosis is changed. Must be non-zero and
        non-negative.
    fasc_threshold : float
        Float variable containing the threshold applied to predicted fascicle
        pixels by our neural networks. By varying this threshold, different
        structures will be classified as fascicle as the threshold for classifying
        a pixel as fascicle is changed.
    fasc_cont_threshold : float
        Float variable containing the threshold applied to predicted fascicle
        segments by our neural networks. By varying this threshold, different
        structures will be classified as fascicle. By increasing, longer fascicle segments
        will be considered, by lowering shorter segments. Must be non-zero and
        non-negative.
    min_width : int
        Integer variable containing the minimal distance between aponeuroses to be
        detected. The aponeuroses must be at least this distance apart to be
        detected. The distance is specified in pixels. Must be non-zero and non-negative.
    min_pennation : int
        Integer variable containing the mininmal (physiological) acceptable pennation
        angle occuring in the analyzed image/muscle. Fascicles with lower pennation
        angles will be excluded. The pennation angle is calculated as the amgle
        of insertion between extrapolated fascicle and detected aponeurosis. Must
        be non-negative.
    min_pennation : int
        Integer variable containing the maximal (physiological) acceptable pennation
        angle occuring in the analyzed image/muscle. Fascicles with higher pennation
        angles will be excluded. The pennation angle is calculated as the amgle
        of insertion between extrapolated fascicle and detected aponeurosis. Must
        be non-negative and larger than min_pennation.
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
    This function is called by the GUI. Note that the functioned was specifically
    designed to be called from the GUI. Thus, tk.messagebox will pop up when errors are
    raised even if the GUI is not started.

    Examples
    --------
    >>> calculateBatch(rootpath="C:/Users/admin/Dokuments/images",
                       apo_modelpath="C:/Users/admin/Dokuments/models/apo_model.h5",
                       fasc_modelpath="C:/Users/admin/Dokuments/models/apo_model.h5",
                       flip_flag_path="C:/Users/admin/Dokuments/flip_flags.txt",
                       filetype="/**/*.tif, scaline="bar", spacing=10, apo_threshold=0.1,
                       fasc_threshold=0.05, fasc_cont_thres=40, curvature=3,
                       min_pennation=10, max_pennation=35,
                       gui=<__main__.DLTrack object at 0x000002BFA7528190>)
    """
    # Get list of files
    list_of_files = glob.glob(rootpath + file_type, recursive=True)

    try:
        # Load models
        model_apo = load_model(apo_modelpath, custom_objects={"IoU": IoU})
        model_fasc = load_model(fasc_modelpath, custom_objects={"IoU": IoU})

    except OSError:
        tk.messagebox.showerror("Information", "Apo/Fasc model path is incorrect.")
        gui.should_stop = False
        gui.is_running = False
        gui.do_break()
        return

    # Check validity of flipflag path and rais execption
    try:
        flip_flags = getFlipFlagsList(flip_file_path)

    except FileNotFoundError:
        tk.messagebox.showerror(
            "Information", "Location of flipflag file is incorrect."
        )
        gui.should_stop = False
        gui.is_running = False
        gui.do_break()
        return

    # Create dictionary with aponeurosis/fascicle analysis values
    dic = {
        "apo_treshold": apo_treshold,
        "fasc_threshold": fasc_threshold,
        "fasc_cont_thresh": fasc_cont_thresh,
        "min_width": min_width,
        "min_pennation": min_pennation,
        "max_pennation": max_pennation,
    }

    # Check if analysis parameters are postive
    for _, value in dic.items():

        if float(value) <= 0:
            tk.messagebox.showerror(
                "Information",
                "Analysis paremters must be non-zero" + " and non-negative",
            )
            gui.should_stop = False
            gui.is_running = False
            gui.do_break()
            return

    # Create Dataframe for result saving
    dataframe = pd.DataFrame(
        columns=["File", "Fasicle Length", "Pennation Angle", "Midthick"]
    )

    # Define list for failed files
    failed_files = []

    # Define count for index in .xlsx file
    count = 0

    # Open PDF for image segmentation saving
    with PdfPages(rootpath + "/ResultImages.pdf") as pdf:

        # Only continue with equal amount of images/flip flags
        if len(list_of_files) == len(flip_flags):

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
                    imported = importAndReshapeImage(imagepath, int(flip))
                    img, img_copy, nonflipped_img, height, width, filename = imported

                    # Determine scaling type and continue analysis
                    if scaling == "Bar":
                        calibrate_fn = calibrateDistanceStatic
                        # Find length of the scaling line
                        calib_dist, scale_statement = calibrate_fn(
                            nonflipped_img, spacing
                        )

                        # Append warning to failed images when no error was found
                        if calib_dist is None:
                            fail = f"Scalingbars not found in {imagepath}"
                            failed_files.append(fail)
                            warnings.warn("Image fails with StaticScalingError")
                            continue

                    # Manual scaling
                    elif scaling == "Manual":
                        calibrate_fn = calibrateDistanceManually
                        calib_dist, scale_statement = calibrate_fn(
                            nonflipped_img, spacing
                        )

                    # No sclaing option
                    else:
                        calib_dist = None
                        scale_statement = ""

                    # Continue with analysis and predict apos and fasicles
                    fasc_l, pennation, _, _, midthick, fig = doCalculations(
                        img,
                        img_copy,
                        height,
                        width,
                        calib_dist,
                        spacing,
                        filename,
                        model_apo,
                        model_fasc,
                        scale_statement,
                        dic,
                    )

                    # Append warning to failes files when no aponeurosis was found and
                    # and continue analysis
                    if fasc_l is None:
                        fail = f"No two aponeuroses found in {imagepath}"
                        failed_files.append(fail)
                        continue

                    # Define output dataframe
                    dataframe2 = pd.DataFrame(
                        {
                            "File": filename,
                            "Fasicle Length": np.median(fasc_l),
                            "Pennation Angle": np.median(pennation),
                            "Midthick": midthick,
                        },
                        index=[count],
                    )

                    # Append results to dataframe
                    dataframe = pd.concat([dataframe, dataframe2], axis=0)

                    # Save figures of fascicles and apos to PDF
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

            except FileNotFoundError:
                tk.messagebox.showerror("Information", "Input directory is incorrect.")
                gui.should_stop = False
                gui.is_running = False
                gui.do_break()
                return

            # Subsequent to analysis of all images, results are saved and the GUI is stopped
            finally:

                # Save predicted area results
                compileSaveResults(rootpath, dataframe)

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
    """
    Function used for manual calculation of fascicle length, muscle thickness
    and pennation angles in longitudinal ultrasonography images of human lower
    limb muscles.

    This function is not restricted to any specific muscles. However, its use is
    restricted to a specific method for assessing muscle thickness fascicle
    length and pennation angles.

    - Muscle thickness:
                       Exactly one segment reaching from the superficial to the
                       deep aponeuroses of the muscle must be drawn. If multiple
                       measurement are drawn, these are averaged. Drawing can
                       be started by clickling the left mouse button and keeping
                       it pressed until it is not further required to draw the line
                       (i.e., the other aponeurosis border is reached). Only the
                       respective y-coordinates of the points where the cursor
                       was clicked and released are considered for calculation of
                       muscle thickness.
    - Fascicle length:
                      Exactly three segments along the fascicleof the muscle must
                      be drawn. If multiple fascicle are drawn, their lengths are
                      averaged. Drawing can be started by clickling the left mouse
                      button and keeping it pressed until one segment is finished
                      (mostly where fascicle curvature occurs the other aponeurosis
                      border is reached). Using the euclidean distance, the total
                      fascicle length is computed as a sum of the segments.
    - Pennation angle:
                      Exactly two segments, one along the fascicle orientation, the
                      other along the aponeurosis orientation must be drawn. The line
                      along the aponeurosis must be started where the line along the
                      fascicle ends. If multiple angle are drawn, they are averaged.
                      Drawing can be started by clickling the left mouse button and keeping
                      it pressed until it is not further required to draw the line
                      (i.e., the aponeurosis border is reached by the fascicle). The
                      angle is calculated using the arc-tan function.
    In order to scale the image, it is required to draw a line of length 10 milimeter
    somewhere in the image. The line can be drawn in the same fashion as for example
    the muscle thickness. Here however, the euclidean distance is used to calculate
    the pixel / centimeter ratio.
    We also provide the functionality to extent the muscle aponeuroses to more easily
    extrapolate fascicles. The lines can be drawn in the same fashion as for example
    the muscle thickness.

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
            "Information", "No image files founds." + "\nEnter correct file type"
        )
        gui.do_break()
        gui.should_stop = False
        gui.is_running = False

    finally:
        # clean up
        gui.should_stop = False
        gui.is_running = False
