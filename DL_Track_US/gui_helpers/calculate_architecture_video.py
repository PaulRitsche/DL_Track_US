"""
Description
-----------
This module contains functions to automatically or manually
analyse muscle architecture in longitudinal ultrasonography videos
of human lower limb muscles.
The scope of the automatic method is limited to the vastus
lateralis, tibialis anterior, gastrocnemius medialis and soleus muscles
due to training data availability.
The scope of the manual method is not limited to specific muscles.
The module was specifically designed to be executed from a GUI.
When used from the GUI, the module saves the analysis results in a .xlsx file
to a given directory. The user needs to provide paths to the video, model,
and flipflag file directories.
With both methods, every frame is analyzed seperately and the results for each
frame are saved.

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
calculateBatch
    Function to calculate muscle architecture in longitudinal ultrasonography
    images of human lower limb muscles. The values computed are fascicle length
    (FL), pennation angle (PA), and muscle thickness (MT).
calculateBatchManual
    Function used for manual calculation of fascicle length, muscle thickness
    and pennation angles in longitudinal ultrasonography images of human lower
    limb muscles.

Notes
-----
Additional information and usage examples can be found at the respective
functions documentations.

See Also
--------
calculate_architecture.py
"""

from __future__ import division

import glob
import os
import time
import tkinter as tk

import cv2
import matplotlib.pyplot as plt
from keras.models import load_model


from DL_Track_US.gui_helpers.do_calculations_video import doCalculationsVideo
from DL_Track_US.gui_helpers.calculate_architecture import exportToExcel
from DL_Track_US.gui_helpers.model_training import (
    IoU,
    dice_bce_loss,
    dice_score,
    focal_tversky,
    hybrid_loss,
)
from DL_Track_US.gui_helpers.manual_tracing import ManualAnalysis
from DL_Track_US.gui_helpers.filter_data import applyFilters, hampelFilterList

plt.style.use("ggplot")
plt.switch_backend("agg")


def importVideo(vpath: str):
    """Function to import a video. Video file types should be common
    ones like .avi or .mp4.

    Parameters
    ----------
    vpath : str
        String variable containing the video. This should be an
        absolute path.

    Returns
    -------
    cap : cv2.VideoCapture
        Object that contains the video in a np.ndarrray format.
        In this way, seperate frames can be accessed.
    vid_len : int
        Integer variable containing the number of frames present
        in cap.
    width : int
        Integer variable containing the image width of the input image.
    filename : str
        String variable containing the name of the input video, not the
        entire path.
    vid_out : cv2.VideoWriter
        Object that is stored in the vpath folder.
        Contains the analyzed video frames and is titled "..._proc.avi"
        The name can be changed but must be different than the input
        video.

    Examples
    --------
    >>> importVideo(vpath="C:/Users/Dokuments/videos/video1.avi")
    """
    # Video properties (do not edit)
    cap = cv2.VideoCapture(vpath)
    vid_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_fps = int(cap.get(cv2.CAP_PROP_FPS))
    vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    filename = os.path.splitext(os.path.basename(vpath))[0]
    outpath = str(vpath[0:-4] + "_proc" + ".avi")
    vid_out = cv2.VideoWriter(
        outpath, cv2.VideoWriter_fourcc(*"MPEG"), vid_fps, (vid_width, vid_height)
    )

    return cap, vid_len, filename, vid_out


def importVideoManual(vpath: str):
    """Function to import a video. Video file types should be common
    ones like .avi or .mp4. This function is used for manual
    analysis of videos.

    Here, no processed video is saved subsequent to analysis.

    Parameters
    ----------
    vpath : str
        String variable containing the video. This should be an
        absolute path.

    Returns
    -------
     cap : cv2.VideoCapture
        Object that contains the video in a np.ndarrray format.
        In this way, seperate frames can be accessed.
    vid_len : np.ndarray
        A copy of the input image.
    vid_width : int
        Integer value containing the image width of the input video.
    vid_height : int
        Integer value containing the image height of the input video.
    width : int
        Integer value containing the image width of the input image.
    filename : str
        String value containing the name of the input video, not the
        entire path.

    Examples
    --------
    >>> importVideo(vpath="C:/Users/Dokuments/videos/video1.avi")
    """

    # Video properties (do not edit)
    cap = cv2.VideoCapture(vpath)
    vid_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    filename = os.path.splitext(os.path.basename(vpath))[0]

    return cap, vid_len, filename


def calculateArchitectureVideo(
    rootpath: str,
    apo_modelpath: str,
    fasc_modelpath: str,
    filetype: str,
    scaling: str,
    flip: str,
    step: int,
    filter_fasc: bool,
    settings: dict,
    gui,
    display_frame,
):
    """Function to calculate muscle architecture in longitudinal
    ultrasonography videos of human lower limb muscles. The values
    computed are fascicle length (FL), pennation angle (PA),
    and muscle thickness (MT).

    The scope of this function is limited. videos of the vastus lateralis,
    tibialis anterior soleus and gastrocnemius  muscles can be analyzed.
    This is due to the limited amount of training data for our convolutional
    neural networks. This functions makes extensive use
    of several other functions and was designed to be executed from a GUI.

    Parameters
    ----------
    rootpath : str
        String variable containing the path to the folder where all videos
        to be analyzed are saved.
    apo_modelpath : str
        String variable containing the absolute path to the aponeurosis
        neural network.
    fasc_modelpath : str
        String variable containing the absolute path to the fascicle
        neural network.
    flip : str
        String variable determining wheter all frames of a video are
        flipped vetically.
        Flipping is necessary as the models were trained in images of
        with specific fascicle orientation.
    filetype : str
        String variable containg the respective type of the videos.
        This is needed to select only the relevant video files
        in the root directory.
    scaling : str
        String variable determining the image scaling method.
        There are three types of scaling available:
        - scaling = "manual" (user must scale the video manually.
          This only needs to be done in the first frame.)
          detecting scaling bars on the right side of the image.)
        - scaling = "No scaling" (video frames are not scaled.)
        Scaling is necessary to compute measurements in centimeter,
        if "no scaling" is chosen, the results are in pixel units.
    step : int
        Integer variable containing the step for the range of video frames.
        If step != 1, frames are skipped according to the size of step.
        This might decrease processing time but also accuracy.
    filter_fasc : bool
        If True, fascicles will be filtered so that no crossings are included.
        This may reduce number of totally detected fascicles.
    settings : dict
        Dictionary containing the settings of the GUI. These specify the
        prediction parameters for the neural networks.
    gui : tk.TK
        A tkinter.TK class instance that represents a GUI. By passing this
        argument, interaction with the GUI is possible i.e., stopping
        the calculation process after each image.
    display_frame : bool
        Boolean variable determining whether the current frame is displayed in main
        UI.

    See Also
    --------
    do_calculations_video.py for exact description of muscle architecture
    parameter calculation.

    Notes
    -----
    For specific explanations of the included functions see the respective
    function docstrings in this module. To see an examplenary video output
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
                       flip="Flip", filetype="/**/*.avi, scaline="manual",
                       filter_fasc=False
                       settings=settings,
                       gui=<__main__.DLTrack object at 0x000002BFA7528190>,
                       display_frame=True)
    """
    list_of_files = glob.glob(rootpath + filetype, recursive=True)

    if len(list_of_files) == 0:
        tk.messagebox.showerror(
            "Information",
            "No video files found." + "\nCheck specified video type or input directory",
        )
        gui.do_break()
        gui.should_stop = False
        gui.is_running = False

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

    try:

        start_time = time.time()

        # Load models outside the loop
        model_apo = load_model(apo_modelpath, custom_objects={"IoU": IoU})

        if settings["segmentation_mode"] == "stacked":
            model_fasc = load_model(
                fasc_modelpath,
                custom_objects={
                    "IoU": IoU,
                    "focal_tversky": focal_tversky,
                    "dice_score": dice_score,
                    "hybrid_loss": hybrid_loss,
                },
            )

        else:
            model_fasc = load_model(
                fasc_modelpath,
                custom_objects={
                    "IoU": IoU,
                    "dice_bce_loss": dice_bce_loss,
                    "dice_score": dice_score,
                },
            )

        for video in list_of_files:
            if gui.should_stop:
                # there was an input to stop the calculations
                break

            # Check if result already exists to skip reprocessing
            filename = os.path.splitext(os.path.basename(video))[0]
            output_excel = os.path.join(rootpath, f"{filename}_results.xlsx")
            if os.path.exists(output_excel):
                print(f"Skipping {filename} — already processed.")
                continue

            # load video
            imported = importVideo(video)
            cap, vid_len, gui.filename, vid_out = imported

            # find length of the scaling line
            if scaling == "Manual":
                calib_dist = gui.calib_dist
            else:
                calib_dist = None

            # predict apos and fasicles
            calculate = doCalculationsVideo
            (
                fasc_l_all,
                pennation_all,
                x_lows_all,
                x_highs_all,
                thickness_all,
            ) = calculate(
                vid_len=vid_len,
                cap=cap,
                vid_out=vid_out,
                flip=flip,
                apo_model=model_apo,
                fasc_model=model_fasc,
                calib_dist=calib_dist,
                dic=settings,
                step=step,
                filter_fasc=filter_fasc,
                gui=gui,
                segmentation_mode=settings["segmentation_mode"],
                frame_callback=display_frame,
            )

            duration = time.time() - start_time
            print(f"Video duration: {duration}")

            # Apply Hampel Filter to the results to avoid outliers
            # This is done here to loop over the final dataframe
            if settings["selected_filter"] == "hampel":
                fasc_l_all_filtered = [
                    hampelFilterList(
                        fasc_l,
                        win_size=settings["hampel_window_size"],
                        num_dev=settings["hampel_num_dev"],
                    )["filtered"]
                    for fasc_l in fasc_l_all
                ]
                pennation_all_filtered = [
                    hampelFilterList(
                        pennation,
                        win_size=settings["hampel_window_size"],
                        num_dev=settings["hampel_num_dev"],
                    )["filtered"]
                    for pennation in pennation_all
                ]

            else:
                fasc_l_all_filtered = [
                    applyFilters(fasc_l, filter_type=settings["selected_filter"])
                    for fasc_l in fasc_l_all
                ]
                pennation_all_filtered = [
                    applyFilters(pennation, filter_type=settings["selected_filter"])
                    for pennation in pennation_all
                ]

            # Save Results
            exportToExcel(
                rootpath,
                fasc_l_all,
                pennation_all,
                x_lows_all,
                x_highs_all,
                thickness_all,
                gui.filename,
                fasc_l_all_filtered,
                pennation_all_filtered,
            )

    except IndexError as e:
        tk.messagebox.showerror(
            "Information",
            f"No Aponeurosis detected. Change aponeurosis threshold. + \n{str(e)}",
        )
        gui.should_stop = False
        gui.is_running = False
        gui.do_break()

    # except Exception as e:
    #     error_message = f"An error occurred:\n{str(e)}"
    #     print(error_message)
    #     tk.messagebox.showerror("Error", error_message)
    #     gui.should_stop = False
    #     gui.is_running = False
    #     gui.do_break()

    finally:
        # clean up
        gui.should_stop = False
        gui.is_running = False


def calculateArchitectureVideoManual(videopath: str, gui):
    """Function used for manual calculation of fascicle length, muscle
    thickness and pennation angles in longitudinal ultrasonography videos
    of human lower limb muscles.

    This function is not restricted to any specific muscles. However, its use
    is restricted to a specific method for assessing muscle thickness fascicle
    length and pennation angles. Moreover, each video frame is analyzed
    seperately.

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
    videopath : str
        String variable containing the absolute path to the video
        to be analyzed.
    gui : tk.TK
        A tkinter.TK class instance that represents a GUI. By passing this
        argument, interaction with the GUI is possible i.e., stopping
        the calculation process after each image frame.

    Notes
    -----
    For specific explanations of the included functions see the respective
    function docstrings in this module. The function outputs an .xlsx
    file in rootpath containing the (averaged) analysis results for each
    image.

    Examples
    --------
    >>> calculateBatchManual(videopath="C:/Users/admin/Dokuments/videoss",
                             gui")
    """
    try:
        # load video
        imported = importVideoManual(videopath)
        cap, vid_len, filename = imported

        # Create folder for frames as they are first stored seperately
        file_path = os.path.dirname(videopath)
        frame_cap = os.path.join(file_path, "frames")

        if not os.path.isdir(frame_cap):
            os.mkdir(frame_cap)

        # Loop through single frames and turn to images
        for a in range(0, vid_len - 1):

            if gui.should_stop:
                # there was an input to stop the calculations
                break

            # FORMAT EACH FRAME, RESHAPE AND COMPUTE NN PREDICTIONS
            _, frame = cap.read()

            # Turn frames to images
            frame_path = os.path.join(frame_cap, f"{filename}{str(a)}.tif")
            cv2.imwrite(frame_path, frame)

        # Get list of all frames
        list_of_frames = glob.glob(frame_cap + "/**/*.tif", recursive=True)

        # annotate thickness, fasicles and angles
        man_analysis = ManualAnalysis(list_of_frames, file_path)
        man_analysis.calculateBatchManual()

    except IndexError:
        tk.messagebox.showerror("Information", "Make sure to select a video file.")
        gui.should_stop = False
        gui.is_running = False
        gui.do_break()

    except FileNotFoundError:
        tk.messagebox.showerror(
            "Information", "Make sure the video file path is correct."
        )
        gui.should_stop = False
        gui.is_running = False
        gui.do_break()

    finally:
        # clean up
        gui.should_stop = False
        gui.is_running = False
