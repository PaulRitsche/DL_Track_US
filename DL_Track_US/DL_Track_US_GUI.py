"""
Description
-----------
This module contains a class with methods to automatically and manually
annotate longitudinal ultrasonography images and videos. When the class
is initiated,a graphical user interface is opened.
This is the main GUI of the DL_Track package.
From here, the user is able to navigate all functionalities of the package.
These extend the methods in this class. The main functionalities of the GUI
contained in this module are automatic and manual evalution of muscle
ultrasonography images.
Inputted images or videos are analyzed and the parameters muscle fascicle
length, pennation angle and muscle thickness are returned for each image
or video frame.
The parameters are analyzed using convolutional neural networks (U-net, VGG16).
This module and all submodules contained in the /gui_helpers modality
are extensions and improvements of the work presented in Cronin et al. (2020).
There, the core functionalities of this code are already outlined and the
comparability of the model segmentations to manual analysis (current gold
standard) is described. Here, we have improved the code by
integrating everything into a graphical user interface.

Functions scope
---------------
For scope of the functions see class documentation.

Notes
-----
Additional information and usage exaples can be found in the video
tutorials provided for this package.

References
----------
[1] VGG16: Simonyan, Karen, and Andrew Zisserman. “Very deep convolutional networks for large-scale image recognition.” arXiv preprint arXiv:1409.1556 (2014)
[2] U-net: Ronneberger, O., Fischer, P. and Brox, T. "U-Net: Convolutional Networks for Biomedical Image Segmentation." arXiv preprint arXiv:1505.04597 (2015)
[3] DL_Track: Cronin, Neil J. and Finni, Taija and Seynnes, Olivier. "Fully automated analysis of muscle architecture from B-mode ultrasound images with deep learning." arXiv preprint arXiv:https://arxiv.org/abs/2009.04790 (2020)
"""

import importlib
import os
import subprocess
import sys
import glob
import webbrowser
import json

import tkinter as tk
from threading import Lock, Thread
from tkinter import E, N, S, StringVar, W, filedialog, ttk

import customtkinter as ctk
from CTkToolTip import *

# Carla imports
# import gui_helpers
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import cv2
import numpy as np
import pandas as pd

# original imports
from DL_Track_US import gui_helpers

# from DL_Track_US.gui_helpers.gui_files import settings
from DL_Track_US.gui_modules import AdvancedAnalysis

# disable interactive backend
plt.ioff()

# TODO fix setting import


class DLTrack(ctk.CTk):
    """
    DLTrack GUI for Muscle Ultrasound Analysis
    ==========================================

    This module provides a graphical user interface (GUI) for the DL_Track package,
    allowing users to annotate longitudinal ultrasonography images and videos both
    manually and automatically. The GUI facilitates navigation through the package's
    functionalities, including automated and manual evaluation of muscle
    ultrasonography images. It computes muscle fascicle length, pennation angle,
    and muscle thickness using convolutional neural networks (U-Net, VGG16).

    The GUI integrates and improves upon the work of Cronin et al. (2020),
    enhancing usability and accessibility for researchers and practitioners.

    Dependencies
    ------------
    - `tkinter` : Standard GUI toolkit for Python.
    - `customtkinter` : Enhanced GUI styling.
    - `threading` : Enables concurrent execution.
    - `matplotlib` : Used for visualization and rendering plots.
    - `cv2` (OpenCV) : Image processing library.
    - `numpy`, `pandas` : Numerical and data manipulation.
    - `PIL` (Pillow) : Image handling.
    - `glob`, `os`, `sys`, `subprocess` : File and system interactions.

    Classes
    -------
    DLTrack
        A graphical interface for muscle ultrasound annotation, supporting automatic
        and manual segmentation, model training, and analysis.

    Attributes
    ----------
    self._lock : threading.Lock
        Lock object for thread safety.
    self._is_running : bool
        Boolean flag indicating whether the GUI is active.
    self._should_stop : bool
        Boolean flag to stop ongoing processes.
    self.main : tkinter.Tk
        Main GUI window.
    self.input : str
        Directory path for input files.
    self.apo_model : str
        Path to aponeurosis segmentation model.
    self.fasc_model : str
        Path to fascicle segmentation model.
    self.analysis_type : {"image", "video", "image_manual", "video_manual"}
        Selected analysis mode.
    self.scaling : {"bar", "manual", "no scaling"}
        Scaling method.
    self.filetype : str
        Selected file type for analysis.
    self.spacing : int
        Spacing distance for pixel/cm ratio calculations.
    self.flipflag : str
        File path for flip flags.
    self.flip : {"no_flip", "flip"}
        Flip option for video analysis.
    self.video : str
        Path to video file for manual analysis.
    self.apo_threshold : float
        Threshold value for aponeurosis segmentation.
    self.fasc_threshold : float
        Threshold value for fascicle segmentation.
    self.fasc_cont_threshold : float
        Threshold for fascicle continuity detection.
    self.min_width : float
        Minimum detectable distance between aponeuroses.
    self.min_pennation : float
        Minimum acceptable pennation angle.
    self.max_pennation : float
        Maximum acceptable pennation angle.
    self.train_image_dir : str
        Path to training image dataset.
    self.mask_path : str
        Path to binary mask dataset.
    self.out_dir : str
        Directory for saving trained models.
    self.batch_size : int
        Batch size for model training.
    self.learning_rate : float
        Learning rate for model training.
    self.epochs : int
        Number of training epochs.
    self.loss : str
        Loss function used for training.

    Methods
    -------
    __init__()
        Initializes the GUI and sets up event listeners.
    mclick(event)
        Detects mouse clicks for calibration.
    calibrate_distance()
        Computes pixel-to-mm conversion for manual calibration.
    calibrateDistanceManually()
        Opens a manual calibration interface.
    display_results(input_df_raw, input_df_filtered, input_df_medians)
        Displays analysis results in the GUI.
    on_processing_complete()
        Updates UI elements after processing.
    display_frame(item)
        Displays the current image or video frame in the GUI.
    update_frame_by_slider(value)
        Changes displayed frame based on slider value.
    update_slider_range()
        Synchronizes slider with processed frames.
    load_settings()
        Loads user-defined settings.
    open_settings()
        Opens the configuration settings file.
    get_input_dir()
        Opens a file dialog to select the input directory.
    get_apo_model_path()
        Opens a file dialog to select the aponeurosis model file.
    get_fasc_model_path()
        Opens a file dialog to select the fascicle model file.
    change_analysis_type()
        Updates UI elements based on the selected analysis type.
    change_spacing()
        Enables or disables spacing selection based on scaling type.
    get_flipfile_path()
        Opens a file dialog to select the flip flag file.
    get_video_path()
        Opens a file dialog to select the video file for manual analysis.
    should_stop()
        Thread-safe property indicating whether the process should stop.
    is_running()
        Thread-safe property indicating if the process is running.
    run_code()
        Runs the selected analysis in a separate thread.
    do_break()
        Stops the currently running analysis process.
    check_processing_complete(thread, callback)
        Monitors the completion of an analysis process.

    Notes
    -----
    - This GUI is an interactive front-end for DL_Track, ensuring easy access
    to image and video analysis.

    References
    ----------
    [1] Simonyan, K., & Zisserman, A. (2014). “Very deep convolutional networks for
        large-scale image recognition.” arXiv preprint arXiv:1409.1556.
    [2] Ronneberger, O., Fischer, P., & Brox, T. (2015). "U-Net: Convolutional Networks
        for Biomedical Image Segmentation." arXiv preprint arXiv:1505.04597.
    [3] Cronin, N. J., Finni, T., & Seynnes, O. (2020). "Fully automated analysis of
        muscle architecture from B-mode ultrasound images with deep learning."
        arXiv preprint arXiv:2009.04790.
    [4] Ritsche et al., (2023). DL_Track_US: a python package to analyse muscle ultrasonography images.
        Journal of Open Source Software, 8(85), 5206, https://doi.org/10.21105/joss.05206
    """

    def __init__(self, *args, **kwargs):
        """
        Initialization of  GUI window upon calling.

        """
        super().__init__(*args, **kwargs)

        # Load settings
        self.load_settings()
        self.mlocs = []

        # set up threading
        self._lock = Lock()
        self._is_running = False
        self._should_stop = False

        # set up gui
        self.title("DL_Track_US")
        master_path = os.path.dirname(os.path.abspath(__file__))
        ctk.set_default_color_theme(
            self.resource_path("gui_helpers/gui_files/gui_color_theme.json")
        )

        iconpath = self.resource_path("gui_helpers/gui_files/DLTrack_logo.ico")
        self.iconbitmap(iconpath)

        self.main = ctk.CTkFrame(self)
        self.main.grid(column=0, row=0, sticky=(N, S, W, E))

        # Configure resizing of user interface
        for row in range(21):
            self.main.rowconfigure(row, weight=1)
        for column in range(4):
            self.main.rowconfigure(column, weight=1)

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=5)
        self.rowconfigure(0, weight=1)
        self.minsize(width=600, height=400)

        # Docs reference button
        info_path = self.resource_path("gui_helpers/gui_files/Info.png")
        # Get info button path
        self.info = ctk.CTkImage(
            light_image=Image.open(info_path),
            size=(30, 30),
        )
        info_button = ctk.CTkButton(
            self.main,
            image=self.info,
            text="Documentation",
            width=30,
            height=30,
            bg_color="#2A484E",
            fg_color="#2A484E",
            border_width=0,
            command=lambda: (
                (webbrowser.open("https://paulritsche.github.io/DL_Track_US/"))
            ),
        )
        info_button.grid(
            row=1,
            column=0,
            sticky=(
                W,
                E,
            ),
        )

        # citation button
        cite_path = self.resource_path("gui_helpers/gui_files/Cite.png")
        # Get info button path
        self.info = ctk.CTkImage(
            light_image=Image.open(cite_path),
            size=(30, 30),
        )
        cite_button = ctk.CTkButton(
            self.main,
            image=self.info,
            text="Cite us",
            width=30,
            height=30,
            bg_color="#2A484E",
            fg_color="#2A484E",
            border_width=0,
            command=lambda: (
                (webbrowser.open("https://joss.theoj.org/papers/10.21105/joss.05206"))
            ),
        )
        cite_button.grid(
            row=1,
            column=1,
            sticky=(
                W,
                E,
            ),
        )

        # Input directory
        ttk.Separator(self.main, orient="horizontal", style="TSeparator").grid(
            column=0, row=3, columnspan=9, sticky=(W, E)
        )
        ctk.CTkLabel(self.main, text="Directories", font=("Verdana", 20, "bold")).grid(
            column=0, row=3, sticky=(W, E)
        )
        input_button = ctk.CTkButton(
            self.main, text="Inputs", command=self.get_input_dir
        )
        input_button.grid(column=0, row=4, sticky=E)

        tooltip_input = CTkToolTip(
            input_button,
            message="Select the directory containing the input images/videos. \nFiles must be of the type specified in the image/video type entry.",
            delay=0.5,
            bg_color="#A8D8CD",
            text_color="#000000",
            alpha=0.7,
        )

        # Apo model path
        apo_model_button = ctk.CTkButton(
            self.main, text="Apo Model", command=self.get_apo_model_path
        )
        apo_model_button.grid(column=1, row=4, sticky=(W, E))
        tooltip_apo = CTkToolTip(
            apo_model_button,
            message="Select the path to an aponeurosis segmentation model. \nThis should be a .h5 file.",
            delay=0.5,
            bg_color="#A8D8CD",
            text_color="#000000",
            alpha=0.7,
        )

        # Fasc model path
        fasc_model_button = ctk.CTkButton(
            self.main, text="Fasc Model", command=self.get_fasc_model_path
        )
        fasc_model_button.grid(column=2, row=4, sticky=W)
        tooltip_fasc = CTkToolTip(
            fasc_model_button,
            message="Select the path to an fascicle segmentation model. \nThis should be a .h5 file.",
            delay=0.5,
            bg_color="#A8D8CD",
            text_color="#000000",
            alpha=0.7,
        )

        # Analysis Type
        # Separators
        ttk.Separator(self.main, orient="horizontal", style="TSeparator").grid(
            column=0, row=8, columnspan=9, sticky=(W, E)
        )
        ctk.CTkLabel(self.main, text="Analysis", font=("Verdana", 20, "bold")).grid(
            column=0, row=8, sticky=(W, E)
        )

        ctk.CTkLabel(self.main, text="Select:").grid(column=0, row=9, sticky=(W, E))

        self.analysis_type = StringVar()
        analysis_values = ["image", "video", "image_manual", "video_manual"]
        analysis_entry = ctk.CTkComboBox(
            self.main,
            width=150,
            variable=self.analysis_type,
            values=analysis_values,
            state="readonly",
        )
        analysis_entry.grid(column=1, row=9, sticky=(W, E))
        self.analysis_type.trace_add("write", self.change_analysis_type)
        self.analysis_type.set("...")

        tooltip_fasc = CTkToolTip(
            fasc_model_button,
            message="Select an analysis option. \nAvailable analysis parameters will change depending on the analysis type.",
            delay=0.5,
            bg_color="#A8D8CD",
            text_color="#000000",
            alpha=0.7,
        )

        # Image Type Label
        self.image_type_label = ctk.CTkLabel(self.main, text="Image Type")
        self.image_type_label.grid(column=0, row=11)

        # Filetype
        self.filetype = StringVar()
        filetype_values = [
            "/*.tif",
            "/*.tiff",
            "/*.png",
            "/*.bmp",
            "/*.jpeg",
            "/*.jpg",
        ]
        self.filetype_entry = ctk.CTkComboBox(
            self.main,
            width=10,
            variable=self.filetype,
            values=filetype_values,
            state="disabled",
        )
        self.filetype_entry.grid(column=1, row=11, sticky=(W, E))
        self.filetype.set("/*.tiff")

        # Scaling label
        self.scaling_label = ctk.CTkLabel(self.main, text="Scaling Type")
        self.scaling_label.grid(column=0, row=12)

        self.scaling = StringVar()
        scaling_values = [
            "Bar",
            "Manual",
            "None",
        ]
        self.scaling_entry = ctk.CTkComboBox(
            self.main,
            width=10,
            variable=self.scaling,
            values=scaling_values,
            state="disabled",
        )
        self.scaling_entry.grid(column=1, row=12, sticky=(W, E))
        self.scaling.set("Bar")
        self.scaling.trace_add("write", self.change_spacing)

        # Spacing label
        self.spacing_label = ctk.CTkLabel(self.main, text="Spacing (mm)")
        self.spacing_label.grid(column=0, row=13)

        # Spacing Combobox
        self.spacing = StringVar()
        spacing_values = ["5", "10", "15", "20"]
        self.spacing_entry = ctk.CTkComboBox(
            self.main,
            width=10,
            variable=self.spacing,
            values=spacing_values,
            state="disabled",
        )
        self.spacing_entry.grid(column=1, row=13, sticky=(W, E))
        self.spacing.set("10")

        # Flipfile button
        self.spacing_button = ctk.CTkButton(
            self.main,
            text="Calibrate",
            command=self.calibrateDistanceManually,
            state="disabled",
        )
        self.spacing_button.grid(column=2, row=13, sticky=W)

        tooltip_calibrate = CTkToolTip(
            self.spacing_button,
            message="Determine pixel/cm ratio manually. \nClick on two points (with spacing distance apart) \nin the image to calibrate.",
            delay=0.5,
            bg_color="#A8D8CD",
            text_color="#000000",
            alpha=0.7,
        )

        # Filter Fascicle label
        self.filter_label = ctk.CTkLabel(self.main, text="Filter Fascicles")
        self.filter_label.grid(column=0, row=14)

        # Fascicle filter buttons
        self.filter_fasc = StringVar()
        self.filter_yes = ctk.CTkRadioButton(
            self.main,
            text="Yes",
            variable=self.filter_fasc,
            value=True,
            state="disabled",
        )
        self.filter_yes.grid(column=1, row=14, sticky=(W, E))
        self.filter_no = ctk.CTkRadioButton(
            self.main,
            text="No",
            variable=self.filter_fasc,
            value=False,
            state="disabled",
        )
        self.filter_no.grid(column=2, row=14, sticky=(W, E))
        self.filter_fasc.set(False)

        # Image Flipping
        self.flip_label = ctk.CTkLabel(self.main, text="Flip File Path")
        self.flip_label.grid(column=0, row=15)

        # Flipfile button
        self.flipfile_button = ctk.CTkButton(
            self.main,
            text="Flip Flags",
            command=self.get_flipfile_path,
            state="disabled",
        )
        self.flipfile_button.grid(column=2, row=15, sticky=W)

        # Flip Combobox for Videos
        self.flip = StringVar()
        self.flip_entry = ctk.CTkComboBox(
            self.main,
            variable=self.flip,
            values=["flip", "no_flip"],
            state="disabled",
        )
        self.flip_entry.grid(column=1, row=15, sticky=(W, E))

        # Stepsize label for Videos
        self.steps_label = ctk.CTkLabel(self.main, text="Step Size")
        self.steps_label.grid(column=0, row=16)

        # Step Combobox
        self.step = StringVar()
        step_values = ["1", "3", "5", "10"]
        self.step_entry = ctk.CTkComboBox(
            self.main,
            width=10,
            values=step_values,
            variable=self.step,
            state="disabled",
        )
        self.step_entry.grid(column=1, row=16, sticky=(W, E))
        self.step.set("1")

        # Break button
        break_button = ctk.CTkButton(self.main, text="Break", command=self.do_break)
        break_button.grid(column=0, row=18, sticky=(W, E))

        # Run button
        run_button = ctk.CTkButton(self.main, text="Run", command=self.run_code)
        run_button.grid(column=1, row=18, sticky=(W, E))

        advanced_button = ctk.CTkButton(
            self.main,
            text="Advanced Methods",
            command=lambda: (AdvancedAnalysis(self),),
            fg_color="#000000",
            text_color="#FFFFFF",
            border_color="yellow3",
        )
        advanced_button.grid(column=2, row=18, sticky=E)

        ttk.Separator(self.main, orient="horizontal", style="TSeparator").grid(
            column=0, row=17, columnspan=9, sticky=(W, E)
        )

        # Settings button
        gear_path = self.resource_path("gui_helpers/gui_files/gear.png")
        self.gear = ctk.CTkImage(
            light_image=Image.open(gear_path),
            size=(30, 30),
        )

        settings_b = ctk.CTkButton(
            self.main,
            text="",
            image=self.gear,
            command=self.open_settings,
            width=30,
            height=30,
            bg_color="#2A484E",
            fg_color="#2A484E",
            border_width=0,
        )
        settings_b.grid(column=2, row=9, sticky=W, pady=(0, 20))

        tooltip_settings = CTkToolTip(
            settings_b,
            message="Open and change analysis settings.\n Save changes.",
            delay=0.5,
            bg_color="#A8D8CD",
            text_color="#000000",
            alpha=0.7,
        )

        for child in self.main.winfo_children():
            child.grid_configure(padx=5, pady=5)

        # Make frame for results
        self.results = ctk.CTkFrame(self)
        self.results.grid(column=1, row=0, columnspan=3, sticky=(N, S, W, E))
        self.results.bind("<Button-1>", self.mclick)

        logo_path = self.resource_path("gui_helpers/gui_files/DLTrack_logo.png")
        self.logo_img = ImageTk.PhotoImage(Image.open(logo_path))

        self.video_canvas = tk.Label(
            self.results, bg="#2A484E", image=self.logo_img, border=3
        )

        self.video_canvas.grid(
            column=0,
            row=0,
            columnspan=6,
            rowspan=8,
            sticky=(N, S, W, E),
        )

        # Configure rows with a loop
        for row in range(5):
            self.results.rowconfigure(row, weight=1)
        self.results.columnconfigure(0, weight=1)

        # Add a slider for frame navigation
        self.frame_slider = tk.Scale(
            self.results,
            from_=0,
            to=1,
            bg="#2A484E",  # Dark background matching DL_Track
            fg="#FFFFFF",  # White text for contrast
            highlightbackground="#2A484E",  # Matches the main background
            troughcolor="#C49102",  # Trough color for consistency
            activebackground="#4A6A6E",  # Active slider color slightly darker
            sliderrelief=tk.FLAT,  # Flat slider knob
            orient=tk.HORIZONTAL,
            command=self.update_frame_by_slider,
        )
        self.frame_slider.grid(column=0, row=10, columnspan=6, sticky=(W, E, S))

        self.processed_frames = []
        self.current_frame_index = 0

        # Create frame for output
        self.terminal = ctk.CTkFrame(
            self.results,
            fg_color="lightgrey",
            border_width=2,
            border_color="White",
        )
        self.terminal.grid(
            column=0,
            row=11,
            columnspan=6,
            pady=8,
            padx=10,
            sticky=(N, S, W, E),
        )

    def resource_path(self, relative_path):
        """Get absolute path to resource (for dev and PyInstaller)"""
        try:
            base_path = sys._MEIPASS
        except AttributeError:
            base_path = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_path, relative_path)

    def mclick(self, event):
        """Instance method to detect mouse click coordinates in image."""
        self.mlocs.append((event.x, event.y))
        self.calib_canvas.create_oval(
            event.x - 5, event.y - 5, event.x + 5, event.y + 5, fill="red"
        )
        if len(self.mlocs) == 2:
            self.calib_canvas.unbind("<Button-1>")
            self.confirm_button.grid(
                column=0, row=5, columnspan=1, sticky=(E), padx=10, pady=10
            )

    def calibrate_distance(self):
        """Calculate the distance between two points and display the result."""
        if len(self.mlocs) < 2:
            tk.messagebox.showerror(
                "Error", "Please select two points before calibrating."
            )
            return

        x1, y1 = self.mlocs[0]
        x2, y2 = self.mlocs[1]
        self.calib_dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Adjust distance based on spacing input
        spacing = self.spacing.get()
        if spacing == 5:
            self.calib_dist *= 2
        elif spacing == 15:
            self.calib_dist *= 2 / 3
        elif spacing == 20:
            self.calib_dist /= 2

        scale_statement = f"10 mm corresponds to {self.calib_dist:.2f} pixels"
        tk.messagebox.showinfo("Calibration Result", scale_statement)
        self.confirm_button.grid_remove()
        self.calib_canvas.destroy()

        self.mlocs.clear()  # Reset for next calibration

    def calibrateDistanceManually(self):
        """Function to manually calibrate an image to convert measurements
        in pixel units to centimeters."""
        if sys.platform.startswith("darwin"):
            tk.messagebox.showerror(
                "Information",
                "Manual scaling not available on MacOS"
                + "\n Continue with 'No Scaling' Scaling Type.",
            )
            return None, None

        # Initialize canvas
        self.calib_canvas = ctk.CTkCanvas(
            self.results, bg="#2A484E", width=400, height=600
        )
        self.calib_canvas.grid(
            column=0, row=0, columnspan=6, rowspan=8, sticky=(N, S, W, E)
        )

        # Add a confirmation button
        self.confirm_button = ctk.CTkButton(
            self.results,
            text="Confirm",
            command=self.calibrate_distance,
        )
        self.confirm_button.grid(
            column=0, row=5, columnspan=1, sticky=(E), padx=10, pady=10
        )
        self.confirm_button.grid_remove()  # Hide button initially

        # Load first video frame
        list_of_files = glob.glob(self.input_dir + self.filetype.get(), recursive=True)
        if not list_of_files:
            tk.messagebox.showerror("Error", "No video files found.")
            return

        cap = cv2.VideoCapture(list_of_files[0])
        vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # reconfigure canvas dimensions
        self.calib_canvas.config(width=vid_width, height=vid_height)

        ret, frame = cap.read()
        cap.release()

        if not ret:
            tk.messagebox.showerror("Error", "Failed to read video frame.")
            return

        # Convert frame to an image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        self.img_tk = ImageTk.PhotoImage(image=img)  # Store reference

        # Clear previous image if any
        self.calib_canvas.delete("all")

        # Display the image on the canvas
        self.calib_canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)

        # Bind mouse click event
        self.calib_canvas.bind("<Button-1>", self.mclick)

    def display_results(
        self,
        input_df_raw,
        input_df_filtered,
        input_df_medians,
    ):
        """
        Instance method that displays all analysis results in the
        output terminal using Pandastable. Input must be a Pandas dataframe.

        Executed trough functions with calculated anylsis results.

        Parameters
        ----------
        input_df_raw : pd.DataFrame
            Dataftame containing the raw analysis results.
        input_df_filtered : pd.DataFrame
            Dataftame containing the filtered analysis results.
        input_df_medians : pd.DataFrame
            Dataftame containing the filtered median results.
        """
        # Calculate the median length of the fascicles
        # Drop the first column if it's just row indices
        data_raw_cleaned = input_df_raw.iloc[:, 1:]
        data_filtered_cleaned = input_df_filtered.iloc[:, 1:]

        # Compute row-wise median for each DataFrame
        medians_raw = data_raw_cleaned.median(axis=1, skipna=True)
        medians_filtered = data_filtered_cleaned.median(axis=1, skipna=True)
        filtered_medians = input_df_medians.iloc[:, 1:]

        # Create a new figure for the plot
        fig, ax = plt.subplots()
        ax.plot(medians_raw, "o-", label="Median Fascicle Length")
        ax.plot(
            medians_filtered,
            "+-",
            label="Median Filtered Fascicle Length",
            color="green",
        )  # Add filtered lengths in green
        ax.plot(
            filtered_medians,
            "x-",
            label="Filtered Median Fascicle Length",
            color="yellow",
        )  # Add filtered lengths in green
        ax.set_xlabel("Frame")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()

        # Display the plot in the video_canvas
        if hasattr(self, "result_canvas"):
            self.result_canvas.get_tk_widget().destroy()

        self.result_canvas = FigureCanvasTkAgg(fig, master=self.terminal)
        self.result_canvas.draw()
        result_widget = self.result_canvas.get_tk_widget()
        result_widget.grid(column=0, row=0, columnspan=16, sticky=(W, E))
        result_widget.config(height=200)

        # Configure resizing of the figure
        self.terminal.grid_columnconfigure(0, weight=1)
        self.terminal.grid_rowconfigure(0, weight=1)

    def on_processing_complete(self):
        """
        Callback to execute after processing completes.
        Updates the slider range based on the processed frames or figures.
        """
        num_items = len(self.processed_frames)
        if num_items > 0:
            self.frame_slider.config(
                from_=0,
                to=num_items - 1,  # Set the maximum value to the last index
            )
            print(f"Slider updated to range 0-{num_items - 1}.")

        # Load the results into the table after processing
        if hasattr(self, "filename") and self.filename:

            results_file_path = os.path.join(
                self.input_dir, f"{self.filename}.xlsx"
            )  # taken from calculate_architecture.py
        else:
            results_file_path = os.path.join(self.input_dir, "Results.xlsx")

        try:
            if os.path.exists(results_file_path):
                results_data_raw = pd.read_excel(
                    results_file_path, sheet_name="Fasc_length_raw"
                )
                results_data_filtered = pd.read_excel(
                    results_file_path, sheet_name="Fasc_length_filtered"
                )
                results_data_medians = pd.read_excel(
                    results_file_path, sheet_name="Fasc_length_filtered_median"
                )

                self.display_results(
                    results_data_raw, results_data_filtered, results_data_medians
                )
        except:
            print("No results file found.")

    def display_frame(self, item):
        """
        Display the current frame or figure on the GUI canvas.
        """

        if item is None:
            print("No frame or figure to display.")
            return

        if hasattr(self, "calib_canvas"):
            self.calib_canvas.destroy()
            del self.calib_canvas

        # Check if the item is an OpenCV frame (NumPy array)
        if isinstance(item, np.ndarray):

            if hasattr(self, "figure_canvas"):
                self.figure_canvas.get_tk_widget().destroy()

            # Process and display the OpenCV frame
            frame_rgb = cv2.cvtColor(item, cv2.COLOR_BGR2RGB)
            frame_image = Image.fromarray(frame_rgb)
            frame_tk = ImageTk.PhotoImage(image=frame_image, size=(600, 800))

            if not hasattr(self, "video_canvas"):
                self.video_canvas = tk.Label(self.results, bg="#2A484E", border=3)
                self.video_canvas.grid(
                    column=0, row=0, columnspan=6, rowspan=8, sticky=(N, S, W, E)
                )

            self.video_canvas.configure(image=frame_tk)
            self.video_canvas.imgtk = frame_tk

            self.processed_frames.append(item)

        # Check if the item is a matplotlib figure
        elif isinstance(item, plt.Figure):
            # Reuse the existing FigureCanvasTkAgg if it exists
            if hasattr(self, "figure_canvas"):
                self.figure_canvas.figure = item
                self.figure_canvas.draw()
            else:
                # Create a new FigureCanvasTkAgg
                self.figure_canvas = FigureCanvasTkAgg(item, master=self.video_canvas)
                self.figure_canvas.draw()

                # Create a widget for the figure
                figure_widget = self.figure_canvas.get_tk_widget()
                figure_widget.grid(
                    column=0, row=0, columnspan=6, rowspan=8, sticky=(W, E, S, N)
                )

            self.processed_frames.append(item)

        else:
            print("Unsupported item type for display.")

    def update_frame_by_slider(self, value):
        """
        Update the displayed frame based on the slider position.
        """
        index = int(value)
        if 0 <= index < len(self.processed_frames):
            self.current_frame_index = index
            self.display_frame(self.processed_frames[index])
            self.frame_slider.config(label=f"Frame: {self.current_frame_index}")

    def update_slider_range(self):
        """
        Updates the slider to match the number of processed frames or figures.
        """
        num_items = len(self.processed_frames)
        if hasattr(self, "frame_slider"):
            self.frame_slider.destroy()

        self.frame_slider = tk.Scale(
            self.results,
            from_=0,
            to=num_items - 1,
            orient=tk.HORIZONTAL,
            label=f"Item: 0",  # Initial label
            command=self.update_frame_by_slider,
            bg="#2A484E",  # Matches the theme
            fg="#FFFFFF",  # Text color
            troughcolor="#4A6A6E",  # Trough color
            activebackground="#1F3A3D",  # Active color
            highlightbackground="#2A484E",  # Border color
            length=400,  # Adjust length of the slider
        )
        self.frame_slider.grid(column=0, row=2, columnspan=2, sticky=(W, E))

    # Methods used in main GUI window when respective buttons are pressed.
    # Define functionalities for buttons used in GUI master window
    def load_settings(self):
        """
        Instance Method to load the setting file for.

        Executed each time when the GUI or a toplevel is openened.
        The settings specified by the user will then be transferred
        to the code and used.
        """
        # If not previously imported, just import it
        # global settings
        # self.settings = importlib.reload(settings)

        with open(self.resource_path("gui_helpers/gui_files/settings.json"), "r") as f:
            settings = json.load(f)
            self.settings = settings

    def open_settings(self):
        """
        Instance Method to open the setting file for.

        Executed when the button "Settings" in master GUI window is pressed.
        A python file is openend containing a dictionary with relevant
        variables that users should be able to customize.
        """
        # Determine relative filepath
        file_path = self.resource_path("gui_helpers/gui_files/settings.json")
        # file_path = filedialog.askopenfilename(title="Open settings file.")

        # Check for operating system and open in default editor
        if sys.platform.startswith("darwin"):  # macOS
            subprocess.run(["open", file_path])
        elif sys.platform.startswith("win32"):  # Windows
            os.startfile(file_path)
        else:  # Linux or other
            subprocess.run(["xdg-open", file_path])

    # Determine input directory
    def get_input_dir(self):
        """Instance method to ask the user to select the input directory.
        All image files (of the same specified filetype) in
        the input directory are analysed.
        """
        self.input_dir = filedialog.askdirectory()
        ctk.CTkLabel(
            self.main,
            text=f"Folder: {os.path.basename(self.input_dir)}",
            font=("Segue UI", 8, "bold"),
        ).grid(column=0, row=5)

    # Get path of aponeurosis model
    def get_apo_model_path(self):
        """Instance method to ask the user to select the apo model path.
        This must be an absolute path and the model must be a .h5 file.
        """
        self.apo_model = filedialog.askopenfilename(title="Open aponeurosis model.")
        ctk.CTkLabel(
            self.main,
            text=f"{os.path.splitext(os.path.basename(self.apo_model))[0]}",
            font=("Segue UI", 8, "bold"),
        ).grid(column=1, row=5)

    # Get path of fascicle model
    def get_fasc_model_path(self):
        """Instance method to ask the user to select the fascicle model path.
        This must be an absolute path and the model must be a .h5 file.
        """
        self.fasc_model = filedialog.askopenfilename(title="Open fascicle model.")
        ctk.CTkLabel(
            self.main,
            text=f"{os.path.splitext(os.path.basename(self.fasc_model))[0]}",
            font=("Segue UI", 8, "bold"),
        ).grid(column=2, row=5)

    def change_analysis_type(self, *args):
        """
        Unified instance method to display the required parameters
        based on the selected analysis type.
        """
        # Based on the analysis type, display the appropriate widgets
        if self.analysis_type.get() == "image":
            """
            Display the required parameters
            that need to be entered by the user when images
            are automatically analyzed.

            This is the baseline setting which is adapted for
            all other analysis types.
            """
            self.filetype_entry.configure(
                state="normal",
                values=[
                    "/*.tif",
                    "/*.tiff",
                    "/*.png",
                    "/*.bmp",
                    "/*.jpeg",
                    "/*.jpg",
                ],
            )
            self.image_type_label.configure(text="Image Type")
            self.scaling_entry.configure(state="readonly")
            self.spacing_entry.configure(state="normal")
            self.filter_yes.configure(state="normal")
            self.filter_no.configure(state="normal")
            self.flip_label.configure(text="Flip File Path")
            self.flip_entry.configure(state="disabled")
            self.flipfile_button.configure(
                text="Flip Flags", state="normal", command=self.get_flipfile_path
            )
            self.step_entry.configure(state="disabled")

        elif self.analysis_type.get() == "video":
            """
            The filetype widget gets changed to match the
            video file types. The scaling does not have the
            option bar anymore. The Flip Flag button is disabled and the Flip Option dropdown enabled.
            Moreover, the Step Size dropdown is enabled.
            """
            self.image_type_label.configure(text="Video Type")
            self.filetype_entry.configure(state="normal", values=["/*.avi", "/*.mp4"])
            self.scaling_entry.configure(values=["Manual", "None"], state="normal")
            # Reset flipping variable for Video
            self.flip_label.configure(text="Flip Option")
            self.flip_entry.configure(state="normal")
            self.flipfile_button.configure(state="disabled")
            self.scaling_entry.configure(values=["Manual", "None"], state="normal")
            self.step_entry.configure(state="normal")
            self.filter_no.configure(state="normal")
            self.filter_yes.configure(state="normal")

        elif self.analysis_type.get() == "image_manual":
            """
            All widgets are disable except for the
            Image Type dropdown to select the filetype
            for the single images.
            """
            self.filetype_entry.configure(
                state="normal",
                values=[
                    "/*.tif",
                    "/*.tiff",
                    "/*.png",
                    "/*.bmp",
                    "/*.jpeg",
                    "/*.jpg",
                ],
            )
            self.image_type_label.configure(text="Image Type")
            self.scaling_entry.configure(state="disabled")
            self.spacing_entry.configure(state="disabled")
            self.filter_yes.configure(state="disabled")
            self.filter_no.configure(state="disabled")
            self.flipfile_button.configure(state="disabled")
            self.flip_entry.configure(state="disabled")
            self.step_entry.configure(state="disabled")

        elif self.analysis_type.get() == "video_manual":
            """
            All widgets are disable except for the
            Video Path button (otherwise Flip Flags) to
            select the single video for analysis
            """
            self.filetype_entry.configure(state="disabled")
            self.image_type_label.configure(text="Video Type")
            self.scaling_entry.configure(state="disabled")
            self.spacing_entry.configure(state="disabled")
            self.filter_yes.configure(state="disabled")
            self.filter_no.configure(state="disabled")
            self.spacing_button.configure(state="disabled")
            self.flipfile_button.configure(
                text="Video Path", command=self.get_video_path, state="normal"
            )
            self.flip_entry.configure(state="disabled")
            self.step_entry.configure(state="disabled")

    def change_spacing(self, *args):
        """
        Enable the spacing button only when the scaling type is set to "Manual".
        """
        if self.scaling.get() == "Manual":
            self.spacing_button.configure(state="normal")
        else:
            self.spacing_button.configure(state="disabled")

    def get_flipfile_path(self):
        """Instance method to ask the user to select the flipfile path.
        The flipfile should contain the flags used for flipping each
        image. If 0, the image is not flipped, if 1 the image is
        flipped. This must be an absolute path.
        """
        self.flipflag_dir = filedialog.askopenfilename(
            title="Open flip flag file for image flipping",
        )

    def get_video_path(self):
        """Instance method to ask the user to select the video path
        for manual video analysis.
        This must be an absolute path.
        """
        self.video_path = filedialog.askopenfilename(
            title="Open video file for manual analysis", filetypes=[("*.mp4", "*.avi")]
        )

    # ---------------------------------------------------------------------------------------------------
    # Methods and properties required for threading

    @property
    def should_stop(self) -> bool:
        """Instance method to define the should_stop
        property getter method. By defining this as a property,
        should_stop is treated like a public attribute even
        though it is private.

        This is used to stop the analysis process running
        in a seperate thread.

        Returns
        -------
        should_stop : bool
            Boolean variable to decide whether the analysis process
            started from the GUI should be stopped. The process is
            stopped when should_stop = True.
        """
        self._lock.acquire()
        # Get private variable _should_stop
        should_stop = self._should_stop
        self._lock.release()
        return should_stop

    @property
    def is_running(self) -> bool:
        """Instance method to define the is_running
        property getter method. By defining this as a property,
        is_running is treated like a public attribute even
        though it is private.

        This is used to stop the analysis process running
        in a seperate thread.

        Returns
        -------
        is_running : bool
            Boolean variable to check whether the analysis process
            started from the GUI is running. The process is only
            stopped when is_running = True.
        """
        self._lock.acquire()
        is_running = self._is_running
        self._lock.release()
        return is_running

    @should_stop.setter
    def should_stop(self, flag: bool):
        """Instance method to define the should_stop
        property setter method. The setter method is used
        to set the self._should_stop attribute as if it was
        a public attribute. The argument "flag" is thereby
        validated to ensure proper input (boolean)
        """
        self._lock.acquire()
        self._should_stop = flag
        self._lock.release()

    @is_running.setter
    def is_running(self, flag: bool):
        """Instance method to define the is_running
        property setter method. The setter method is used
        to set the self._is_running attribute as if it was
        a public attribute. The argument "flag" is thereby
        validated to ensure proper input (boolean)
        """
        self._lock.acquire()
        self._is_running = flag
        self._lock.release()

    # ---------------------------------------------------------------------------------------------------
    # Function required to run the code

    def run_code(self):
        """Instance method to execute the analysis process when the
        "run" button is pressed.

        Which analysis process is executed depends on the user
        selection. By pressing the button, a seperate thread is started
        in which the analysis is run. This allows the user to break any
        analysis process. Moreover, the threading allows interaction
        with the main GUI during ongoing analysis process. This function
        handles most of the errors occuring during specification of
        file and analysis parameters. All other exeptions are raised in
        other function of this package.

        Raises
        ------
        AttributeError
            The execption is raised when the user didn't specify the
            file or training parameters correctly. A tk.messagebox
            is openend containing hints how to solve the issue.
        FileNotFoundError
            The execption is raised when the user didn't specify the
            file or training parameters correctly. A tk.messagebox
            is openend containing hints how to solve the issue.
        PermissionError
            The execption is raised when the user didn't specify the
            file or training parameters correctly. A tk.messagebox
            is openend containing hints how to solve the issue.
        """
        try:
            if self.is_running:
                # don't run again if it is already running
                return
            self.is_running = True

            # load settings
            self.load_settings()

            # empty processed frames and reset slider
            if hasattr(self, "processed_frames"):
                self.processed_frames = []

            # Get input dir
            selected_input_dir = self.input_dir

            # Make sure some kind of input directory is specified.
            if len(selected_input_dir) < 3:
                tk.messagebox.showerror("Information", "Input directory is incorrect.")
                self.should_stop = False
                self.is_running = False
                self.do_break()
                return

            # Define dictionary containing settings
            # settings = {
            #     "aponeurosis_detection_threshold": self.settings.aponeurosis_detection_threshold,
            #     "aponeurosis_length_threshold": self.settings.aponeurosis_length_threshold,
            #     "fascicle_detection_threshold": self.settings.fascicle_detection_threshold,
            #     "fascicle_length_threshold": self.settings.fascicle_length_threshold,
            #     "minimal_muscle_width": self.settings.minimal_muscle_width,
            #     "minimal_pennation_angle": self.settings.minimal_pennation_angle,
            #     "maximal_pennation_angle": self.settings.maximal_pennation_angle,
            #     "fascicle_calculation_method": self.settings.fascicle_calculation_method,
            #     "fascicle_contour_tolerance": self.settings.fascicle_contour_tolerance,
            #     "aponeurosis_distance_tolerance": self.settings.aponeurosis_distance_tolerance,
            #     "selected_filter": self.settings.selected_filter,
            #     "hampel_window_size": self.settings.hampel_window_size,
            #     "hampel_num_dev": self.settings.hampel_num_dev,
            #     "segmentation_mode": self.settings.segmentation_mode,
            # }

            # Start thread depending on Analysis type
            if self.analysis_type.get() == "image":

                def processing_done_callback():
                    self.on_processing_complete()  # Update slider range here

                selected_filetype = self.filetype.get()

                # Make sure some kind of filetype is specified.
                if len(selected_filetype) < 3:
                    tk.messagebox.showerror("Information", "Filetype is invalid.")
                    self.should_stop = False
                    self.is_running = False
                    self.do_break()
                    return

                # Get relevant UI parameters
                selected_flipflag_path = self.flipflag_dir
                selected_apo_model_path = self.apo_model
                selected_fasc_model_path = self.fasc_model
                selected_scaling = self.scaling.get()
                selected_spacing = self.spacing.get()
                selected_filter_fasc = self.filter_fasc.get()

                thread = Thread(
                    target=gui_helpers.calculateBatch,
                    args=(
                        selected_input_dir,
                        selected_apo_model_path,
                        selected_fasc_model_path,
                        selected_flipflag_path,
                        selected_filetype,
                        selected_scaling,
                        int(selected_spacing),
                        int(selected_filter_fasc),
                        self.settings,
                        self,
                        self.display_frame,
                    ),
                )

            elif self.analysis_type.get() == "video":

                def processing_done_callback():
                    self.on_processing_complete()  # Update slider range here

                selected_filetype = self.filetype.get()

                # Make sure some kind of filetype is specified.
                if len(selected_filetype) < 3:
                    tk.messagebox.showerror("Information", "Filetype is invalid.")
                    self.should_stop = False
                    self.is_running = False
                    self.do_break()
                    return

                selected_step = self.step.get()

                # Make sure some kind of step is specified.
                if len(selected_step) < 1 or int(selected_step) < 1:
                    tk.messagebox.showerror("Information", "Frame Steps is invalid.")
                    self.should_stop = False
                    self.is_running = False
                    self.do_break()
                    return

                if self.scaling.get() == "Manual":
                    self.calibrateDistanceManually()

                selected_flip = self.flip.get()
                selected_apo_model_path = self.apo_model
                selected_fasc_model_path = self.fasc_model
                selected_scaling = self.scaling.get()
                selected_spacing = self.spacing.get()
                selected_filter_fasc = self.filter_fasc.get()

                thread = Thread(
                    target=gui_helpers.calculateArchitectureVideo,
                    args=(
                        selected_input_dir,
                        selected_apo_model_path,
                        selected_fasc_model_path,
                        selected_filetype,
                        selected_scaling,
                        selected_flip,
                        int(selected_step),
                        int(selected_filter_fasc),
                        self.settings,
                        self,
                        self.display_frame,
                    ),
                )
            elif self.analysis_type.get() == "image_manual":

                selected_filetype = self.filetype.get()

                # Make sure some kind of filetype is specified.
                if len(selected_filetype) < 3:
                    tk.messagebox.showerror("Information", "Filetype is invalid.")
                    self.should_stop = False
                    self.is_running = False
                    self.do_break()
                    return

                thread = Thread(
                    target=gui_helpers.calculateBatchManual,
                    args=(
                        selected_input_dir,
                        selected_filetype,
                        self,
                    ),
                )
            else:
                selected_video_path = self.video.get()

                # Make sure some kind of input directory is specified.
                if len(selected_video_path) < 3:
                    tk.messagebox.showerror(
                        "Information", "Input directory is incorrect."
                    )
                    self.should_stop = False
                    self.is_running = False
                    self.do_break()
                    return

                thread = Thread(
                    target=gui_helpers.calculateArchitectureVideoManual,
                    args=(
                        selected_video_path,
                        self,
                    ),
                )

            thread.start()

            self.after(
                100,
                lambda: self.check_processing_complete(
                    thread, processing_done_callback
                ),
            )

        # Error handling
        except AttributeError:
            tk.messagebox.showerror(
                "Information",
                "Check input parameters."
                + "\nPotential error sources:"
                + "\n - Invalid specified directory.",
            )
            self.do_break()
            self.should_stop = False
            self.is_running = False

        except FileNotFoundError:
            tk.messagebox.showerror(
                "Information",
                "Check input parameters."
                + "\nPotential error source:"
                + "\n - Invalid specified directory.",
            )
            self.do_break()
            self.should_stop = False
            self.is_running = False

        except PermissionError:
            tk.messagebox.showerror(
                "Information",
                "Check input parameters."
                + "\nPotential error source:"
                + "\n - Invalid specified directory.",
            )
            self.do_break()
            self.should_stop = False
            self.is_running = False

        except ValueError:
            tk.messagebox.showerror(
                "Information", "Analysis parameter entry fields" + " must not be empty."
            )
            self.do_break()
            self.should_stop = False
            self.is_running = False

    def do_break(self):
        """Instance method to break the analysis process when the
        button "break" is pressed.

        This changes the instance attribute self.should_stop
        to True, given that the analysis is already running.
        The attribute is checked befor every iteration
        of the analysis process.
        """
        if self.is_running:
            self.should_stop = True

    def check_processing_complete(self, thread, callback):
        """
        Checks if the processing thread has finished.
        """
        if thread.is_alive():
            self.after(100, lambda: self.check_processing_complete(thread, callback))

        else:
            callback()


# ---------------------------------------------------------------------------------------------------
# Function required to run the GUI from the prompt


def runMain() -> None:
    """Function that enables usage of the gui from command promt
    as pip package.

    Notes
    -----
    The GUI can be executed by typing 'python -m DL_Track_US_GUI.py' in a
    terminal subsequtently to installing the pip package and activating the
    respective library.

    It is not necessary to download any files from the repository when the pip
    package is installed.

    For documentation of DL_Track see top of this module.
    """
    app = DLTrack()
    # app._state_before_windows_set_titlebar_color = "zoomed"
    app.mainloop()


# This statement is required to execute the GUI by typing
# 'python DL_Track_US_GUI.py' in the prompt
# when navigated to the folder containing the file and all dependencies.
if __name__ == "__main__":
    app = DLTrack()
    # app._state_before_windows_set_titlebar_color = "zoomed"
    app.mainloop()
