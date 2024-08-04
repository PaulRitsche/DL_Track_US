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
import tkinter as tk
from threading import Lock, Thread
from tkinter import Canvas, E, N, S, StringVar, Tk, W, filedialog, ttk

import customtkinter as ctk
import matplotlib
import matplotlib.pyplot as plt
import settings
from gui_modules import AdvancedAnalysis
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from PIL import Image

from DL_Track_US import gui_helpers

# from DL_Track_US import settings
# from DL_Track_US.gui_modules import AdvancedAnalysis


matplotlib.use("TkAgg")


class DLTrack(ctk.CTk):
    """
    Python class to automatically or manually annotate longitudinal muscle
    ultrasonography images/videos of human lower limb muscles.
    An analysis tkinter GUI is opened upon initialization of the class.
    By clicking the buttons, the user can switch between different
    analysis modes for image/video analysis and model training.
    The GUI consists of the following elements.
    - Input Directory:
    By pressing the "Input" button, the user is asked to select
    an input directory containing all images/videos to be
    analyzed. This can also be entered directly in the entry field.
    - Apo Model Path:
    By pressing the "Apo Model" button, the user is asked to select
    the aponeurosis model used for aponeurosis segmentation. The absolute
    model path can be entered directly in the enty field as well.
    - Fasc Model Path:
    By pressing the "Fasc Model" button, the user is asked to select
    the fascicle model used for aponeurosis segmentation. The absolute
    model path can be entered directly in the enty field as well.
    - Analysis Type:
    The analysis type can be selected. There are four analysis types,
    the selection of which will trigger more analysis parameters
    to be displayed.
    Image (automatic image analysis), Video (automatic video analysis),
    Image Manual (manual image analysis), Video Manual (manual image analysis).
    - Break:
    By pressing the "break" button, the user can stop the analysis process
    after each finished image or image frame analysis.
    - Run:
    By pressing the "run" button, the user can start the analysis process.
    - Model training:
    By pressing the "train model" button, a new window opens and the
    user can train an own neural network based on existing/own
    training data.

    Attributes
    ----------
    self._lock : _thread.lock
        Thread object to lock the self._lock variable for access by another
        thread.
    self._is_running : bool, default = False
        Boolen variable determining the active state
        of the GUI. If False, the is not running. If True
        the GUI is running.
    self._should_stop : bool, default = False
        Boolen variable determining the active state
        of the GUI. If False, the is allowed to continue running. If True
        the GUI is stopped and background processes deactivated.
    self.main : tk.TK
        tk.TK instance which is the base frame for the GUI.
    self.input : tk.Stringvar
        tk.Stringvariable containing the path to the input directory.
    self.apo_model : tk.Stringvar
        tk.Stringvariable containing the path to the aponeurosis
        model.
    self.fasc_model : tk.Stringvar
        tk.Stringvariable containing the path to the fascicle
        model.
    self.analysis_type : {"image", "video", "image_manual", "video_manual"}
        tk.Stringvariable containing the selected analysis type.
        This can be "image", "video", "image_manual", "video_manual".
    self.scaling : {"bar", "manual","no scaling"}
        tk.Stringvariable containing the selected scaling type.
        This can be "bar", "manual" or "no scaling".
    self.filetype : tk.Stringvar
        tk.Stringvariabel containing the selected filetype for
        the images to be analyzed. The user can select from the
        dopdown list or enter an own filetype. The formatting
        should be kept constant.
    self.spacing : {10, 5, 15, 20}
        tk.Stringvariable containing the selected spacing distance
        used for computation of pixel / cm ratio. This must only be
        specified when the analysis type "bar" or "manual" is selected.
    self.flipflag : tk.Stringvar
        tk.Stringvariable containing the path to the file with the flip
        flags for each image in the input directory.
    self.flip : {"no_flip", "flip"}
        tk.Stringvariable determining wheter all frames in the video
        file will be flipped during automated analysis of videos. This
        can be "no_flip" or "flip".
    self.video : tk.Stringvar
        tk.Stringvariable containing the absolute path to the video
        file being analyzed during manual video analysis.
    self.apo_threshold : tk.Stringvar
        tk.Stringvariable containing the threshold applied to predicted
        aponeurosis pixels by our neural networks. Must be non-zero and
        non-negative.
    self.fasc_threshold : tk.Stringvar
        tk.Stringvariable containing the threshold applied to predicted
        fascicle pixels by our neural networks. Must be non-zero and
        non-negative.
    self.fasc_cont_threshold : tk.Stringvar
        tk.Stringvariable containing the threshold applied to predicted
        fascicle segments by our neural networks. Must be non-zero and
        non-negative.
    self.min_width : tk.stringvar
        tk.Stringvariablecontaining the minimal distance between aponeuroses
        to be detected. Must be non-zero and non-negative.
    self.min_pennation : tk.Stringvar
        tk.Stringvariable containing the mininmal (physiological) acceptable
        pennation angle occuring in the analyzed image/muscle.
        Must be non-negative.
    self.max_pennation : tk.Stringvariable
        tk.Stringvariable containing the maximal (physiological)
        acceptable pennation angle occuring in the analyzed image/muscle.
        Must be non-negative and larger than min_pennation.
    self.train_image_dir : tk.Stringvar
        tk.Straingvar containing the path to the directory of the training
        images. Image must be in RGB format.
    self.mask_path : tk.Stringvar
        tk.Stringvariable containing the path to the directory of the mask
        images. Masks must be binary.
    self.out_dir : tk.Stringvar
        tk.Stringvariable containing the path to the directory where the
        trained model should be saved.
    self.batch_size : tk.Stringvar
        tk.Stringvariable containing the batch size per iteration through the
        network during model training. Must be non-negative and non-zero.
    self.learning_rate : tk.Stringvariable
        tk.Stringvariable the learning rate used during model training.
        Must be non-negative and non-zero.
    self.epochs : tk.Stringvar
        tk.Straingvariable containing the amount of epochs that the model
        is trained befor training is aborted. Must be non-negative and
        non-zero.
    self.loss : tk.Stringvar
        tk.Stringvariable containing the loss function that is used during
        training.

    Methods
    -------
    get_input_dir
        Instance method to ask the user to select the input directory.
    get_apo_model_path
        Instance method to ask the user to select the apo model path.
    get_fasc_model_path
        Instance method to ask the user to select the fascicle model path.
    image_analysis
        Instance method to display the required parameters
        that need to be entered by the user when images
        are automatically analyzed.
    video_analysis
        Instance method to display the required parameters
        that need to be entered by the user when videos
        are automatically analyzed.
    image_manual
        Instance method to display the required parameters
        that need to be entered by the user when images are
        evaluated manually.
    video_manual
        Instance method to display the required parameters
        that need to be entered by the user when videos are
        evaluated manually.
    get_flipfile_path
        Instance method to ask the user to select the flipfile path.
    get_video_path
        Instance method to ask the user to select the video path
        for manual video analysis.
    run_code
        Instance method to execute the analysis process when the
        "run" button is pressed.
    do_break
        Instance method to break the analysis process when the
        button "break" is pressed.
    open_window
        Instance method to open new window for analysis parameter input.
    train_model_window
        Instance method to open new window for model training.
    get_train_dir
        Instance method to ask the user to select the training image
        directory path.
    get_mask_dir
        Instance method to ask the user to select the training mask
        directory path.
    get_output_dir
        Instance method to ask the user to select the output
        directory path.
    train_model
        Instance method to execute the model training when the
        "start training" button is pressed.
    change_analysis_type
        Instance method to change GUI layout based on the analysis type.

    Notes
    -----
    This class contains only instance attributes.
    The instance methods contained in this class are solely purposed for
    support of the main GUI instance method. They cannot be used
    independantly or seperately.

    For more detailed documentation of the functions employed
    in this GUI upon running the analysis or starting model training
    see the respective modules in the /gui_helpers subfolder.

    See Also
    --------
    model_training.py, calculate_architecture.py,
    calculate_architecture_video.py
    """

    def __init__(self, *args, **kwargs):
        """
        Initialization of  GUI window upon calling.

        """
        super().__init__(*args, **kwargs)

        # Load settings
        self.load_settings()

        # set up threading
        self._lock = Lock()
        self._is_running = False
        self._should_stop = False

        # set up gui
        self.title("DL_Track_US")
        master_path = os.path.dirname(os.path.abspath(__file__))
        ctk.set_default_color_theme(
            master_path + "/gui_helpers/gui_files/gui_color_theme.json"
        )

        iconpath = master_path + "/gui_helpers/home_im.ico"
        # root.iconbitmap(iconpath)

        self.main = ctk.CTkFrame(self)
        self.main.grid(column=0, row=0, sticky=(N, S, W, E))

        # Configure resizing of user interface
        for row in range(21):
            self.main.rowconfigure(row, weight=1)
        for column in range(3):
            self.main.rowconfigure(column, weight=1)

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        # Buttons
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

        # Apo model path
        apo_model_button = ctk.CTkButton(
            self.main, text="Apo Model", command=self.get_apo_model_path
        )
        apo_model_button.grid(column=1, row=4, sticky=(W, E))

        # Fasc model path
        fasc_model_button = ctk.CTkButton(
            self.main, text="Fasc Model", command=self.get_fasc_model_path
        )
        fasc_model_button.grid(column=2, row=4, sticky=W)

        # Entryboxes
        # TODO Include analysis in main UI window.

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

        # Image Type Label
        self.image_type_label = ctk.CTkLabel(self.main, text="Image Type")
        self.image_type_label.grid(column=0, row=13)

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
        self.filetype_entry.grid(column=1, row=13, sticky=(W, E))
        self.filetype.set("/*.tiff")

        # Scaling label
        self.scaling_label = ctk.CTkLabel(self.main, text="Scaling Type")
        self.scaling_label.grid(column=0, row=14)

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
        self.scaling_entry.grid(column=1, row=14, sticky=(W, E))
        self.scaling.set("Bar")

        # Spacing label
        self.spacing_label = ctk.CTkLabel(self.main, text="Spacing (mm)")
        self.spacing_label.grid(column=0, row=15)

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
        self.spacing_entry.grid(column=1, row=15, sticky=(W, E))
        self.spacing.set("10")

        # Filter Fascicle label
        self.filter_label = ctk.CTkLabel(self.main, text="Filter Fascicles")
        self.filter_label.grid(column=0, row=16)

        # Fascicle filter buttons
        self.filter_fasc = StringVar()
        self.filter_yes = ctk.CTkRadioButton(
            self.main,
            text="Yes",
            variable=self.filter_fasc,
            value=True,
            state="disabled",
        )
        self.filter_yes.grid(column=1, row=16, sticky=(W, E))
        self.filter_no = ctk.CTkRadioButton(
            self.main,
            text="No",
            variable=self.filter_fasc,
            value=False,
            state="disabled",
        )
        self.filter_no.grid(column=2, row=16, sticky=(W, E))
        self.filter_fasc.set(False)

        # Image Flipping
        self.flip_label = ctk.CTkLabel(self.main, text="Flip File Path")
        self.flip_label.grid(column=0, row=17)

        # Flipfile button
        self.flipfile_button = ctk.CTkButton(
            self.main,
            text="Flip Flags",
            command=self.get_flipfile_path,
            state="disabled",
        )
        self.flipfile_button.grid(column=2, row=17, sticky=W)

        # Flip Combobox for Videos
        self.flip = StringVar()
        self.flip_entry = ctk.CTkComboBox(
            self.main,
            variable=self.flip,
            values=["flip", "no_flip"],
            state="disabled",
        )
        self.flip_entry.grid(column=1, row=17, sticky=(W, E))

        # Stepsize label for Videos
        self.steps_label = ctk.CTkLabel(self.main, text="Step Size")
        self.steps_label.grid(column=0, row=18)

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
        self.step_entry.grid(column=1, row=18, sticky=(W, E))
        self.step.set("1")

        # Break button
        break_button = ctk.CTkButton(self.main, text="Break", command=self.do_break)
        break_button.grid(column=0, row=20, sticky=(W, E))

        # Run button
        run_button = ctk.CTkButton(self.main, text="Run", command=self.run_code)
        run_button.grid(column=1, row=20, sticky=(W, E))

        advanced_button = ctk.CTkButton(
            self.main,
            text="Advanced Methods",
            command=lambda: (AdvancedAnalysis(self),),
            fg_color="#000000",
            text_color="#FFFFFF",
            border_color="yellow3",
        )
        advanced_button.grid(column=2, row=20, sticky=E)

        ttk.Separator(self.main, orient="horizontal", style="TSeparator").grid(
            column=0, row=19, columnspan=9, sticky=(W, E)
        )

        # Settings button
        gear_path = master_path + "/gui_helpers/gui_files/Gear.png"
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

        for child in self.main.winfo_children():
            child.grid_configure(padx=5, pady=5)

        # Make frame for results
        self.results = ctk.CTkFrame(self)
        self.results.grid(column=1, row=0, sticky=(N, S, W, E))
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=4)
        self.results.rowconfigure(0, weight=1)
        self.results.columnconfigure(0, weight=1)

        # Configure rows with a loop
        for row in range(5):
            self.results.rowconfigure(row, weight=1)

        # Create logo canvas figure
        self.logo_canvas = Canvas(
            self.results,
            width=800,
            height=600,
            bg="white",
        )
        self.logo_canvas.grid(
            row=0,
            column=0,
            rowspan=6,
            sticky=(N, S, E, W),
            pady=(5, 0),
        )

        # Load the logo as a resizable matplotlib figure
        logo_path = master_path + "/gui_helpers/gui_files/Gear.png"
        logo = plt.imread(logo_path)
        logo_fig, ax = plt.subplots()
        ax.imshow(logo)
        ax.axis("off")  # Turn off axis
        logo_fig.tight_layout()  # Adjust layout padding

        # Plot the figure in the in_gui_plotting canvas
        self.canvas = FigureCanvasTkAgg(logo_fig, master=self.logo_canvas)
        self.canvas.get_tk_widget().pack(
            expand=True,
            fill="both",
            padx=5,
            pady=5,
        )
        plt.close()
        # This solution is more flexible and memory efficient than previously.

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
        global settings
        self.settings = importlib.reload(settings)

    def open_settings(self):
        """
        Instance Method to open the setting file for.

        Executed when the button "Settings" in master GUI window is pressed.
        A python file is openend containing a dictionary with relevant
        variables that users should be able to customize.
        """
        # Determine relative filepath
        file_path = os.path.dirname(os.path.abspath(__file__)) + "/settings.py"

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
            self.flip_label.configure(text="Video Path")
            self.flipfile_button.configure(
                text="Video Path", command=self.get_video_path, state="normal"
            )
            self.flip_entry.configure(state="disabled")
            self.step_entry.configure(state="disabled")

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
    # Open new toplevel instance for analysis parameter specification------------------------------------------------------------------------------------------

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
        # try:
        if self.is_running:
            # don't run again if it is already running
            return
        self.is_running = True

        # load settings
        self.load_settings()

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
        settings = {
            "aponeurosis_detection_threshold": self.settings.aponeurosis_detection_threshold,
            "aponeurosis_length_threshold": self.settings.aponeurosis_length_threshold,
            "fascicle_detection_threshold": self.settings.fascicle_detection_threshold,
            "fascicle_length_threshold": self.settings.fascicle_length_threshold,
            "minimal_muscle_width": self.settings.minimal_muscle_width,
            "minimal_pennation_angle": self.settings.minimal_pennation_angle,
            "maximal_pennation_angle": self.settings.maximal_pennation_angle,
            "fascicle_calculation_method": self.settings.fascicle_calculation_method,
            "fascicle_contour_tolerance": self.settings.fascicle_contour_tolerance,
            "aponeurosis_distance_tolerance": self.settings.aponeurosis_distance_tolerance,
        }

        # Start thread depending on Analysis type
        if self.analysis_type.get() == "image":

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
            selected_min_width = self.settings.minimal_muscle_width
            selected_min_pennation = self.settings.minimal_pennation_angle
            selected_max_pennation = self.settings.maximal_pennation_angle
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
                    settings,
                    self,
                ),
            )
        elif self.analysis_type.get() == "video":

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

            selected_flip = self.flip.get()
            selected_apo_model_path = self.apo_model
            selected_fasc_model_path = self.fasc_model
            selected_scaling = self.scaling.get()
            selected_spacing = self.spacing.get()
            selected_filter_fasc = self.filter_fasc.get()
            selected_apo_threshold = 0.2
            selected_apo_length_threshold = 600
            selected_fasc_threshold = 0.05
            selected_fasc_cont_threshold = 40
            selected_min_width = 60
            selected_min_pennation = 10
            selected_max_pennation = 40
            thread = Thread(
                target=gui_helpers.calculateArchitectureVideo,
                args=(
                    selected_input_dir,
                    selected_apo_model_path,
                    selected_fasc_model_path,
                    selected_filetype,
                    selected_scaling,
                    selected_flip,
                    int(selected_spacing),
                    int(selected_step),
                    int(selected_filter_fasc),
                    float(selected_apo_threshold),
                    int(selected_apo_length_threshold),
                    float(selected_fasc_threshold),
                    int(selected_fasc_cont_threshold),
                    int(selected_min_width),
                    int(selected_min_pennation),
                    int(selected_max_pennation),
                    self,
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
                tk.messagebox.showerror("Information", "Input directory is incorrect.")
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

        # # Error handling
        # except AttributeError:
        #     tk.messagebox.showerror(
        #         "Information",
        #         "Check input parameters."
        #         + "\nPotential error sources:"
        #         + "\n - Invalid specified directory."
        #         "\n - Analysis Type not set" + "\n - Analysis parameters not set.",
        #     )
        #     self.do_break()
        #     self.should_stop = False
        #     self.is_running = False

        # except FileNotFoundError:
        #     tk.messagebox.showerror(
        #         "Information",
        #         "Check input parameters."
        #         + "\nPotential error source:"
        #         + "\n - Invalid specified directory.",
        #     )
        #     self.do_break()
        #     self.should_stop = False
        #     self.is_running = False

        # except PermissionError:
        #     tk.messagebox.showerror(
        #         "Information",
        #         "Check input parameters."
        #         + "\nPotential error source:"
        #         + "\n - Invalid specified directory.",
        #     )
        #     self.do_break()
        #     self.should_stop = False
        #     self.is_running = False

        # except ValueError:
        #     tk.messagebox.showerror(
        #         "Information", "Analysis parameter entry fields" + " must not be empty."
        #     )
        #     self.do_break()
        #     self.should_stop = False
        #     self.is_running = False

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
    app._state_before_windows_set_titlebar_color = "zoomed"
    app.mainloop()


# This statement is required to execute the GUI by typing
# 'python DL_Track_US_GUI.py' in the prompt
# when navigated to the folder containing the file and all dependencies.
if __name__ == "__main__":
    app = DLTrack()
    app._state_before_windows_set_titlebar_color = "zoomed"
    app.mainloop()
