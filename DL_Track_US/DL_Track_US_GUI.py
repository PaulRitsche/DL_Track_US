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
import os

import tkinter as tk
from threading import Lock, Thread
from tkinter import E, N, S, StringVar, Tk, W, filedialog, ttk

from DL_Track_US import gui_helpers


class DLTrack:
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

    def __init__(self, root):

        # set up threading
        self._lock = Lock()
        self._is_running = False
        self._should_stop = False

        # set up gui
        root.title("DL_Track_US")
        master_path = os.path.dirname(os.path.abspath(__file__))
        iconpath = master_path + "/home_im.ico"
        root.iconbitmap(iconpath)

        self.main = ttk.Frame(root, padding="10 10 12 12")
        self.main.grid(column=0, row=0, sticky=(N, S, W, E))
        # Configure resizing of user interface
        self.main.columnconfigure(0, weight=1)
        self.main.columnconfigure(1, weight=1)
        self.main.columnconfigure(2, weight=1)
        self.main.columnconfigure(3, weight=1)
        self.main.columnconfigure(4, weight=1)
        self.main.columnconfigure(5, weight=1)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        # Style
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="DarkSeaGreen3")
        style.configure(
            "TLabel",
            font=("Lucida Sans", 12),
            foreground="black",
            background="DarkSeaGreen3",
        )
        style.configure(
            "TRadiobutton",
            background="DarkSeaGreen3",
            foreground="black",
            font=("Lucida Sans", 12),
        )
        style.configure(
            "TButton",
            background="papaya whip",
            foreground="black",
            font=("Lucida Sans", 11),
        )
        style.configure(
            "TEntry",
            font=("Lucida Sans", 12),
            background="papaya whip",
            foregrund="black",
        )
        style.configure("TCombobox", background="sea green",
                        foreground="black")

        # Entryboxes
        # Input directory
        self.input = StringVar()
        input_entry = ttk.Entry(self.main, width=30, textvariable=self.input)
        input_entry.grid(column=2, row=6, columnspan=3, sticky=(W, E))
        self.input.set("C:/Users/admin/Documents")

        # Apo Model path
        self.apo_model = StringVar()
        apo_model_entry = ttk.Entry(self.main, width=30,
                                    textvariable=self.apo_model)
        apo_model_entry.grid(column=2, row=7, columnspan=3, sticky=(W, E))
        self.apo_model.set("C:/Users/admin/Documents")

        # Fasc Model path
        self.fasc_model = StringVar()
        fasc_model_entry = ttk.Entry(self.main, width=30,
                                     textvariable=self.fasc_model)
        fasc_model_entry.grid(column=2, row=8, columnspan=3, sticky=(W, E))
        self.fasc_model.set("C:/Users/admin/Documents")

        # Radiobuttons
        # Analysis Type
        self.analysis_type = StringVar()
        image = ttk.Radiobutton(
            self.main,
            text="Image",
            variable=self.analysis_type,
            value="image",
            command=self.image_analysis,
        )
        image.grid(column=2, row=10, sticky=(W, E))
        video = ttk.Radiobutton(
            self.main,
            text="Video",
            variable=self.analysis_type,
            value="video",
            command=self.video_analysis,
        )
        video.grid(column=3, row=10, sticky=(W, E))
        image_manual = ttk.Radiobutton(
            self.main,
            text="Image Manual",
            variable=self.analysis_type,
            value="image_manual",
            command=self.image_manual,
        )
        image_manual.grid(column=2, row=11, sticky=(W, E))
        video_manual = ttk.Radiobutton(
            self.main,
            text="Video Manual",
            variable=self.analysis_type,
            value="video_manual",
            command=self.video_manual,
        )
        video_manual.grid(column=3, row=11, sticky=(W, E))

        # Buttons
        # Input directory
        input_button = ttk.Button(self.main, text="Input",
                                  command=self.get_input_dir)
        input_button.grid(column=5, row=6, sticky=E)

        # Apo model path
        apo_model_button = ttk.Button(
            self.main, text="Apo Model", command=self.get_apo_model_path
        )
        apo_model_button.grid(column=5, row=7, sticky=E)

        # Fasc model path
        fasc_model_button = ttk.Button(
            self.main, text="Fasc Model", command=self.get_fasc_model_path
        )
        fasc_model_button.grid(column=5, row=8, sticky=E)

        # Break button
        break_button = ttk.Button(self.main, text="Break",
                                  command=self.do_break)
        break_button.grid(column=2, row=20, sticky=(W, E))

        # Run button
        run_button = ttk.Button(self.main, text="Run", command=self.run_code)
        run_button.grid(column=3, row=20, sticky=(W, E))

        # Model training button
        training = ttk.Button(
            self.main, text="Train Model", command=self.train_model_window
        )
        training.grid(column=5, row=20, sticky=E)

        # Labels
        ttk.Label(self.main, text="Directories", font=("Verdana", 14)).grid(
            column=1, row=5, sticky=(W, E)
        )
        ttk.Label(self.main, text="Input Directory").grid(column=1, row=6)
        ttk.Label(self.main, text="Apo Model Path").grid(column=1, row=7)
        ttk.Label(self.main, text="Fasc Model Path").grid(column=1, row=8)
        ttk.Label(self.main, text="Analysis Type", font=("Verdana", 14)).grid(
            column=1, row=9, sticky=(W, E)
        )

        for child in self.main.winfo_children():
            child.grid_configure(padx=5, pady=5)

    # Methods used in main GUI window when respective buttons are pressed.

    # Determine input directory
    def get_input_dir(self):
        """Instance method to ask the user to select the input directory.
        All image files (of the same specified filetype) in
        the input directory are analysed.
        """
        input_dir = filedialog.askdirectory()
        self.input.set(input_dir)

    # Get path of aponeurosis model
    def get_apo_model_path(self):
        """Instance method to ask the user to select the apo model path.
        This must be an absolute path and the model must be a .h5 file.
        """
        apo_model_dir = filedialog.askopenfilename()
        self.apo_model.set(apo_model_dir)

    # Get path of fascicle model
    def get_fasc_model_path(self):
        """Instance method to ask the user to select the fascicle model path.
        This must be an absolute path and the model must be a .h5 file.
        """
        fasc_model_dir = filedialog.askopenfilename()
        self.fasc_model.set(fasc_model_dir)

    # Analysis type selection

    def image_analysis(self):
        """Instance method to display the required parameters
        that need to be entered by the user when images
        are automatically analyzed.

        Several parameters are displayed:
        - Image type:
        The user can enter the type of the image or
        select the examples from the dropdown list.
        The formatting must be kept constant.
        - Scaling type:The user can select the type of scaling used.
        The options are bar (automatic scaling based on
        scaling bars present in the right side of the image),
        manual (manual scaling required) or no-scaling.
        - Spacing (mm):
        If scaling==bar or scaling==manual, the distance
        between the scaling bars/points selected by the user
        must be specified. This can be 5, 10, 15 or 20 mm.
        - Flip File Path:
        The user must specify the absolute path to the
        .txt file containing the flip flag for each image
        contained in the input directory. The flag should
        be 0 when the fascicles are correctly orientated
        (from bottom left to top right) and 1 otherwise.
        - Analysis Parameter:
        The user must specify the analysis parameters
        used for computation of fascicle length, muscle
        thickness and pennation angle. The parameters
        will have an influence on computation outcomes.
        """
        # Labels
        ttk.Label(self.main, text="Image Properties",
                  font=("Verdana", 14)).grid(
            column=1, row=9, sticky=(W, E)
        )
        ttk.Label(self.main, text="Image Type").grid(column=1, row=13)
        ttk.Label(self.main, text="Scaling Type").grid(column=1, row=14)
        ttk.Label(self.main, text="Spacing (mm)").grid(column=1, row=15)
        ttk.Label(self.main, text="Flip File Path").grid(column=1, row=16)

        # Hide unnecessary widgets
        ttk.Label(self.main, text="                       ").grid(
            column=3, row=13, columnspan=6, sticky=(W, E), ipady=7
        )
        ttk.Label(self.main, text="                        ").grid(
            column=1, row=17, columnspan=6, sticky=(W, E), ipady=7
        )
        ttk.Label(self.main, text="                        ").grid(
            column=1, row=18, columnspan=6, sticky=(W, E), ipady=7
        )
        # Radiobuttons
        # Image Type
        self.scaling = StringVar()
        static = ttk.Radiobutton(
            self.main, text="Bar", variable=self.scaling, value="Bar"
        )
        static.grid(column=2, row=14, sticky=(W, E))
        manual = ttk.Radiobutton(
            self.main, text="Manual", variable=self.scaling, value="Manual"
        )
        manual.grid(column=3, row=14, sticky=(W, E))
        no_scaling = ttk.Radiobutton(
            self.main, text="No Scaling", variable=self.scaling, value="No"
        )
        no_scaling.grid(column=4, row=14, sticky=(W, E))
        self.scaling.set("Bar")

        # Comboboxes
        # Filetype
        self.filetype = StringVar()
        filetype = (
            "/**/*.tif",
            "/**/*.tiff",
            "/**/*.png",
            "/**/*.bmp",
            "/**/*.jpeg",
            "/**/*.jpg",
        )
        filetype_entry = ttk.Combobox(self.main, width=10,
                                      textvariable=self.filetype)
        filetype_entry["values"] = filetype
        # filetype_entry["state"] = "readonly"
        filetype_entry.grid(column=2, row=13, sticky=(W, E))
        self.filetype.set("/**/*.tiff")

        # Spacing
        self.spacing = StringVar()
        spacing = (5, 10, 15, 20)
        spacing_entry = ttk.Combobox(self.main, width=10,
                                     textvariable=self.spacing)
        spacing_entry["values"] = spacing
        spacing_entry["state"] = "readonly"
        spacing_entry.grid(column=2, row=15, sticky=(W, E))
        self.spacing.set(10)

        # Buttons
        # Flipfile model path
        flipfile_button = ttk.Button(
            self.main, text="Flip Flags", command=self.get_flipfile_path
        )
        flipfile_button.grid(column=5, row=16, sticky=E)

        # Analysis parameter button
        analysis = ttk.Button(
            self.main, text="Analysis Parameters", command=self.open_window
        )
        analysis.grid(column=5, row=17, sticky=E, pady=5)

        # Entry
        # Flip File path
        self.flipflag = StringVar()
        flipflag_entry = ttk.Entry(self.main, width=30,
                                   textvariable=self.flipflag)
        flipflag_entry.grid(column=2, row=16, columnspan=2, sticky=(W, E))
        self.flipflag.set("Desktop/DL_Track/FlipFlags.txt")

    def video_analysis(self):
        """Instance method to display the required parameters
        that need to be entered by the user when videos
        are automatically analyzed. Several parameters are
        displayed:
        - Video type:
        The user can enter the type of the image or
        select the examples from the dropdown list.
        The formatting must be kept constant.
        - Scaling type:
        The user can select the type of scaling used.
        The options are manual (manual scaling required) or
        no-scaling. Manual scaling must be performed in
        the first frame only.
        - Spacing (mm):
        If scaling==bar or scaling==manual, the distance
        between the scaling bars/points selected by the user
        must be specified. This can be 5, 10, 15 or 20 mm.
        - Flip Options:
        The user must specify whether the whole video
        should be vertially flipped. This might be necessary
        when the fascicle orientation is different from the one
        our models were trained on. If the fascicles are
        oriented from bottom left to top right "no_flip" should
        be selected. If not, "flip" must be selected.
        - Analysis Parameter:
        The user must specify the analysis parameters
        used for computation of fascicle length, muscle
        thickness and pennation angle. The parameters
        will have an influence on computation outcomes.
        """
        # Reset flipping variable for Video
        self.flip = None

        # Labels
        ttk.Label(self.main, text="Video Properties ",
                  font=("Verdana", 14)).grid(
            column=1, row=9, sticky=(W, E)
        )
        ttk.Label(self.main, text="Video Type").grid(column=1, row=13)
        tk.Label(
            self.main,
            text="Scaling Type",
            font=("Lucida Sans", 13),
            foreground="black",
            background="DarkSeaGreen3",
        ).grid(column=1, row=14)
        ttk.Label(self.main, text="Spacing (mm)").grid(column=1, row=15)
        ttk.Label(self.main, text="Flip Options").grid(column=1, row=16)
        ttk.Label(self.main, text="Frame Steps").grid(column=1, row=17)

        # Remove unnecessary widgets
        ttk.Label(self.main, text="                        ").grid(
            column=3, row=13, columnspan=6, sticky=(W, E), ipady=7
        )
        ttk.Label(self.main, text="                        ").grid(
            column=4, row=14, columnspan=6, sticky=(W, E), ipady=7
        )
        ttk.Label(self.main, text="                        ").grid(
            column=2, row=15, columnspan=5, sticky=(W, E)
        )
        ttk.Label(self.main, text="                        ").grid(
            column=5, row=16, columnspan=3, sticky=(W, E), ipady=7
        )
        ttk.Label(self.main, text="                        ").grid(
            column=5, row=17, columnspan=3, sticky=(W, E), ipady=7
        )
        # Radiobuttons
        # Video Type
        self.scaling = StringVar()
        manual = ttk.Radiobutton(
            self.main, text="Manual", variable=self.scaling, value="Manual"
        )
        manual.grid(column=2, row=14, sticky=(W, E))
        no_scaling = ttk.Radiobutton(
            self.main, text="No Scaling", variable=self.scaling, value="No"
        )
        no_scaling.grid(column=3, row=14, sticky=(W, E))
        self.scaling.set("Manual")

        # Flip
        self.flip = StringVar()
        flip_y = ttk.Radiobutton(
            self.main, text="Flip", variable=self.flip, value="flip"
        )
        flip_y.grid(column=2, row=16, sticky=(W, E))
        flip_n = ttk.Radiobutton(
            self.main, text="Don't flip", variable=self.flip, value="no_flip"
        )
        flip_n.grid(column=3, row=16, sticky=(W, E))
        self.flip.set("flip")

        # Comboboxes
        # Filetype
        self.filetype = StringVar()
        filetype = ("/**/*.avi", "/**/*.mp4")
        filetype_entry = ttk.Combobox(self.main, width=10,
                                      textvariable=self.filetype)
        filetype_entry["values"] = filetype
        # filetype_entry["state"] = "readonly"
        filetype_entry.grid(column=2, row=13, sticky=(W, E))
        self.filetype.set("/**/*.avi")

        # Spacing
        self.spacing = StringVar()
        spacing = (5, 10, 15, 20)
        spacing_entry = ttk.Combobox(self.main, width=10,
                                     textvariable=self.spacing)
        spacing_entry["values"] = spacing
        spacing_entry["state"] = "readonly"
        spacing_entry.grid(column=2, row=15, sticky=(W, E))
        self.spacing.set(10)

        # Step
        self.step = StringVar()
        step_entry = ttk.Combobox(self.main, width=10, textvariable=self.step)
        step = (1, 3, 5, 10)
        step_entry["values"] = step
        step_entry.grid(column=2, row=17, sticky=(W, E))
        self.step.set(1)

        # Buttons
        # Analysis parameter button
        analysis = ttk.Button(
            self.main, text="Analysis Parameters", command=self.open_window
        )
        analysis.grid(column=5, row=18, sticky=W, pady=5)

    def image_manual(self):
        """Instance method to display the required parameters
        that need to be entered by the user when images are
        evaluated manually.
        - Image type:
        The user can enter the type of the image or
        select the examples from the dropdown list.
        The formatting must be kept constant.
        """
        # Labels
        ttk.Label(self.main, text="Image Properties",
                  font=("Verdana", 14)).grid(
            column=1, row=9, sticky=(W, E)
        )
        ttk.Label(self.main, text="  Image Type  ").grid(column=1, row=13)

        # Remove unnecessary widgets
        ttk.Label(self.main, text="                       ").grid(
            column=3, columnspan=2, row=13, sticky=(W, E)
        )
        ttk.Label(self.main, text="                       ").grid(
            column=1, columnspan=5, row=14, sticky=(W, E)
        )
        ttk.Label(self.main, text="                       ").grid(
            column=1, columnspan=5, row=15, sticky=(W, E)
        )
        ttk.Label(self.main, text="                        ").grid(
            column=1, columnspan=6, row=16, sticky=(W, E), ipady=7
        )
        ttk.Label(self.main, text="                        ").grid(
            column=1, columnspan=6, row=17, sticky=(W, E), ipady=7
        )
        ttk.Label(self.main, text="                        ").grid(
            column=1, row=18, columnspan=6, sticky=(W, E), ipady=7
        )

        # Comboboxes
        # Filetype
        self.filetype = StringVar()
        filetype = (
            "/**/*.tif",
            "/**/*.tiff",
            "/**/*.png",
            "/**/*.bmp",
            "/**/*.jpeg",
            "/**/*.jpg",
            "/**/*.avi",
        )
        filetype_entry = ttk.Combobox(self.main, width=10,
                                      textvariable=self.filetype)
        filetype_entry["values"] = filetype
        # filetype_entry["state"] = "readonly"
        filetype_entry.grid(column=2, row=13, sticky=(W, E))
        self.filetype.set("/**/*.tif")

    def video_manual(self):
        """Instance method to display the required parameters
        that need to be entered by the user when videos are
        evaluated manually.
        - File path:
                    The user must specify the absolute path to the
                    video file to be analyzed.
        """
        # Labels
        ttk.Label(self.main, text="Video Properties ",
                  font=("Verdana", 14)).grid(
            column=1, row=9, sticky=(W, E)
        )
        ttk.Label(self.main, text="  File Path  ").grid(column=1, row=13)

        # Remove unnecessary widgets
        ttk.Label(self.main, text="                       ").grid(
            column=1, columnspan=5, row=14, sticky=(W, E)
        )
        ttk.Label(self.main, text="                       ").grid(
            column=1, columnspan=5, row=15, sticky=(W, E)
        )
        ttk.Label(self.main, text="                        ").grid(
            column=1, columnspan=6, row=16, sticky=(W, E), ipady=7
        )
        ttk.Label(self.main, text="                        ").grid(
            column=1, columnspan=6, row=17, sticky=(W, E), ipady=7
        )
        ttk.Label(self.main, text="                        ").grid(
            column=1, row=18, columnspan=6, sticky=(W, E), ipady=7
        )

        # Entries
        self.video = StringVar()
        video_entry = ttk.Entry(self.main, width=30, textvariable=self.video)
        video_entry.grid(column=2, row=13, columnspan=3, sticky=(W, E))
        self.video.set("C:/Users/admin/Documents/fasc_video.avi")

        # Buttons
        # Get video path
        vpath = ttk.Button(self.main, text="Video Path",
                           command=self.get_video_path)
        vpath.grid(column=5, row=13, sticky=E)

    def get_flipfile_path(self):
        """Instance method to ask the user to select the flipfile path.
        The flipfile should contain the flags used for flipping each
        image. If 0, the image is not flipped, if 1 the image is
        flipped. This must be an absolute path.
        """
        flipflag_dir = filedialog.askopenfilename()
        self.flipflag.set(flipflag_dir)

    def get_video_path(self):
        """Instance method to ask the user to select the video path
        for manual video analysis.
        This must be an absolute path.
        """
        video_path = filedialog.askopenfilename()
        self.video.set(video_path)
        return video_path

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

            # Get input dir
            selected_input_dir = self.input.get()

            # Make sure some kind of input directory is specified.
            if len(selected_input_dir) < 3:
                tk.messagebox.showerror("Information",
                                        "Input directory is incorrect.")
                self.should_stop = False
                self.is_running = False
                self.do_break()
                return

            # Get selected analysis
            selected_analysis = self.analysis_type.get()

            # Start thread depending on Analysis type
            if selected_analysis == "image":

                selected_filetype = self.filetype.get()

                # Make sure some kind of filetype is specified.
                if len(selected_filetype) < 3:
                    tk.messagebox.showerror("Information",
                                            "Filetype is invalid.")
                    self.should_stop = False
                    self.is_running = False
                    self.do_break()
                    return

                selected_flipflag_path = self.flipflag.get()
                selected_apo_model_path = self.apo_model.get()
                selected_fasc_model_path = self.fasc_model.get()
                selected_scaling = self.scaling.get()
                selected_spacing = self.spacing.get()
                selected_apo_threshold = self.apo_threshold.get()
                selected_fasc_threshold = self.fasc_threshold.get()
                selected_fasc_cont_threshold = self.fasc_cont_threshold.get()
                selected_min_width = self.min_width.get()
                selected_min_pennation = self.min_pennation.get()
                selected_max_pennation = self.max_pennation.get()
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
                        float(selected_apo_threshold),
                        float(selected_fasc_threshold),
                        int(selected_fasc_cont_threshold),
                        int(selected_min_width),
                        int(selected_min_pennation),
                        int(selected_max_pennation),
                        self,
                    ),
                )
            elif selected_analysis == "video":

                selected_filetype = self.filetype.get()

                # Make sure some kind of filetype is specified.
                if len(selected_filetype) < 3:
                    tk.messagebox.showerror("Information",
                                            "Filetype is invalid.")
                    self.should_stop = False
                    self.is_running = False
                    self.do_break()
                    return

                selected_step = self.step.get()

                # Make sure some kind of step is specified.
                if len(selected_step) < 1 or int(selected_step) < 1:
                    tk.messagebox.showerror("Information",
                                            "Frame Steps is invalid.")
                    self.should_stop = False
                    self.is_running = False
                    self.do_break()
                    return

                selected_flip = self.flip.get()
                selected_apo_model_path = self.apo_model.get()
                selected_fasc_model_path = self.fasc_model.get()
                selected_scaling = self.scaling.get()
                selected_spacing = self.spacing.get()
                selected_apo_threshold = self.apo_threshold.get()
                selected_fasc_threshold = self.fasc_threshold.get()
                selected_fasc_cont_threshold = self.fasc_cont_threshold.get()
                selected_min_width = self.min_width.get()
                selected_min_pennation = self.min_pennation.get()
                selected_max_pennation = self.max_pennation.get()
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
                        float(selected_apo_threshold),
                        float(selected_fasc_threshold),
                        int(selected_fasc_cont_threshold),
                        int(selected_min_width),
                        int(selected_min_pennation),
                        int(selected_max_pennation),
                        self,
                    ),
                )
            elif selected_analysis == "image_manual":

                selected_filetype = self.filetype.get()

                # Make sure some kind of filetype is specified.
                if len(selected_filetype) < 3:
                    tk.messagebox.showerror("Information",
                                            "Filetype is invalid.")
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

        # Error handling
        except AttributeError:
            tk.messagebox.showerror(
                "Information",
                "Check input parameters."
                + "\nPotential error sources:"
                + "\n - Invalid specified directory."
                "\n - Analysis Type not set" +
                "\n - Analysis parameters not set.",
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
                "Information", "Analysis parameter entry fields" +
                " must not be empty."
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

    # ---------------------------------------------------------------------------------------------------
    # Open new toplevel instance for analysis parameter specification

    def open_window(self):
        """Instance method to open new window for analysis parameter input.
        The window is opened upon pressing of the "analysis parameters"
        button.

        Several parameters are displayed.
        - Apo Threshold:
        The user must input the aponeurosis threshold
        either by selecting from the dropdown list or
        entering a value. By varying this threshold, different
        structures will be classified as aponeurosis as the threshold for
        classifying a pixel as aponeurosis is changed. Float, be non-zero and
        non-negative.
        - Fasc Threshold:
        The user must input the fascicle threshold
        either by selecting from the dropdown list or
        entering a value. By varying this threshold, different
        structures will be classified as fascicle as the threshold for
        classifying a pixel as fascicle is changed.
        Float, must be non-negative and non-zero.
        - Fasc Cont Threshold:
        The user must input the fascicle contour threshold
        either by selecting from the dropdown list or
        entering a value. By varying this threshold, different
        structures will be classified as fascicle. By increasing, longer
        fascicle segments will be considered, by lowering shorter segments.
        Integer, must be non-zero and non-negative.
        - Minimal Width:
        The user must input the minimal with either by selecting from the
        dropdown list or entering a value. The aponeuroses must be at least
        this distance apart to be detected. The distance is specified in
        pixels. Integer, must be non-zero and non-negative.
        - Min Pennation:
        The user must enter the minimal pennation angle physiologically apt
        occuring in the analyzed image/video. Fascicles with lower pennation
        angles will be excluded. The pennation angle is calculated as the angle
        of insertion between extrapolated fascicle and detected aponeurosis.
        Integer, must be non-negative.
        - Max Pennation:
        The user must enter the minimal pennation angle physiologically apt
        occuring in the analyzed image/video. Fascicles with lower pennation
        angles will be excluded. The pennation angle is calculated as the angle
        of insertion between extrapolated fascicle and detected aponeurosis.
        Integer, must be non-negative.

        The parameters are set upon pressing the "set parameters" button.
        """
        # Create window
        window = tk.Toplevel(bg="DarkSeaGreen3")
        window.title("Analysis Parameter Window")
        # Add icon to window
        window_path = os.path.dirname(os.path.abspath(__file__))
        iconpath = window_path + "/home_im.ico"
        window.iconbitmap(iconpath)
        window.grab_set()

        # Labels
        ttk.Label(window, text="Analysis Parameters",
                  font=("Verdana", 14)).grid(
            column=1, row=11, padx=10
        )
        ttk.Label(window, text="Apo Threshold").grid(column=1, row=12)
        ttk.Label(window, text="Fasc Threshold").grid(column=1, row=13)
        ttk.Label(window, text="Fasc Cont Threshold").grid(column=1, row=14)
        ttk.Label(window, text="Minimal Width").grid(column=1, row=15)
        ttk.Label(window, text="Minimal Pennation").grid(column=1, row=17)
        ttk.Label(window, text="Maximal Pennation").grid(column=1, row=18)

        # Apo threshold
        self.apo_threshold = StringVar()
        athresh = (0.1, 0.3, 0.5, 0.7, 0.9)
        apo_entry = ttk.Combobox(window, width=10,
                                 textvariable=self.apo_threshold)
        apo_entry["values"] = athresh
        apo_entry.grid(column=2, row=12, sticky=(W, E))
        self.apo_threshold.set(0.2)

        # Fasc threshold
        self.fasc_threshold = StringVar()
        fthresh = [0.1, 0.3, 0.5]
        fasc_entry = ttk.Combobox(window, width=10,
                                  textvariable=self.fasc_threshold)
        fasc_entry["values"] = fthresh
        fasc_entry.grid(column=2, row=13, sticky=(W, E))
        self.fasc_threshold.set(0.05)

        # Fasc cont threshold
        self.fasc_cont_threshold = StringVar()
        fcthresh = (20, 30, 40, 50, 60, 70, 80)
        fasc_cont_entry = ttk.Combobox(
            window, width=10, textvariable=self.fasc_cont_threshold
        )
        fasc_cont_entry["values"] = fcthresh
        fasc_cont_entry.grid(column=2, row=14, sticky=(W, E))
        self.fasc_cont_threshold.set(40)

        # Minimal width
        self.min_width = StringVar()
        mwidth = (20, 30, 40, 50, 60, 70, 80, 90, 100)
        width_entry = ttk.Combobox(window, width=10,
                                   textvariable=self.min_width)
        width_entry["values"] = mwidth
        width_entry.grid(column=2, row=15, sticky=(W, E))
        self.min_width.set(60)

        # Minimal pennation
        self.min_pennation = StringVar()
        min_pennation_entry = ttk.Entry(
            window, width=10, textvariable=self.min_pennation
        )
        min_pennation_entry.grid(column=2, row=17, sticky=(W, E))
        self.min_pennation.set(10)

        # Maximal pennation
        self.max_pennation = StringVar()
        max_pennation_entry = ttk.Entry(
            window, width=10, textvariable=self.max_pennation
        )
        max_pennation_entry.grid(column=2, row=18, sticky=(W, E))
        self.max_pennation.set(40)

        # Set Params button
        set_params = ttk.Button(window, text="Set parameters",
                                command=window.destroy)
        set_params.grid(column=1, row=19, sticky=(W, E))

        # Add padding
        for child in window.winfo_children():
            child.grid_configure(padx=5, pady=5)

    # ---------------------------------------------------------------------------------------------------
    # Open new toplevel instance for model training

    def train_model_window(self):
        """Instance method to open new window for model training.
        The window is opened upon pressing of the "analysis parameters"
        button.

        Several parameters are displayed.
        - Image Directory:
        The user must select or input the image directory. This
        path must to the directory containing the training images.
        Images must be in RGB format.
        - Mask Directory:
        The user must select or input the mask directory. This
        path must to the directory containing the training images.
        Masks must be binary.
        - Output Directory:
        The user must select or input the mask directory. This
        path must lead to the directory where the trained model
        and the model weights should be saved.
        - Batch Size:
        The user must input the batch size used during model training by
        selecting from the dropdown list or entering a value.
        Although a larger batch size has advantages during model trainig,
        the images used here are large. Thus, the larger the batch size,
        the more compute power is needed or the longer the training duration.
        Integer, must be non-negative and non-zero.
        - Learning Rate:
        The user must enter the learning rate used for model training by
        selecting from the dropdown list or entering a value.
        Float, must be non-negative and non-zero.
        - Epochs:
        The user must enter the number of Epochs used during model training by
        selecting from the dropdown list or entering a value.
        The total amount of epochs will only be used if early stopping
        does not happen.
        Integer, must be non-negative and non-zero.
        - Loss Function:
        The user must enter the loss function used for model training by
        selecting from the dropdown list. So far, this can be "BCE" only
        (binary cross-entropy).

        Model training is started by pressing the "start training" button.
        Although all parameters relevant for model training can be adapted,
        we advise users with limited experience to keep the pre-defined
        settings. These settings are best practice and devised from the
        original papers that proposed the models used here.
        Singularly the batch size should be adapted to 1 if comupte power is
        limited (no GPU or GPU with RAM lower than 8 gigabyte).
        """
        # Open Window
        window = tk.Toplevel(bg="DarkSeaGreen3")
        window.title("DL_Track_US - Model Training")
        # Add icon to window
        window_path = os.path.dirname(os.path.abspath(__file__))
        iconpath = window_path + "/home_im.ico"
        window.iconbitmap(iconpath)
        window.grab_set()

        # Labels
        ttk.Label(window, text="Training Parameters",
                  font=("Verdana", 14)).grid(
            column=1, row=0, padx=10
        )
        ttk.Label(window, text="Image Directory").grid(column=1, row=2)
        ttk.Label(window, text="Mask Directory").grid(column=1, row=3)
        ttk.Label(window, text="Output Directory").grid(column=1, row=4)
        ttk.Label(window, text="Batch Size").grid(column=1, row=5)
        ttk.Label(window, text="Learning Rate").grid(column=1, row=6)
        ttk.Label(window, text="Epochs").grid(column=1, row=7)
        ttk.Label(window, text="Loss Function").grid(column=1, row=8)

        # Entryboxes
        # Train image directory
        self.train_image_dir = StringVar()
        train_image_entry = ttk.Entry(
            window, width=30, textvariable=self.train_image_dir
        )
        train_image_entry.grid(column=2, row=2, columnspan=3, sticky=(W, E))
        self.train_image_dir.set(
            "C:/Users/"
        )

        # Mask directory
        self.mask_dir = StringVar()
        mask_entry = ttk.Entry(window, width=30, textvariable=self.mask_dir)
        mask_entry.grid(column=2, row=3, columnspan=3, sticky=(W, E))
        self.mask_dir.set(
            "C:/Users/"
        )

        # Output path
        self.out_dir = StringVar()
        out_entry = ttk.Entry(window, width=30, textvariable=self.out_dir)
        out_entry.grid(column=2, row=4, columnspan=3, sticky=(W, E))
        self.out_dir.set("C:/Users/")

        # Buttons
        # Train image button
        train_img_button = ttk.Button(window, text="Images",
                                      command=self.get_train_dir)
        train_img_button.grid(column=5, row=2, sticky=E)

        # Mask button
        mask_button = ttk.Button(window, text="Masks",
                                 command=self.get_mask_dir)
        mask_button.grid(column=5, row=3, sticky=E)

        # Input directory
        out_button = ttk.Button(window, text="Output",
                                command=self.get_output_dir)
        out_button.grid(column=5, row=4, sticky=E)

        # Model train button
        model_button = ttk.Button(
            window, text="Start Training", command=self.train_model
        )
        model_button.grid(column=5, row=10, sticky=E)

        # Comboboxes
        # Batch size
        self.batch_size = StringVar()
        size = ("1", "2", "3", "4", "5", "6")
        size_entry = ttk.Combobox(window, width=10,
                                  textvariable=self.batch_size)
        size_entry["values"] = size
        size_entry.grid(column=2, row=5, sticky=(W, E))
        self.batch_size.set("1")

        # Learning rate
        self.learn_rate = StringVar()
        learn = ("0.005", "0.001", "0.0005", "0.0001", "0.00005", "0.00001")
        learn_entry = ttk.Combobox(window, width=10,
                                   textvariable=self.learn_rate)
        learn_entry["values"] = learn
        learn_entry.grid(column=2, row=6, sticky=(W, E))
        self.learn_rate.set("0.00001")

        # Number of training epochs
        self.epochs = StringVar()
        epoch = ("30", "40", "50", "60", "70", "80")
        epoch_entry = ttk.Combobox(window, width=10, textvariable=self.epochs)
        epoch_entry["values"] = epoch
        epoch_entry.grid(column=2, row=7, sticky=(W, E))
        self.epochs.set("3")

        # Loss function
        self.loss_function = StringVar()
        loss = ("BCE")
        loss_entry = ttk.Combobox(window, width=10,
                                  textvariable=self.loss_function)
        loss_entry["values"] = loss
        loss_entry["state"] = "readonly"
        loss_entry.grid(column=2, row=8, sticky=(W, E))
        self.loss_function.set("BCE")

        # Add padding
        for child in window.winfo_children():
            child.grid_configure(padx=5, pady=5)

    # Methods used for model training

    def get_train_dir(self):
        """Instance method to ask the user to select the training image
        directory path. All image files (of the same specified filetype) in
        the directory are analysed. This must be an absolute path.
        """
        train_image_dir = filedialog.askdirectory()
        self.train_image_dir.set(train_image_dir)

    def get_mask_dir(self):
        """Instance method to ask the user to select the training mask
        directory path. All mask files (of the same specified filetype) in
        the directory are analysed.The mask files and the corresponding
        image must have the exact same name. This must be an absolute path.
        """
        mask_dir = filedialog.askdirectory()
        self.mask_dir.set(mask_dir)

    def get_output_dir(self):
        """Instance method to ask the user to select the output
        directory path. Here, all file created during model
        training (model file, weight file, graphs) are saved.
        This must be an absolute path.
        """
        out_dir = filedialog.askdirectory()
        self.out_dir.set(out_dir)

    def train_model(self):
        """Instance method to execute the model training when the
        "start training" button is pressed.

        By pressing the button, a seperate thread is started
        in which the model training is run. This allows the user to break any
        training process at certain stages. When the analysis can be
        interrupted, a tk.messagebox opens asking the user to either
        continue or terminate the analysis. Moreover, the threading allows
        interaction with the GUI during ongoing analysis process.
        """
        try:
            # See if GUI is already running
            if self.is_running:
                # don't run again if it is already running
                return
            self.is_running = True

            # Get input paremeter
            selected_images = self.train_image_dir.get()
            selected_masks = self.mask_dir.get()
            selected_outpath = self.out_dir.get()

            # Make sure some kind of filetype is specified.
            if (
                len(selected_images) < 3
                or len(selected_masks) < 3
                or len(selected_outpath) < 3
            ):
                tk.messagebox.showerror("Information",
                                        "Specified directories invalid.")
                self.should_stop = False
                self.is_running = False
                self.do_break()
                return

            selected_batch_size = int(self.batch_size.get())
            selected_learning_rate = float(self.learn_rate.get())
            selected_epochs = int(self.epochs.get())
            selected_loss_function = self.loss_function.get()

            # Start thread
            thread = Thread(
                target=gui_helpers.trainModel,
                args=(
                    selected_images,
                    selected_masks,
                    selected_outpath,
                    selected_batch_size,
                    selected_learning_rate,
                    selected_epochs,
                    selected_loss_function,
                    self,
                ),
            )

            thread.start()

        # Error handling
        except ValueError:
            tk.messagebox.showerror(
                "Information", "Analysis parameter entry fields" +
                " must not be empty."
            )
            self.do_break()
            self.should_stop = False
            self.is_running = False

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
# Function required to run the GUI frm the prompt


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
    root = Tk()
    DLTrack(root)
    root.mainloop()


# This statement is required to execute the GUI by typing
# 'python DL_Track_US_GUI.py' in the prompt
# when navigated to the folder containing the file and all dependencies.
if __name__ == "__main__":
    root = Tk()
    DLTrack(root)
    root.mainloop()
