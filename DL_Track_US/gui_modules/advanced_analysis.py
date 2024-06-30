"""

"""

import os
import platform
import customtkinter as ctk
from tkinter import ttk, W, E, N, S, StringVar, BooleanVar, filedialog
from DL_Track_US import gui_helpers
import importlib


class AdvancedAnalysis:
    """
    Function to open a toplevel where masks
    can either be created for training purposes or
    can be inspected subsequent to labelling.
    """

    def __init__(self, parent):

        self.parent = parent
        self.parent.load_settings()

        # Open Window
        self.advanced_window = ctk.CTkToplevel(fg_color="#808080")
        self.advanced_window.title("Advanced Methods Window")

        head_path = os.path.dirname(os.path.abspath(__file__))
        iconpath = head_path + "/gui_helpers/home_im.ico"
        self.advanced_window.iconbitmap(iconpath)

        # if platform.startswith("win"):
        #     self.a_window.after(200, lambda: self.a_window.iconbitmap(iconpath))

        self.advanced_window.grab_set()

        ctk.CTkLabel(self.advanced_window, text="Select Method").grid(column=1, row=0)

        # Mask Option
        self.advanced_option = StringVar()
        advanced_entry = ["Train Model", "Inspect Masks"]
        advanced_entry = ctk.CTkComboBox(
            self.advanced_window,
            width=20,
            variable=self.advanced_option,
            state="readonly",
            values=advanced_entry,
        )
        advanced_entry.grid(column=1, row=1, sticky=(W, E))
        self.advanced_option.set("...")
        self.advanced_option.trace_add("write", self.on_mask_change)

        # Add padding
        for child in self.advanced_window.winfo_children():
            child.grid_configure(padx=5, pady=5)

    def on_mask_change(self, *args):
        """
        Depending on which mask opration is selected,
        this function adapts the GUI.

        Instance method to open new window for model training.
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
        The total amount of epochs will only be used if early stopping does not happen.
        Integer, must be non-negative and non-zero.
        - Loss Function:
        The user must enter the loss function used for model training by
        selecting from the dropdown list. These can be "BCE" (binary
        cross-entropy), "Dice" (Dice coefficient) or "FL"(Focal loss).

        Model training is started by pressing the "start training" button. Although
        all parameters relevant for model training can be adapted, we advise users with
        limited experience to keep the pre-defined settings. These settings are best
        practice and devised from the original papers that proposed the models used
        here. Singularly the batch size should be adapted to 1 if comupte power is limited
        (no GPU or GPU with RAM lower than 8 gigabyte).

        There is an "Augment Images" button, which allows to generate new training images.
        The images and masks for the data augmentation are taken from the chosen image directory
        and mask directory. The new images are saved under the same directories.
        """
        # make new frame
        self.advanced_window_frame = ctk.CTkFrame(
            self.advanced_window,
        )
        self.advanced_window_frame.grid(column=1, row=2, sticky=(N, S, W, E))

        try:

            if self.advanced_option.get() == "Inspect Masks":

                if hasattr(self, "mask_button"):
                    self.mask_button.destroy()

                # Train image directory
                self.raw_image_dir = StringVar()
                dir1_button = ctk.CTkButton(
                    self.advanced_window_frame,
                    text="Image Dir",
                    command=lambda: (self.raw_image_dir.set(filedialog.askdirectory())),
                )
                dir1_button.grid(column=0, row=2, sticky=(W, E))

                # Mask directory
                self.mask_image_dir = StringVar()
                self.dir2_button = ctk.CTkButton(
                    self.advanced_window_frame,
                    text="Mask Dir",
                    command=lambda: (
                        self.mask_image_dir.set(filedialog.askdirectory()),
                        ctk.CTkLabel(
                            self.advanced_window_frame, text=f"{self.mask_image_dir}"
                        ).grid(column=0, row=3),
                    ),
                )
                self.dir2_button.grid(column=1, row=2, sticky=(W, E))

                # Start index
                self.start_label = ctk.CTkLabel(
                    self.advanced_window_frame, text="Start at Image:"
                )
                self.start_label.grid(column=0, row=4, sticky=W)
                start_idx = StringVar()
                self.idx = ttk.Entry(
                    self.advanced_window_frame, width=10, textvariable=start_idx
                )
                self.idx.grid(column=1, row=4, sticky=W)
                start_idx.set("0")

                ttk.Separator(self.advanced_window_frame, orient="horizontal").grid(
                    column=1, row=5, columnspan=4
                )

                # Inspect button
                self.inspect_button = ctk.CTkButton(
                    self.advanced_window_frame,
                    text="Inspect Masks",
                    command=lambda: (
                        self.advanced_window.destroy(),
                        gui_helpers.find_outliers(
                            dir1=self.raw_image_dir.get(),
                            dir2=self.mask_image_dir.get(),
                        ),
                        gui_helpers.overlay_directory_images(
                            image_dir=self.raw_image_dir.get(),
                            mask_dir=self.mask_image_dir.get(),
                            start_index=int(start_idx.get()),
                        ),
                    ),
                )
                self.inspect_button.grid(column=1, row=6, sticky=(W, E))

            elif self.advanced_option.get() == "Create Masks":

                # Forget widgets
                if hasattr(self, "train_image_dir"):
                    for widget in self.advanced_window_frame.winfo_children():
                        widget.destroy()

                if hasattr(self, "dir2_button"):
                    self.dir2_button.destroy()
                    self.mask_image_entry.destroy()
                    self.inspect_button.destroy()
                    self.idx.destroy()
                    self.start_label.destroy()

                # Train image directory
                self.raw_image_dir = StringVar()
                image_entry = ttk.Entry(
                    self.advanced_window_frame,
                    width=30,
                    textvariable=self.raw_image_dir,
                )
                image_entry.grid(column=1, row=2, columnspan=2, sticky=(W, E))
                self.raw_image_dir.set("Select Raw Image Directory")

                dir1_button = ctk.CTkButton(
                    self.advanced_window_frame,
                    text="Image Dir",
                    command=lambda: (self.raw_image_dir.set(filedialog.askdirectory())),
                )
                dir1_button.grid(column=3, row=2, sticky=(W, E))

                # Train image directory
                self.raw_image_dir = StringVar()
                mask_entry = ttk.Entry(
                    self.advanced_window_frame,
                    width=30,
                    textvariable=self.raw_image_dir,
                )
                mask_entry.grid(column=1, row=2, columnspan=2, sticky=(W, E))
                self.raw_image_dir.set("Select Raw Image Directory")

                # Mask button
                self.mask_button = ctk.CTkButton(
                    self.advanced_window_frame,
                    text="Create Masks",
                    command=lambda: (
                        self.advanced_window.destroy(),
                        gui_helpers.create_acsa_masks(
                            input_dir=self.raw_image_dir.get(), muscle_name="image"
                        ),
                    ),
                )
                self.mask_button.grid(column=2, row=3, sticky=(W, E))

            elif self.advanced_option.get() == "Train Model":
                # Labels
                ctk.CTkLabel(
                    self.advanced_window_frame,
                    text="Training Directories",
                    font=("Verdana", 14),
                ).grid(column=1, row=0, padx=10)
                ctk.CTkLabel(self.advanced_window_frame, text="Image Directory").grid(
                    column=1, row=2
                )
                ttk.Label(self.advanced_window_frame, text="Mask Directory").grid(
                    column=1, row=3
                )
                ttk.Label(self.advanced_window_frame, text="Output Directory").grid(
                    column=1, row=4
                )

                ttk.Label(
                    self.advanced_window_frame,
                    text="Hyperparameters",
                    font=("Verdana", 14),
                ).grid(column=1, row=6, padx=10)
                ttk.Label(self.advanced_window_frame, text="Batch Size").grid(
                    column=1, row=7
                )
                ttk.Label(self.advanced_window_frame, text="Learning Rate").grid(
                    column=1, row=8
                )
                ttk.Label(self.advanced_window_frame, text="Epochs").grid(
                    column=1, row=9
                )
                ttk.Label(self.advanced_window_frame, text="Loss Function").grid(
                    column=1, row=10
                )

                # Entryboxes
                # Train image directory
                self.train_image_dir = StringVar()
                train_image_entry = ttk.Entry(
                    self.advanced_window_frame,
                    width=30,
                    textvariable=self.train_image_dir,
                )
                train_image_entry.grid(column=2, row=2, columnspan=3, sticky=(W, E))
                self.train_image_dir.set("C:/Users/admin/Documents/DeepACSA")

                # Mask directory
                self.mask_dir = StringVar()
                mask_entry = ttk.Entry(
                    self.advanced_window_frame, width=30, textvariable=self.mask_dir
                )
                mask_entry.grid(column=2, row=3, columnspan=3, sticky=(W, E))
                self.mask_dir.set("C:/Users/admin/Documents/DeepACSA")

                # Output path
                self.out_dir = StringVar()
                out_entry = ttk.Entry(
                    self.advanced_window_frame, width=30, textvariable=self.out_dir
                )
                out_entry.grid(column=2, row=4, columnspan=3, sticky=(W, E))
                self.out_dir.set("C:/Users/admin/Documents/DeepACSA")

                # Buttons
                # Train image button
                train_img_button = ttk.Button(
                    self.advanced_window_frame,
                    text="Images",
                    command=self.get_train_dir,
                )
                train_img_button.grid(column=5, row=2, sticky=E)

                # Mask button
                mask_button = ttk.Button(
                    self.advanced_window_frame, text="Masks", command=self.get_mask_dir
                )
                mask_button.grid(column=5, row=3, sticky=E)

                # Data augmentation button
                data_augmentation_button = ttk.Button(
                    self.advanced_window_frame,
                    text="Augment Images",
                    command=self.augment_images,
                )
                data_augmentation_button.grid(column=4, row=12, sticky=E)

                # Input directory
                out_button = ttk.Button(
                    self.advanced_window_frame,
                    text="Output",
                    command=self.get_output_dir,
                )
                out_button.grid(column=5, row=4, sticky=E)

                # Model train button
                model_button = ttk.Button(
                    self.advanced_window_frame,
                    text="Start Training",
                    command=self.train_model,
                )
                model_button.grid(column=5, row=12, sticky=E)

                # Comboboxes
                # Batch size
                self.batch_size = StringVar()
                size = ("1", "2", "3", "4", "5", "6")
                size_entry = ttk.Combobox(
                    self.advanced_window_frame, width=10, textvariable=self.batch_size
                )
                size_entry["values"] = size
                size_entry.grid(column=2, row=7, sticky=(W, E))
                self.batch_size.set("1")

                # Learning rate
                self.learn_rate = StringVar()
                learn = ("0.005", "0.001", "0.0005", "0.0001", "0.00005", "0.00001")
                learn_entry = ttk.Combobox(
                    self.advanced_window_frame, width=10, textvariable=self.learn_rate
                )
                learn_entry["values"] = learn
                learn_entry.grid(column=2, row=8, sticky=(W, E))
                self.learn_rate.set("0.00001")

                # Number of training epochs
                self.epochs = StringVar()
                epoch = ("30", "40", "50", "60", "70", "80")
                epoch_entry = ttk.Combobox(
                    self.advanced_window_frame, width=10, textvariable=self.epochs
                )
                epoch_entry["values"] = epoch
                epoch_entry.grid(column=2, row=9, sticky=(W, E))
                self.epochs.set("3")

                # Loss function
                self.loss_function = StringVar()
                loss = "BCE"
                loss_entry = ttk.Combobox(
                    self.advanced_window_frame,
                    width=10,
                    textvariable=self.loss_function,
                )
                loss_entry["values"] = loss
                loss_entry["state"] = "readonly"
                loss_entry.grid(column=2, row=10, sticky=(W, E))
                self.loss_function.set("BCE")

                # Seperators
                ttk.Separator(
                    self.advanced_window_frame, orient="horizontal", style="TSeparator"
                ).grid(column=0, row=5, columnspan=9, sticky=(W, E))
                ttk.Separator(
                    self.advanced_window_frame, orient="horizontal", style="TSeparator"
                ).grid(column=0, row=11, columnspan=9, sticky=(W, E))

        except FileNotFoundError:
            tk.messagebox.showerror("Information", "Enter the correct folder path!")

        # Add padding
        for child in self.advanced_window_frame.winfo_children():
            child.grid_configure(padx=5, pady=5)
