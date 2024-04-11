"""
Description
-----------
This module contains a class to manually annotate longitudinal ultrasonography
images and videos. When the class is initiated, a graphical user interface is
opened. There, the user can annotate muscle fascicle length, pennation angle
and muscle thickness. Moreover, the images can be scaled in order to get
measurements in centimeters rather than pixels. By clicking the respective
buttons in the GUI, the user can switch between the different parameters to
analyze. The analysis is not restricted to any specific muscles. However,
its use is restricted to a specific method for assessing muscle thickness,
fascicle length and pennation angles. Moreover, each video frame is analyzed
separately. An .xlsx file is retuned containing the analysis results for
muscle fascicle length, pennation angle and muscle thickness.

Functions scope
---------------
For scope of the functions see class documentation.

Notes
-----
Additional information and usage examples can be found at the respective
functions docstrings.
"""

import math
import os
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
import pandas as pd
from PIL import Image, ImageGrab, ImageTk
from scipy.spatial import distance


class ManualAnalysis:
    """Python class to manually annotate longitudinal muscle
    ultrasonography images/videos of human lower limb muscles.
    An analysis tkinter GUI is opened upon initialization of the class.
    By clicking the buttons, the user can switch between different
    parameters to analyze in the images.

    - Muscle thickness:
                       Exactly one segment reaching from the superficial to
                       the deep aponeuroses of the muscle must be drawn.
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
                      lengths are averaged. Drawing can be started by
                      clickling the left mouse button and keeping it pressed
                      until one segment is finished (mostly where fascicle
                      curvature occurs the other aponeurosis border is
                      reached). Using the euclidean distance, the total
                      fascicle length is computed as a sum of the segments.
    - Pennation angle:
                      Exactly two segments, one along the fascicle
                      orientation, the other along the aponeurosis orientation
                      must be drawn. The line along the aponeurosis must be
                      started where the line along the fascicle ends. If
                      multiple angle are drawn, they are averaged. Drawing
                      can be started by clickling the left mouse button and
                      keeping it pressed until it is not further required to
                      draw the line (i.e., the aponeurosis border is reached
                      by the fascicle). The angle is calculated using the
                      arc-tan function.
    In order to scale the image/video frame, it is required to draw a line
    of length 10 milimeter somewhere in the image. The line can be drawn in
    the same fashion as for example the muscle thickness. Here however, the
    euclidean distance is used to calculate the pixel / centimeter ratio.
    This has to be done for every frame. We also provide the functionality
    to extend the muscle aponeuroses to more easily extrapolate fascicles.
    The lines can be drawn in the same fashion as for example the muscle
    thickness. During the analysis process, care must be taken to not
    accidentally click the left mouse button as those coordinates might
    mess up the results given that calculations are based on a strict
    analysis protocol.

    Attributes
    ----------
    self.image_list : list
        A list variable containing the absolute paths to all images / video to
        be analyzed.
    self.rootpath : str
        Path to root directory where images / videos for the analysis are
        saved.
    self.lines : list
        A list variable containing all lines that are drawn upon the image
        by the user. The list is emptied each time the analyzed parameter
        is changed.
    self.scale_coords : list
        A list variable containing the xy-coordinates coordinates of the
        scaling line start- and endpoints to calculate the distance between.
        The list is emptied each time a new image is scaled.
    self.thick_coords : list
        A list variable containing the xy-coordinates coordinates of the
        muscle thickness line start- and endpoints to calculate the distance
        between. The list is emptied each time a new image is scaled. Only
        the y-coordintes are used for further analysis.
    self.fasc_coords : list
        A list variable containing the xy-coordinates coordinates of the
        fascicle length line segments start- and endpoints to calculate
        the total length of the fascicle. The list is emptied each time a
        new image is analyzed.
    self.pen_coords : list
        A list variable containing the xy-coordinates coordinates of the
        pennation angle line segments start- and endpoints to calculate
        the angle of the fascicle. The list is emptied each time a new
        image is analyzed.
    self.coords : dict
        Dictionary variable storing the xy-coordinates of mouse events
        during analysis. Mouse events are clicking and releasing of the
        left mouse button as well as dragging of the cursor.
    self.count : int, default = 0
        Index of image / video frame currently analysis in the list of
        image file / video frame paths. The default is 0 as the first
        image / frame analyzed always has the idex 0 in the list.
    self.dataframe : pd.DataFrame
        Panadas dataframe that stores the analysis results such as file
        name, fascicle length, pennation angle and muscle thickness.
        This dataframe is then saved in an .xlsx file.
    self.head : tk.TK
        tk.Toplevel instance opening a window containing the manual
        image analysis options.
    self.mode : tk.Stringvar, default = thick
        tk.Stringvar variable containing the current parameter analysed
        by the user. The parameters are
        - muscle thickness : thick
        - fascicle length : fasc
        - pennation angle : pen
        - scaling : scale
        - aponeurosis drawing : apo
        The variable is updaten upon selection of the user. The default
        is muscle thickness.
    self.cavas : tk.Canvas
        tk.Canvas variable representing the canvas the image is plotted
        on in the GUI. The canvas is used to draw on the image.
    self.img : ImageTk.PhotoImage
        ImageTk.PhotoImage variable containing the current image that is
        analyzed. It is necessary to load the image in this way in order
        to plot the image.
    self.dist : int
        Integer variable containing the length of the scaling line in
        pixel units. This variable is then used to scale the image.

    Methods
    -------
    __init__
        Instance method to initialize the class.
    calculateBatchManual
        Instance method creating the GUI for manual image analysis.
    """
    def __init__(self, img_list: str, rootpath: str):
        """Instance method to initialize the Manual Analysis class.

        Parameters
        ----------
        img_list : str
            A list variable containing the absolute paths to all images /
            video to be analyzed.
        rootpath : str
            Path to root directory where images / videos for the analysis are
            saved.

        Examples
        --------
        >>> man_analysis = ManualAnalysis(
            img_list=["C:/user/Dokuments/images/image1.tif",
            "C:/user/Dokuments/images/image2.tif",
            "C:/user/Dokuments/images/image3.tif"],
            rootpath="C:/user/Dokuments/images")
        """
        # Get input images
        self.image_list = img_list
        self.rootpath = rootpath

        # keep a reference to all lines by keeping them in a list
        self.lines = []

        # Keep all coordinates for each analysis
        self.scale_coords = []
        self.thick_coords = []
        self.fasc_coords = []
        self.pen_coords = []

        # Dictionary containing line coordinates
        self.coords = {"x": 0, "y": 0, "x2": 0, "y2": 0}

        # Define count for image selection
        self.count = 0

        # Define dataframe for export
        self.dataframe = pd.DataFrame(
            columns=["File", "Fasicle Length", "Pennation Angle", "Thickness"]
        )

    def calculateBatchManual(self):
        """Instance method creating a GUI for manual annotation of longitudinal
        ultrasoud images of human lower limb muscles.

        The GUI contains several analysis options for the current image
        openend. The user is able to switch between analysis of muscle
        thickness, fascicle length and pennation angle. Moreover, the
        image can be scaled and the aponeuroses can be extendet to easy
        fascicle extrapolation. When one image is finished, the GUI
        updated by clicking "next image". The Results can be saved by
        clicking "save results". It is possible to save each image
        seperately. The GUI can be closed by clicking "break analysis"
        or simply close the window.

        Notes
        -----
        The GUI can be initated from the main DL_Track GUI. It is also
        possible to initiate the ManualAnalis class from the command
        promt and interact with its GUI as a standalone. To do so,
        the lines 231 (self.head = tk.Tk()) and 314 (self.head.mainloop())
        must be uncommented and the line 233 (self.head = tk.Toplevel())
        must be commented.
        """
        # This line must be uncommented when the class is initiated from
        # the command prompt
        # self.head = tk.Tk()

        self.head = tk.Toplevel()
        self.head.title("DL_Track_US - Manual Analysis")
        master_path = os.path.dirname(os.path.abspath(__file__))
        iconpath = master_path + "/home_im.ico"
        #self.head.iconbitmap(iconpath)

        # Style
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#808080")
        style.configure(
            "TRadiobutton",
            background="#808080",
            foreground="black",
            font=("Lucida Sans", 12),
        )
        style.configure(
            "TButton",
            background="papaya whip",
            foreground="black",
            font=("Lucida Sans", 11),
        )

        # Create frame for Buttons
        toolbar = ttk.Frame(self.head)
        toolbar.pack(fill="x")

        # Create different analysis types
        self.mode = tk.StringVar(self.head, "thick")

        scaling = ttk.Radiobutton(
            toolbar,
            text="Scale Image",
            variable=self.mode,
            value="scale",
            command=self.setScale,
        )
        scaling.pack(side="left")

        scaling.pack(side="left")
        aponeurosis = ttk.Radiobutton(
            toolbar,
            text="Draw Aponeuroses",
            variable=self.mode,
            value="apo",
            command=self.setApo,
        )
        aponeurosis.pack(side="left")

        thickness = ttk.Radiobutton(
            toolbar,
            text="Muscle Thickness",
            variable=self.mode,
            value="thick",
            command=self.setThickness,
        )
        thickness.pack(side="left")

        fascicle = ttk.Radiobutton(
            toolbar,
            text="Muscle Fascicles",
            variable=self.mode,
            value="fasc",
            command=self.setFascicles,
        )
        fascicle.pack(side="left")

        pennation = ttk.Radiobutton(
            toolbar,
            text="Pennation Angle",
            variable=self.mode,
            value="pen",
            command=self.setAngles,
        )
        pennation.pack(side="left")

        # Define Button fot break
        stop = ttk.Button(
            toolbar, text="Break Analysis", command=self.stopAnalysis
        )
        stop.pack(side="right")

        # Define Button for next image
        next_img = ttk.Button(
            toolbar, text="Next Image", command=self.updateImage
        )
        next_img.pack(side="right")

        # Define Button for save image
        save = ttk.Button(
            toolbar, text="Save Results", command=self.saveResults
        )
        save.pack(side="right")

        # Define Canvas for image
        self.canvas = tk.Canvas(self.head, bg="black", height=1500, width=1500)
        self.canvas.pack(fill="both")

        # Add padding
        for child in toolbar.winfo_children():
            child.pack_configure(padx=5, pady=5)

        # Load image
        resized_img = Image.open(self.image_list[0]).resize((1024, 800))
        self.img = ImageTk.PhotoImage(
            resized_img,
            master=self.canvas,
        )
        my_image = self.canvas.create_image(
            750, 500, anchor="center", image=self.img
        )

        # Bind mouse events to canvas
        self.canvas.bind("<ButtonPress-1>", self.click)
        self.canvas.bind("<ButtonRelease-1>", self.release)
        self.canvas.bind("<B1-Motion>", self.drag)

        # This line must be uncommented when the class is initiated from
        # the command prompt
        # self.head.mainloop()

    # ----------------------------------------------------------------
    # Functionalities used in head

    def click(self, event: str):
        """Instance method to record mouse clicks on canvas.

        When the left mouse button is clicked on the canvas, the
        xy-coordinates are stored for further analysis. When the button
        is clicked multiple times, multiple values are stored.

        Parameters
        ----------
        event : str
            tk.Event variable containing the mouse event
            that is bound to the instance method.

        Examples
        --------
        >>> self.canvas.bind("<ButtonPress-1>", self.click)
        """
        # define start point for line
        self.coords["x"] = self.canvas.canvasx(event.x)
        self.coords["y"] = self.canvas.canvasy(event.y)

        # create a line on this point and store it in the list
        self.lines.append(
            self.canvas.create_line(
                self.coords["x"],
                self.coords["y"],
                self.coords["x"],
                self.coords["y"],
                fill="white",
                width=5,
                smooth=True,
            )
        )
        # determine list to append to.
        if self.mode.get() == "scale":
            self.scale_coords.append([self.coords["x"], self.coords["y"]])
        elif self.mode.get() == "thick":
            self.thick_coords.append([self.coords["x"], self.coords["y"]])
        elif self.mode.get() == "fasc":
            self.fasc_coords.append([self.coords["x"], self.coords["y"]])
        elif self.mode.get() == "pen":
            self.pen_coords.append([self.coords["x"], self.coords["y"]])

    def release(self, event: str):
        """Instance method to record mouse button releases on canvas.

        When the left mouse button is released on the canvas, the
        xy-coordinates are stored for further analysis. When the button
        is released multiple times, multiple values are stored.

        Parameters
        ----------
        event : str
            tk.Event variable containing the mouse event
            that is bound to the instance method.

        Examples
        --------
        >>> self.canvas.bind("<ButtonRelease-1>", self.click)
        """
        # get coordinates of end of line
        self.coords["x2"] = self.canvas.canvasx(event.x)
        self.coords["y2"] = self.canvas.canvasy(event.y)

        # determine list to append to.
        if self.mode.get() == "scale":
            self.scale_coords.append([self.coords["x2"], self.coords["y2"]])
        elif self.mode.get() == "thick":
            self.thick_coords.append([self.coords["x2"], self.coords["y2"]])
        elif self.mode.get() == "fasc":
            self.fasc_coords.append([self.coords["x2"], self.coords["y2"]])
        elif self.mode.get() == "pen":
            self.pen_coords.append([self.coords["x2"], self.coords["y2"]])

    def drag(self, event: str):
        """Instance method to record mouse cursor dragging on canvas.

        When the cursor is dragged on the canvas, the
        xy-coordinates are stored and updated for further analysis.
        This is used to draw the line that follows the cursor
        on the screen. Coordinates are only recorded when the left
        mouse button is pressed.

        Parameters
        ----------
        event : str
            tk.Event variable containing the mouse event
            that is bound to the instance method.

        Examples
        --------
        >>> self.canvas.bind("<B1-Motion>", self.click)
        """
        # update the coordinates from the event
        self.coords["x2"] = self.canvas.canvasx(event.x)
        self.coords["y2"] = self.canvas.canvasy(event.y)

        # Change the coordinates of the last created line to the new
        # coordinates
        self.canvas.coords(
            self.lines[-1],
            self.coords["x"],
            self.coords["y"],
            self.coords["x2"],
            self.coords["y2"],
        )

    def setScale(self):
        """Instance method to display the instructions for image scaling.

        This function is bound to the "Scale Image" radiobutton and
        appears each time the button is selected. In this way, the user
        is reminded on how to scale the image.
        """
        messagebox.showinfo(
            "Information",
            "Draw 10mm long scaling line."
            + "\nThis is only necessary in the first image.",
        )

    def setApo(self):
        """Instance method to display the instructions for aponeuroses
        extending.

        This function is bound to the "Draw Aponeurosis" radiobutton and
        appears each time the button is selected. In this way, the user
        is reminded on how to draw aponeurosis extension on the image.
        """
        messagebox.showinfo("Information", "Draw lines to extend aponeuroses.")

    def setThickness(self):
        """
        Instance method to display the instructions for muscle thickness
        analysis.

        This function is bound to the "Muscle Thickness" radiobutton and
        appears each time the button is selected. In this way, the user
        is reminded on how to analyze muscle thickness.
        """
        messagebox.showinfo(
            "Information",
            "Draw lines to measure muscle thickness."
            + "\nSelect at least three sites.",
        )

    def setFascicles(self):
        """Instance method to display the instructions for muscle fascicle
        analysis.

        This function is bound to the "Muscle Fascicles" radiobutton and
        appears each time the button is selected. In this way, the user
        is reminded on how to analyze muscle fascicles.
        """
        messagebox.showinfo(
            "Information",
            "Draw lines to measure muscle fascicles."
            + "\nSelect at least three fascicles."
            + "\nLines MUST consist of three segments.",
        )

    def setAngles(self):
        """Instance method to display the instructions for pennation angle
        analysis.

        This function is bound to the "Pennation Angle" radiobutton and
        appears each time the button is selected. In this way, the user
        is reminded on how to analyze pennation angle.
        """
        messagebox.showinfo(
            "Information",
            "Draw lines to measure pennation angles."
            + "\nSelect at least three angles."
            + "\nAngle MUST consist of two segments.",
        )

    def saveResults(self):
        """Instance Method to save the analysis results to a pd.DataFrame.

        A list of each variable to be saved must be used. The list must
        contain the coordinates of the recorded events during
        parameter analysis. Then, the respective parameters are calculated
        using the coordinates in each list and inculded in a dataframe.
        Estimates or fascicle length, pennation angle,
        muscle thickness are saved.

        Notes
        -----
        In order to correctly calculate the muscle parameters, the amount of
        coordinates must be specific. See class documentatio of documentation
        of functions used to calculate parameters.

        See Also
        --------
        self.calculateThickness, self.calculateFascicle,
        self.calculatePennation
        """
        # Get results
        thickness = self.calculateThickness(self.thick_coords)
        fascicles = self.calculateFascicles(self.fasc_coords)
        pennation = self.calculatePennation(self.pen_coords)

        # Check number of input images and counter
        if len(self.scale_coords) >= 1:
            self.dist = np.abs(
                self.scale_coords[0][1] - self.scale_coords[1][1]
            )

            # Save Results with scaling
            dataframe2 = pd.DataFrame(
                {
                    "File": self.image_list[self.count],
                    "Fasicle Length": np.mean(fascicles) / self.dist,
                    "Pennation Angle": np.mean(pennation),
                    "Thickness": np.mean(thickness) / self.dist,
                },
                index=[self.count],
            )

            self.dataframe = pd.concat([self.dataframe, dataframe2], axis=0)

        else:
            # Save Results with scaling
            dataframe2 = pd.DataFrame(
                {
                    "File": self.image_list[self.count],
                    "Fasicle Length": np.mean(fascicles),
                    "Pennation Angle": np.mean(pennation),
                    "Thickness": np.mean(thickness),
                },
                index=[self.count],
            )

            self.dataframe = pd.concat([self.dataframe, dataframe2], axis=0)

        # Save the drawing on the canvas to an image
        # Define coordinates for cropping
        x = (
            self.head.winfo_rootx() + self.head.winfo_x()
        )  # Top x of root + head
        y = (
            self.head.winfo_rooty() + self.head.winfo_y()
        )  # Top y of root + head
        x1 = x + 2 * self.head.winfo_width()  # include twice the width
        y1 = y + 2 * self.head.winfo_height()  # include twice the height
        # Save the screenshot to root location
        ImageGrab.grab().crop((x, y, x1, y1)).save(
            f"{self.image_list[self.count]}_analyzed.png"
        )

    def updateImage(self):
        """Instance method to update the current image displayed
        on the canvas in the GUI.

        The image is updated upon clicking of "Next Image"
        button in the GUI. Images can be updated until all
        images have been analyzed. Even when the image is updated,
        the analysis results are stored.
        """
        # clear canvas
        self.canvas.delete("all")

        # Empty lists
        self.lines = []
        self.thick_coords = []
        self.fasc_coords = []
        self.pen_coords = []

        # Reset Coordinates
        self.coords = {"x": 0, "y": 0, "x2": 0, "y2": 0}

        if self.count < len(self.image_list) - 1:
            self.count += 1

            # Count for images
            self.img = ImageTk.PhotoImage(
                Image.open(self.image_list[self.count]).resize((1024, 800))
            )
            my_image = self.canvas.create_image(
                750, 500, anchor="center", image=self.img
            )

        else:
            # Save Results to Excel
            self.compileSaveResults()
            messagebox.showerror("Error", "No more images available.")
            # Quit session
            self.closeWindow()

    def stopAnalysis(self):
        """Instance method to stop the current analysis
        on the canvas in the GUI.

        The analysis is stopped upon clicking of "Break Analysis"
        button in the GUI. The current analysis results are saved
        when the analysis is terminated. The analysis can be
        terminated at any timepoint during manual annotation. Prior to
        terminating, the user is asked for confirmation.
        """
        # Save results
        self.compileSaveResults()
        # Terminate analysis
        messagebox.askyesno(
            "Attention", "Would you like to stop the Analysis?"
        )
        self.closeWindow()

    # --------------------------------------------------------------------------------------------------------------------
    # Utilities used by functionalities in head

    def getAngle(self, a: list, b: list, c: list) -> float:
        """Instance method to calculate angle between three points.

        The angle is calculated using the arc tangent. The arc tangent is used
        to find the slope in radians when Y and X co-ordinates are given. The
        output is the arc tangent of y/x in radians, which is between PI and
        -PI. Then the output is converted to degrees.

        Parameters
        ----------
        a : list
            List variable containing the xy-coordinates of the first mouse
            click event of the pennation angle annotation. This must be the
            point at the beginning of the fascicle segment.
        b : list
            List variable containing the xy-coordinates of the second mouse
            click event of the pennation angle annotation. This must be the
            point at the intersection between fascicle and aponeurosis.
        c : list
            List variable containing the xy-coordinates of the fourth
            mouse event of the pennation angle annotation. This must be
            the endpoint of the aponeurosis segment.

        Returns
        -------
        ang : float
            Float variable containing the pennation angle in degrees between
            the segments drawn on the canvas by the user. Only the angle
            smaller than 180 degrees is returned.

        Example
        -------
        >>> getAngle(a=[904.0, 41.0], b=[450,380], c=[670,385])
        25.6
        """
        ang = math.degrees(
            math.atan2(c[1] - b[1], c[0] - b[0])
            - math.atan2(a[1] - b[1], a[0] - b[0])
        )

        if ang < 0:
            ang = ang * (-1)

        return ang if ang < 180 else 360 - ang

    def calculateThickness(self, thick_list: list) -> list:
        """Instance method to calculate distance between deep and superficial
        muscle aponeuroses, also known as muscle thickness.

        The length of the segment is calculated by determining the absolute
        distance of the y-coordinates of two points.
        Here, the muscle thickness is calculated considering only the
        y-coordinates of the start and end points of the drawn segments.
        In this way, incorrect placement of the segments by drawing skewed
        lines can be accounted for. Then the thickness is outputted in
        pixel units.

        Parameters
        ----------
        thick_list : list
            List variable containing the xy-coordinates of the first mouse
            click event and the mouse release event of the muscle thickness
            annotation. This must be the points at the beginning and end of
            the thickness segment.

        Returns
        -------
        thickness : list
            List variable containing the muscle thickness in pixel units.
            This value is calculated using only the
            difference of the y-coordinates of the two points. If multiple
            segments are drawn, multiple thickness values are outputted.

        Example
        -------
        >>> calculateThickness(thick_list=[[450, 41.0], [459, 200]])
        [159]
        """
        # Keep track of thickness values
        thickness = []
        # Keep track of segments
        count = 0

        # Iterate through thick_list to calculate thickness but part
        # by 2 because 2 points belong to on muscle thickness measurement
        for point in range(0, int(len(thick_list) / 2)):

            dist = np.abs(thick_list[count][1] - thick_list[count + 1][1])
            thickness.append(dist)
            count += 2

        return thickness

    def calculateFascicles(self, fasc_list: list) -> list:
        """Instance method to calculate muscle fascicle legth as a sum of three
        annotated segments.

        The length of three line segment is calculated by summing their
        individual length. Here, the length of a single annotated fascicle
        is calculated considering the three drawn segments of the respective
        fascicle. The euclidean distance between the start and endpoint of
        each segment is calculated and summed. Then the length of the
        fascicle is outputted in pixel units.

        Parameters
        ----------
        fasc_list : list
            List variable containing the xy-coordinates of the first mouse
            click event and the mouse release event of each annotated fascicle
            segment.

        Returns
        -------
        fascicles : list
            List variable containing the fascicle length in pixel units.
            This value is calculated by summing the euclidean distance of
            each drawn fascicle segment. If multiple fascicles are drawn,
            multiple fascicle length values are outputted.

        Example
        -------
        >>> calculateFascicles(fasc_list=[[392.0, 622.0], [632.0, 544.0],
                                          [632.0, 544.0],
                                          [1090.0, 415.0], [1090.0, 415.0],
                                          [1274.0, 381.0],
                                          [449.0, 627.0], [748.0, 541.0],
                                          [748.0, 541.0],
                                          [1109.0, 429.0], [1109.0, 429.0],
                                          [1297.0, 390.0]])
        [915.2921723246823, 881.0996335404545]
        """
        # turn list to array
        fasc_list = np.asarray(fasc_list)
        # Keep track of fascicle lengths
        fascicles = []
        # Keep count of segments
        count = 0

        # loop through points in fasc_list but part the length
        # by 6 because 6 points belong to one fascicle
        for _ in range(0, int(len(fasc_list) / 6)):

            # Iterate through segments to calculate and draw one fascicle
            # and iterate three times because a fascicle consists of three
            # segments
            fascicle = []  # Segment length are stored here
            for _ in range(3):

                # calculate the distance between two points belonging to one
                # segment
                dist = distance.euclidean(
                    fasc_list[count], fasc_list[count + 1]
                )
                fascicle.append(dist)
                # increase count to jump to next segment
                count += 2

            # When looped through all segments append length of whole fascicle
            fascicles.append(sum(fascicle))

        return fascicles

    def calculatePennation(self, pen_list: list) -> list:
        """Instance method to calculate muscle pennation angle between three
        points.

        The angle between three points is calculated using the arc tangent.
        Here, the points used for calculation are the start and endpoint of
        the line segment drawn alongside the fascicle as well as the endpoint
        of the segment drawn along the aponeurosis. The pennation angle
        is outputted in degrees.

        Parameters
        ----------
        pen_list : list
            List variable containing the xy-coordinates of the first mouse
            click event and the mouse release event of each annotated
            pennation angle segment.

        Returns
        -------
        pen_angles : list
            List variable containing the pennation angle in degrees.
            If multiple pennation angles are drawn, multiple angle values
            are outputted.

        See Also
        --------
        self.getAngle()

        Example
        -------
        >>> calculateFascicles(pen_list=[[760.0, 579.0], [620.0, 629.0],
                                         [620.0, 629.0],
                                         [780.0, 631.0], [533.0, 571.0],
                                         [378.0, 627.0], [378.0, 627.0],
                                         [558.0, 631.0]] )
        [20.369984003523715, 21.137327423466722]
        """
        # Keep track of angles
        pen_angles = []
        # Keep count of segments
        count = 0

        # interate through pen_list but part the length by
        # 4 because 4 points belong to the same angle
        for _ in range(0, int(len(pen_list) / 4)):

            # Iterate through points to calculate and draw one fascicle
            angle = self.getAngle(
                pen_list[count], pen_list[count + 1], pen_list[count + 3]
            )

            pen_angles.append(angle)
            # Increase count to jump to next angle
            count += 4

        return pen_angles

    def compileSaveResults(self):
        """Instance method to save the analysis results.

        A pd.DataFrame object must be used. The results
        inculded in the dataframe are saved to an .xlsx file. Depending
        on the form of class initialization, the .xlsx file is
        either saved to self.root (GUI) or self.out (command prompt).

        Notes
        -----
        An .xlsx file is saved to a specified location
        containing all analysis results.
        """
        excelpath = self.rootpath + "/Manual_Results.xlsx"
        if os.path.exists(excelpath):
            with pd.ExcelWriter(
                excelpath, mode="a", if_sheet_exists="replace"
            ) as writer:
                data = self.dataframe
                data.to_excel(writer, sheet_name="Results")
        else:
            with pd.ExcelWriter(excelpath, mode="w") as writer:
                data = self.dataframe
                data.to_excel(writer, sheet_name="Results")

    def closeWindow(self):
        """Instance method to close window upon call.

        This method is evoked by clicking the button "Break Analysis".
        It is necessary to use a different function, otherwise
        the window would be destroyed upon starting.
        """
        self.head.destroy()


# When using standalone uncomment To do so, uncomment the lines 231
# (self.head = tk.Tk()) and 314 (self.head.mainloop())
# and comment the line 233 (self.head = tk.Toplevel()).
# The class can then be initalized by typing
# "python manual_tracing.py" and specifying the filetype and rootpath
# in the code below. The whole codeblock below must be uncommented

# if __name__ == "__main__":

#     filetype = "/**/*.avi"
#     rootpath = "C:/Users/admin/Desktop"
#     img_list = glob.glob(rootpath + filetype, recursive=True)
#     analysis = ManualAnalysis(img_list, rootpath)
#     analysis.calculateBatchManual()
