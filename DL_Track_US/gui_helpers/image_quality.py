"""
Description
-----------
This module contains a class to augment image brightness, contrast and noise.
Moreover, using the class, the quality of images can be evaluated. The quality
score is calculated based on mean pixel intensity, standard deviation or pixel
intensity, noise estimation, blur estimation and signal-to-noise ratio.
These parameters were selected because we believe that they are most
relevant for the evaluation of ultrasound images.
The module was specifically designed to be executed from a GUI, but each
function can be used separately.
When used from the GUI, only the image evaluation funtions are used. A .xlsx
file containing the parameters will be saved to a specified directory. This
is the directory where all images to be evaluated should be.

Functions scope
---------------
For scope of the functions see class documentation.

Notes
-----
Additional information and usage examples can be found at the respective
functions docstrings.
"""

import glob
import os
import time
import cv2

import numpy as np
import pandas as pd
from scipy import stats
from sewar.full_ref import ssim
from skimage import restoration, util


class ImageQuali:
    """
    Python class to adapt and assess the quality of images.
    The functionalities offered by this class can adapt the
    brightness, contrast and noise of images. Image quality can
    also be assessed. All instance methods can be used
    separately but the main instance is 'adaptImages' where images can be
    augmented and their quality assessed.

    Attributes
    ----------
    self.root : str
        Path to root directory containing the images to be augmented/
        analyzed.
    self.out : str
        Path to output directory where augmented images are saved.
        Note that the quality assessment .xlsx file is always saved
        to the root path.
    self.mlocs : list
        List variable to store coordinates of mouse clicks. This is
        only used when images are cropped.

    Methods
    -------
    __init__
        Instance method to initialize class
    mclick
         Instance method to detect mouse click coorindates in image.
    getBoundingBox
        Instance method to get the user selected bounding box and
        crop the image
    saveResults
        Instance method to save image quality assessment results to
        excel sheet
    getSignalToNoise
        Instance method to calculate the signel to noise ratio of a given image.
    calculateNoise
        Instance method to assess image quality/noise in several measures
    makeImageContrast
        Instance method to change image contrast/brightness
    makeImageNoise
        Instance method to add speckle noise to image
    makeImageBlur
        Instance method to blur the input image
    adaptImages
        Instance method to apply quality changes and assess qualiy

    Notes
    -----
    This class solely consists of instance methods.
    Each method can be used on its own with the respective input parameters.
    For more information on the instance methods check the respective
    docstrings.

    Some of the instance methods are not used when the class is initialized.
    These include the following ones: mclick, getBoundingBox, makeImageContrast,
    makeImageBlur, makeImageNoise.
    Moreover, the self.out instance attribute is initialized to be the same
    directory as the self.input attribute.
    """

    def __init__(self, rootpath: str, outpath: str):
        """
        Instance method to initialize the ImageQuali class.

        Parameters
        ----------
        rootpath : str
            Path to root directory containing images to be
            augmented/analyzed.
        outpath : str
            Path to output directory where images/results
            are saved.
        mlocs : list
            List variable to store detected mouse click
            coordinates for bounding box selection.
        """
        self.root = rootpath
        self.out = outpath
        self.mlocs = []

    def mclick(self, event, x_val, y_val, flags, param):
        """
        Instance method to detect mouse click coorindates in image.

        This instance is used when the image to be analyzed should be
        cropped.  Upon klicking of the mouse button, the coordinates
        of the cursor position are stored in the instance attribute
        self.mlocs.

        Parameters
        ----------
        event
            Event flag specified as Cv2 mouse event left mousebutton down.
        x_val
            Value of x-coordinate of mouse event to be recorded
        y_val
            Value of y-coordinate of mouse event to be recorded
        flags
            Specific condition whenever a mouse event occurs. This
            is not used here but needs to be specified as input
            parameter.
        param
            User input data. This is not required here but needs to
            be specified as input parameter.
        """
        # if the left mouse button was clicked, record the (x, y) coordinates
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mlocs.append(y_val)
            self.mlocs.append(x_val)

    def getBoundingBox(self, img: np.ndarray) -> np.ndarray:
        """
        Instance method to get the selected bounding box and
        crop the image.

        The image will be cropped based on the coordinates stored
        in self.mlocs and the bounding box will always be a
        rectangle. See Notes for information on exact placement of mouse clicks.

        Parameters
        ----------
        img : np.ndarray
            Grayscale image to be analysed as a numpy array. The image must
            be loaded prior to bounding box selection, specifying a path
            is not valid.

        Returns
        -------
        crop_img : np.ndarray
            The cropped grayscale image based on the bounding box as a numpy array.
            The size of the image is not adapted and is similar to the size
            of the selected bounding box.

        Notes
        -----
        To function properly, the user must select exactly four point
        on the image around which the bounding box built:
        - The first point must be the top left
        - The second point top right
        - The third point bottom left
        - The fourth bottom right
        The cropped image might look weird when other orders are used.

        Examples
        --------
        >>> getBoundingBox(img=([[[[0.22414216 0.19730392 0.22414216] ... [0.2509804  0.2509804  0.2509804 ]]]))
        ([[[[0.22414216 0.19730392 0.22414216] ... [0.2509804  0.2509804  0.2509804 ]]])
        """
        # display the image and wait for a keypress
        cv2.imshow("image", img)
        cv2.setMouseCallback("image", self.mclick)
        key = cv2.waitKey(0)

        # if the 'q' key is pressed, break from the loop
        if key == ord("q"):
            cv2.destroyAllWindows()

        # Crop image according to selection
        crop_img = img[
            self.mlocs[1] : self.mlocs[1] + self.mlocs[4],
            self.mlocs[0] : self.mlocs[0] + self.mlocs[3],
        ]

        # Reset coordinates
        self.mlocs = []

        return crop_img

    def saveResults(self, path: str, dataframe: pd.DataFrame) -> None:
        """
        Instance method to save the quality assessment results.

        A pd.DataFrame object must be inputted. The results
        inculded in the dataframe are saved to an .xlsx file. Depending
        on the form of class initialization, the .xlsx file is
        either saved to self.root (GUI) or self.out (command prompt).

        Parameters
        ----------
        path : str
            String variable containing the path to where the .xlsx file
            should be saved.
        dataframe : pd.DataFrame
            Pandas dataframe variable containing the image analysis results
            for every image anlyzed.

        Notes
        -----
        An .xlsx file is saved to a specified location
        containing all analysis results.

        Examples
        --------
        >>> saveResults(img_path = "C:/Users/admin/Dokuments/images",
                        dataframe = [['File',"image1"],['mean',12],
                                     ['Sigma',0.56]])
        """
        # Define path
        excelpath = path + "/Quality.xlsx"

        # Check if directory is existent
        if os.path.exists(excelpath):
            with pd.ExcelWriter(excelpath, mode="a") as writer:
                data = dataframe
                data.to_excel(writer, sheet_name="Results")
        else:
            with pd.ExcelWriter(excelpath, mode="w") as writer:
                data = dataframe
                data.to_excel(writer, sheet_name="Results")

    def getSignalToNoise(self, img: np.ndarray, axis: int = None, ddof: int = 0) -> float:
        """
        Instance method to calculate the signal to noise ratio (SNR) of a given image.

        Here we calculate the SNR using the mean of the image brightness and the
        standard deviation.

        Parameters
        ----------
        img : np.ndarray
            Grayscale image to be analysed as an numpy array. The image must
            be loaded prior to bounding box selection, specifying a path
            is not valid.
        axis : int, default = None
            Axis or axes along which the standard deviation is computed.
            The default is to compute the standard deviation of the flattened array.
        ddof : int, default = 0
            Means Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
            where N represents the number of elements. By default ddof is zero.

        Returns
        -------
        sig_noi_rat : float
            Float variabel containing the SNR for the input image.

        Examples
        --------
        >>> getSignalToNoise(img=([[[[0.22414216 0.19730392 0.22414216] ... [0.2509804  0.2509804  0.2509804 ]]]))
        0.317
        """
        img = np.asanyarray(img)
        # Calculate image mean
        me = img.mean(axis)
        # Caclulate img sd
        sd = img.std(axis=axis, ddof=ddof)
        # Caclulate SNR
        sig_noi_rat = np.where(sd == 0, 0, me / sd)

        return sig_noi_rat

    def getSSIM(self, comp_imgs_path: str, filetype: str = "/**/*.tif") -> None:
        """
        Instance method to calculate the structural similarity index (SSIM)
        between two images.

        The method requires two directories as input. Both directories are required
        to contain images of a given filetype that are compared. Each image in one
        folder is compared to each image in the other folder. The results are saved
        in the root directory that was specifed upon class initialization.

        Parameters
        ----------
        comp_ims_path : str
            String variable containing the path to where one set of images should be
            located. The other set of images provided for the comparison should be
            located in root.
        filetype : str, default = "/**/*.tif"
            String variable containg the respective type of the images.
            Both image sets must be of the same type.

        Examples
        --------
         >>> getSSIM(comp_img_path = "C:/Users/admin/Dokuments/images")
        """
        start = time.time()
        # Get all image files to be adapted
        list_of_files_comp = glob.glob(comp_imgs_path + filetype, recursive=True)
        list_of_files = glob.glob(self.root + filetype, recursive=True)

        # Create DF for export
        dataframe = pd.DataFrame(columns=["File", "SSIM"])

        # Loop through all images to calculate ssim
        for comp in list_of_files_comp:

            img_comp = cv2.imread(comp, 0)
            img_comp = cv2.resize(img_comp, (512, 512))
            filename_comp = os.path.splitext(os.path.basename(comp))[0]

            # Crop image to relevant region
            img_comp = self.getBoundingBox(img=img_comp)
            img_comp = cv2.resize(img_comp, (512, 512))

            # Loop trough all images to be compared to calculate ssim for each
            for image in list_of_files:

                # load and resize image
                img = cv2.imread(image, 0)
                filename = os.path.splitext(os.path.basename(image))[0]

                # Crop image to relevant region
                img = self.getBoundingBox(img=img)
                img = cv2.resize(img, (512, 512))

                # Calculate ssim
                ssim_score, _ = ssim(img, img_comp)

                dataframe = dataframe.append(
                    {"File": filename + "_vs_" + filename_comp, "SSIM": ssim_score},
                    ignore_index=True,
                )

                print(f"Image {filename} compared to Image {filename_comp}")

        # Calculate mean
        dataframe = dataframe.append(
            {"MeanSSIM": np.mean(dataframe["SSIM"])}, ignore_index=True
        )
        # Define excelpath

        self.saveResults(path=self.root, dataframe=dataframe)

    def calculateNoise(self, img: np.ndarray, crop: int = 0) -> float:
        """
        Instance method to calculate image quality/comparability parameters.

        The quality parameters we use are mean pixel values (mittel),
        standard deviation of pixel values (std_dev), an image noise estimate (sigma),
        an image blur estimate (blur). We decided to use these parameters
        because we deemed them to be most valid and relevant for ultrasound images.

        Parameters
        ----------
        img : np.ndarray
            Grayscale image to be analysed as an numpy array. The image must
            be loaded prior to bounding box selection, specifying a path
            is not valid.
        crop : int, default = 0
            Integer variable determining if image is cropped.
            If crop == 1, user must select bounding box. Must be either 1 or 0.

        Returns
        -------
        mittel : float
            Float value containing average of pixel values. This values lies
            somewhere between 0 and 255. It is calculated either on all image
            pixels or the pixels in the bounding box if crop == 1.
        std_dev : float
            Float value containing average of pixel values. This value
            can be understood as a proxy for the dynamic range, as the "variablity"
            of pixel values is captured here. It is calculated either on all image
            pixels or the pixels in the bounding box if crop == 1.
        sigma : float
            Float value containing the noise estimation in the input image. This
            value is calculated using the skimage.restoration.estimate_sigma
            function. It is calculated either on all image pixels or the pixels
            in the bounding box if crop == 1.
        blur : float
            Float value containing the estimate of image blur. The blur estimate
            is calculated using a Laplacian filter (usually used for edge filtering)
            and the resulting variance of the filter output. It is calculated either
            on all image pixels or the pixels in the bounding box if crop == 1.

        Examples
        --------
        >>> calculateNoise(img=([[[[0.22414216 0.19730392 0.22414216] ... [0.2509804  0.2509804  0.2509804 ]]]))
        20, 55.686, 3.379, 0.361, 3834.921418
        """
        # Check if cropping is specified
        if crop == 1:
            # crop image according to selection
            img = self.getBoundingBox(img=img)

        # Calculate SNR
        sig_noi_rat = round(float(self.getSignalToNoise(img=img)), 3)

        # Estimat image noise using Laplacian filter
        # The higher the blur value the better
        blur = cv2.Laplacian(img, cv2.CV_64F).var()

        # flatten image
        pixels = img.ravel()

        # Calculate spacial parameters
        mittel = round(np.mean(pixels))
        std_dev = round(pixels.std(), 3)

        # estimate Sigma as representative for noise
        sigma = round(restoration.estimate_sigma(img), 3)

        return mittel, std_dev, sigma, sig_noi_rat, blur

    def makeImageContrast(
        self, img: np.ndarray, brightness: int, contrast: int
    ) -> np.ndarray:
        """
        Instance method to change image contrast and/or brightness.

        Both, the image brightness and the image contrast can be augmented either
        seperately or simultaneously. Both values are adapted using cv2.addWeighted
        which basically performs a linear transform of pixel values using
        the input parameters brightness and contrast.

        Parameters
        ----------
        img : np.ndarray
            Grayscale image to be analysed as an numpy array. The image must
            be loaded prior to augmentation, specifying a path
            is not valid.
        brightness : int
            Integer value by which the brightness should be increased/decreased.
            Value can be negative, zero or positive.
        contrast : int
            Integer value by which the contrast should be increased/decreased.
            Value can be negative, zero or positive.

        Returns
        -------
        bc_img : np.ndarray
            The augmented grayscale image based on contrast/brightness value as a numpy array.
            The image can be binary if the contrast is increased by a high enough value.
        """
        # Adapt brightness
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow) / 255
            gamma_b = shadow

            # add weighted is basically y = a*(img) + b for brightness and contrast of pixels
            bc_img = cv2.addWeighted(img, alpha_b, img, 0, gamma_b)
        else:
            bc_img = img.copy()

        # Adapt contrast
        if contrast != 0:
            f = 131 * (contrast + 127) / (127 * (131 - contrast))
            alpha_c = f
            gamma_c = 127 * (1 - f)

            bc_img = cv2.addWeighted(bc_img, alpha_c, bc_img, 0, gamma_c)

        return bc_img

    def makeImageNoise(
        self, img: np.ndarray, var: float, mean: float, noise: str = "speckle"
    ) -> np.ndarray:
        """
        Instance method to add noise to an image.

        The noise that is added is either speckle noise or gaussian
        noise based on user specefication. Based on specification the
        extend of the added noise can be altered via mean and variance
        specification. The noise is added using the skimage.util.random_noise
        function.

        Parameters
        ----------
        img : np.ndarray
            Grayscale image to be analysed as an numpy array. The image must
            be loaded prior to augmentation, specifying a path
            is not valid.
        var : int
            Float value defining the variance of the noise distribution.
            The larger the variance, the more intense the noise.
        mean : int
            Float value defining the mean of the noise distribution.
            This shifts the whole noise appearance.
        noise : str, default = speckle
            String value containing specification what noiseis to be applied.
            There are two options of noise:
            - noise = "speckle"
            - noise = "gaussian"
            An error will be raised when another option is specified.

        Returns
        -------
        n_img : np.ndarray
            The augmented grayscale image based on noise specification as a numpy array.
        """
        # Check if image needs to be adapted
        if var == 0 & mean == 0:
            n_img = img.copy()

        else:
            # Check which noise type should be applied
            if noise == "speckle":
                # Apply specle noise with defined varaince & mean
                n_img = util.random_noise(img, mode="speckle", mean=mean, var=var)
                # The above returns floating point on rang [0,1] so it need to be converted
                n_img = np.array(255 * n_img, dtype="uint8")
            else:
                # Apply gaussian noise with defined varaince & mean
                n_img = util.random_noise(img, mode="gaussian", mean=mean, var=var)
                # The above returns floating point on rang [0,1] so it need to be converted
                n_img = np.array(255 * n_img, dtype="uint8")

        return n_img

    def makeImageBlur(self, img: np.ndarray, sigma: int) -> np.ndarray:
        """
        Instance method to blur the input image.

        The blur applied to the image is gaussian blur. To apply the blur,
        the cv2.GuassianBlur function is used. If the sigma values equals 0,
        then the input image is returned.

        Parameters
        ----------
        img : np.ndarray
            Grayscale image to be analysed as an numpy array. The image must
            be loaded prior to augmentation, specifying a path
            is not valid.
        sigma : int
            Integer value defining the sigma used for the gaussian kernel
            during blurring operation. The higher the sigma, the higher
            the applied blur. When sigma = 0, the original value is returned.

        Returns
        -------
        b_img : np.ndarray
            The augmented grayscale image based on blur specification as a numpy array.
        """
        if sigma == 0:
            b_img = img.copy()

        else:
            # Apply gaussian blur
            b_img = cv2.GaussianBlur(img, (11, 11), sigmaX=sigma, sigmaY=sigma)

        return b_img

    def adaptImages(
        self,
        brightness: int,
        contrast: int,
        mean: int,
        variance: int,
        save_img: str,
        filetype: str,
        sigma_b: int = 0,
    ):
        """
        Instance method to adapt brightness, contrast and noise of an image. Also,
        the quality of an image is assessed.

        The created images and the results are saved to the specified
        output directory. When all image augmentation parameters are set to 0,
        the images in the root directory are not augmented rather their quality
        is evaluated. Therefore, when only the image quality should be evaluated,
        all augmentation parameters should be set to 0.

        Parameters
        ----------
        brightness : int
            Integer value by which the brightness should be increased/decreased.
            Value can be negative, zero or positive.
        contrast : int
            Integer value by which the contrast should be increased/decreased.
            Value can be negative, zero or positive.
        var : int
            Float value defining the variance of the noise distribution.
            The larger the variance, the more intense the noise.
        mean : int
            Float value defining the mean of the noise distribution.
            This shifts the whole noise appearance.
        save_img : str
            String value defining if the adapted should be saved.
            There are two options for save_img:
            - save_img = "Yes". Images will be saved.
            - save_img = "No". Images will not be saved.
            An error will be raised when something other is specified.
        sigma_b : int, default = 0
            Integer value defining the sigma used for the gaussian kernel
            during blurring operation. The higher the sigma, the higher
            the applied blur. When sigma = 0, the original value is returned.

        See Also
        --------
        makeImageContrast, makeImageNoise, calculateNoise, makeImageBlur,
        saveResults function.

        Examples
        --------
        >>> adaptImages(brightness=0, contrast=0, mean=0, variance=0,
                            filetype=/**/*.tif, sigma_b=0, save_img="No")
        """
        # Get all image files to be adapted
        list_of_files = glob.glob(self.root + filetype, recursive=True)

        # Create DF for export
        dataframe = pd.DataFrame(
            columns=["File", "Mean", "Standard Dev", "Sigma", "SNR", "Blur", "ImScore"]
        )

        # Test for image files in directory
        if len(list_of_files) > 1:

            # Iterate through images
            for image in list_of_files:

                # load image
                img = cv2.imread(image, 0)
                img = cv2.resize(img, (512, 512))
                filename = os.path.splitext(os.path.basename(image))[0]

                # Adapt image brightness and contrast
                adapted_img = self.makeImageContrast(
                    img=img, brightness=brightness, contrast=contrast
                )
                # Adapt image noise
                adapted_img = self.makeImageNoise(
                    img=img, var=variance, mean=mean, noise="gaussian"
                )

                # Adapt image blur
                adapted_img = self.makeImageBlur(img=img, sigma=sigma_b)

                # Calculate Noise parameters
                mittel, std_dev, sigma, sig_noi_rat, blur = self.calculateNoise(
                    img=adapted_img
                )

                # Take weighted geometric mean
                params = np.array([mittel, std_dev, sigma, sig_noi_rat, blur])
                geo_m = stats.gmean(params)

                # Append to dataframe
                dataframe = dataframe.append(
                    {
                        "File": filename,
                        "Mean": mittel,
                        "Standard Dev": std_dev,
                        "Sigma": sigma,
                        "SNR": sig_noi_rat,
                        "Blur": blur,
                        "ImScore": geo_m,
                    },
                    ignore_index=True,
                )

                if save_img == "Yes":
                    cv2.imwrite(
                        self.out + "/" + filename + "adapt" + ".tif", adapted_img
                    )

            # Save Results
            self.saveResults(path=self.out, dataframe=dataframe)

        else:
            raise NameError("No files found in directory")


# ---------------------------------------------------------------------------------------------------
# Usage from command prompt
# Specify the values according to your needs and uncomment the codeblock below.
# To run type <<< python image_quality.py >>>

# if __name__ == '__main__':

#     brightness = 0
#     contrast = 0
#     mean = 0
#     variance = 0
#     sigma = 0
#     root = "C:/Users/admin/Documents/images"
#     out = "C:/Users/admin/Documents"

#     imqual = ImageQuali(rootpath=root, outpath=out)
#     imqual.adaptImages(brightness=brightness, contrast=contrast, mean=mean, variance=variance,
#                        sigma_b=sigma, save_img="No")
