import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from do_calculations import contourEdge, sortContours
from scipy.signal import savgol_filter
from skimage.morphology import skeletonize

pred_apo = np.load(
    r"C:\Users\carla\Documents\Master_Thesis\Example_Images\Paul_35_images\aponeurosis_masks\im_35_re.npy"
)
pred_apo_t = (pred_apo > 0).astype(np.uint8)
image_gray = pred_apo_t * 255

save_dir = r"C:\Users\carla\Documents\Master_Thesis\Example_Images\Paul_35_images"
os.makedirs(save_dir, exist_ok=True)
file_path = os.path.join(save_dir, "pred_apo.npy")
np.save(file_path, image_gray)

pred_fasc = np.load(
    r"C:\Users\carla\Documents\Master_Thesis\Example_Images\Paul_35_images\fascicle_masks\im_35_re.npy"
)
pred_fasc_t = (pred_fasc > 0).astype(np.uint8)
image_gray = pred_fasc_t * 255

file_path = os.path.join(save_dir, "pred_fasc.npy")
np.save(file_path, image_gray)
