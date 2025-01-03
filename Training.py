print("Training in Progress........")
import os
import numpy as np
from scipy.io import wavfile
import nibabel as nib
import cv2
from skimage import img_as_ubyte
from PIL import Image as im
import time


import warnings
warnings.filterwarnings("ignore")

def convert_to_2d(nifti_img, time_point=0, slice_index=None):
    # Extract the data array from the NIfTI image object
    data = nifti_img.get_fdata()
    # Get the dimensions of the 4D data array (x, y, z, time)
    nx, ny, nz, nt = data.shape
    # Default to the middle slice along the z-axis if slice_index is not provided
    if slice_index is None:
        slice_index = nz // 2
    # Select the 2D slice for the given time point and slice index
    slice_2d = data[:, :, slice_index, time_point]
    return slice_2d

def main():
    data = "..//Dataset"
    list_d = os.listdir(data)
    dataset = "..//Dataset//data//"
    listdata = os.listdir(dataset)
    data = []
    label = []
    time.sleep(10)
    for i in listdata[0:1]:
        d = dataset + listdata[0]
        d1 = dataset + listdata[1]
        listd = os.listdir(d)
        for j in listd:
            if "augments.npy" not in list_d:
                f = d + "\\" + j
                f = d + "\\" + j
                f = f.split("\\")
                f = f[-1];
                f = f.split(".");
                f = f[0][:2]
                f1 = d1 + "\\" + j
                img = nib.load(f1)
                slice_2d = convert_to_2d(img, time_point=0, slice_index=None)
                data = im.fromarray(slice_2d.T)
                image = np.array(data)
                # Add synthetic speckle noise to the image
                # Convert image to grayscale (if it's not already)
                noise_std = 20.0
                noisy_image = image + np.random.normal(0, noise_std, image.shape)
                # Apply Frost filtering
                window_size = 2
                filtered_image = cv2.medianBlur(image, 5)
    label = np.asarray(["0", "1"])
    if "augments.npy" not in list_d:
        from Code import Proposed
        from Code import CNet_GAN
        from Code import GAN
        from Code import SA_GAN
        from Code import VAE
        from Code import CyclicGAN
        from Code import StyleGAN

if __name__ == "__main__":
    main()
print("Training has been done successfully")



