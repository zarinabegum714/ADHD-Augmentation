import tkinter
from tkinter import *
from tkinter import Tk, messagebox
from tkinter import filedialog
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import ImageTk, Image
import numpy as np
from scipy.ndimage import correlate
from skimage import img_as_ubyte
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img
import warnings
warnings.filterwarnings("ignore")
from Code import Training


win= Tk()
win.geometry("1150x600+0+0")
win.title("BRAIN fMRI AUGMENTATION")

def disable_button():
   win.destroy()

def browse():
    global file_name
    path = filedialog.askopenfilename(
        title="Select a file of any type",
        filetypes=[("All files", ".jpg")]
    )
    file_name = path.split("/")
    file_name = file_name[-1].split(".")

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

    nifti_file = path
    # Load the NIfTI file
    # img = nib.load(nifti_file)
    # Convert 4D to 2D for visualization (selecting the middle slice and first time point)
    # slice_2d = convert_to_2d(img, time_point=0, slice_index=None)
    # Display the 2D slice
    # plt.imshow(slice_2d.T, cmap='gray', origin='lower')
    # plt.savefig("..//Img_res//Ip_img//"+file_name[0]+".jpg")
    # plt.close()
    # img = cv2.imread("..//Img_res//Ip_img//"+file_name[0]+".jpg")
    img = cv2.imread(path)
    im = Image.fromarray(img)
    im = im.resize((150, 150), Image.LANCZOS)
    im.save("..//Img_res//Ip_img//"+file_name[0]+".jpg")
    imgtk = ImageTk.PhotoImage(image=im)
    label = tkinter.Label(frame2, image=imgtk)
    label.image = imgtk
    label.place(x=30, y=20)
    messagebox.showinfo("INFO", "Image has been browsed successfully")
    button1.configure(state="disabled")

def denoising():
    def frost_filter(image, window_size, noise_std):
        # Convert image to float
        image = image.astype(np.float64)
        # Calculate local means using a rectangular window
        mean_I = correlate(image, np.ones((window_size, window_size))) / (window_size ** 2)
        # Calculate local variances
        mean_I2 = correlate(image ** 2, np.ones((window_size, window_size))) / (window_size ** 2)
        variance_I = mean_I2 - mean_I ** 2
        # Estimate noise variance
        noise_variance = noise_std ** 2
        # Compute adaptive weights
        alpha = variance_I / (variance_I + noise_variance)
        # Ensure alpha is within [0, 1]
        alpha = np.clip(alpha, 0, 1)
        # Compute filtered image
        filtered_image = mean_I + alpha * (image - mean_I)
        # Clip to ensure values are within valid range (0-255 for uint8)
        filtered_image = np.clip(filtered_image, 0, 255)
        return filtered_image.astype(np.uint8)
    # Load a sample image (you can use your own image as well)
    image = cv2.imread("..//Img_res//Ip_img//"+file_name[0]+".jpg")
    image = img_as_ubyte(image)  # Convert to uint8 if not already
    # Add synthetic speckle noise to the image
    # Convert image to grayscale (if it's not already)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    noise_std = 20.0
    noisy_image = image + np.random.normal(0, noise_std, image.shape)
    # Apply Frost filtering
    window_size = 2
    filtered_image = cv2.medianBlur(gray_image, 5)
    plt.imshow(filtered_image, cmap='gray')
    plt.axis('off')
    plt.savefig("..//Img_res//Denoised//"+file_name[0]+".jpg")
    plt.close()
    img = cv2.imread("..//Img_res//Denoised//"+file_name[0]+".jpg")
    im = Image.fromarray(img)
    im = im.resize((150, 150), Image.LANCZOS)
    imgtk = ImageTk.PhotoImage(image=im)
    label = tkinter.Label(frame2, image=imgtk)
    label.image = imgtk
    label.place(x=200, y=20)
    messagebox.showinfo("INFO", "Denoising has been done successfully")
    button2.configure(state="disabled")
    
def augmentation():
    datagen = ImageDataGenerator(
        rotation_range=40,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=(0.5, 1.5),
    )

    # Defining augmentation parameters and generating 5 samples
    # Loading a sample image
    img = load_img("..//Img_res//Denoised//"+file_name[0]+".jpg")
    # Converting the input sample image to an array
    x = img_to_array(img)
    # Reshaping the input image
    x = x.reshape((1,) + x.shape)
    # Generating and saving 5 augmented samples
    # using the above defined parameters.
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir='..\\Img_res\\Augmented',
                              save_prefix='file_name[0]', save_format='jpeg'):
        i += 1
        if i > 10:
            break
    messagebox.showinfo("INFO", "Augmentation has ben done successfully")
    button3.configure(state="disabled")

def results():
    from Code import Results
    img = cv2.imread("..//Graphs//Confusion_matrix.jpg")
    im = Image.fromarray(img)
    im = im.resize((150, 150), Image.LANCZOS)
    imgtk = ImageTk.PhotoImage(image=im)
    label = tkinter.Label(frame2, image=imgtk)
    label.image = imgtk
    label.place(x=370, y=20)

    img = cv2.imread("..//Graphs//Accuracy.jpg")
    im = Image.fromarray(img)
    im = im.resize((150, 150), Image.LANCZOS)
    imgtk = ImageTk.PhotoImage(image=im)
    label = tkinter.Label(frame2, image=imgtk)
    label.image = imgtk
    label.place(x=540, y=20)

    img = cv2.imread("..//Graphs//MCC.jpg")
    im = Image.fromarray(img)
    im = im.resize((150, 150), Image.LANCZOS)
    imgtk = ImageTk.PhotoImage(image=im)
    label = tkinter.Label(frame2, image=imgtk)
    label.image = imgtk
    label.place(x=710, y=20)

    img = cv2.imread("..//Graphs//Kappa_Coefficient.jpg")
    im = Image.fromarray(img)
    im = im.resize((150, 150), Image.LANCZOS)
    imgtk = ImageTk.PhotoImage(image=im)
    label = tkinter.Label(frame2, image=imgtk)
    label.image = imgtk
    label.place(x=30, y=200)

    img = cv2.imread("..//Graphs//G-Mean.jpg")
    im = Image.fromarray(img)
    im = im.resize((150, 150), Image.LANCZOS)
    imgtk = ImageTk.PhotoImage(image=im)
    label = tkinter.Label(frame2, image=imgtk)
    label.image = imgtk
    label.place(x=200, y=200)

    img = cv2.imread("..//Graphs//FDR.jpg")
    im = Image.fromarray(img)
    im = im.resize((150, 150), Image.LANCZOS)
    imgtk = ImageTk.PhotoImage(image=im)
    label = tkinter.Label(frame2, image=imgtk)
    label.image = imgtk
    label.place(x=370, y=200)

    img = cv2.imread("..//Graphs//time.jpg")
    im = Image.fromarray(img)
    im = im.resize((150, 150), Image.LANCZOS)
    imgtk = ImageTk.PhotoImage(image=im)
    label = tkinter.Label(frame2, image=imgtk)
    label.image = imgtk
    label.place(x=540, y=200)
    plt.show()

    messagebox.showinfo("INFO", "Results and graphs have been generated successfully")
    button4.configure(state="disabled")

heading = tkinter.Label(win,text="AN AUTOMATIC BRAIN fMRI AUGMENTATION FOR ADHD IDENTIFICATION USING \n"
                   "NEUROWAVELET CAPSULE NETWORK WITH GENERATIVE ADVERSARIAL NETWORK",font=('Times New Roman bold', 15),fg="DeepPink2")
heading.place(x=150, y=10)

frame=tkinter.Frame(win, height=350, width=150,  highlightthickness=1, highlightbackground="black")
frame.place(x=20, y=100)

button1=tkinter.Button(frame, text="Browse", fg = "white", bg = "blue", height= 1, width= 11, font=('Times New Roman bold', 12), command=browse)
button1.place(x=15, y=40)

button2=tkinter.Button(frame, text="Denoising", fg = "white", bg = "blue", height= 1, width= 11, font=('Times New Roman bold', 12), command = denoising)
button2.place(x=15, y=100)

button3=tkinter.Button(frame, text="Augmentation", fg = "white", bg = "blue", height= 1, width= 11, font=('Times New Roman bold', 12), command = augmentation)
button3.place(x=15, y=160)

button4=tkinter.Button(frame, text="Result", fg = "white", bg = "blue", height= 1, width= 11, font=('Times New Roman bold', 12), command = results)
button4.place(x=15, y=220)

button5=tkinter.Button(frame, text="Exit", fg = "white", bg = "blue", height= 1, width= 11, font=('Times New Roman bold', 12), command=disable_button)
button5.place(x=15, y=280)

frame2=tkinter.Frame(win, height=450, width=900, bg='white', highlightthickness=1, highlightbackground="black")
frame2.place(x=200, y=100)

win.mainloop()





















