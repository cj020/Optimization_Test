# Loop over all files in the folder
import os
import pydicom


folder = r"C:\Users\jichen\Downloads\T00060-20260422T210320Z-3-001\T00060" # r: means raw string, which tells Python to treat the backslashes as literal characters and not as escape characters. This is important for Windows file paths, which use backslashes.

for f in os.listdir(folder): # os.listdir(folder): returns all file names in the folder, f = one file name at a time
    path = os.path.join(folder, f) # os.path.join(folder, f): combines the folder path and file name to get the full path of the file
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True) # pydicom.dcmread(path, stop_before_pixels=True): reads the DICOM file at the specified path, but only reads the metadata (not the pixel data) to save time and memory
        print(f, ds.Modality) # ds.Modality tells you what type of file it is
    except:
        pass


# Read CT series
# Do NOT just load files randomly. You must sort them properly
import SimpleITK as sitk

reader = sitk.ImageSeriesReader() # reader = sitk.ImageSeriesReader()': creates an instance of the ImageSeriesReader class from the SimpleITK library, which is used to read a series of DICOM images as a single 3D image.
series_IDs = reader.GetGDCMSeriesIDs(folder) # reader.GetGDCMSeriesIDs(folder): retrieves a list of unique series IDs from the DICOM files in the specified folder. Each series ID corresponds to a different imaging series (e.g., CT, MRI) in the DICOM dataset. 

print("Series IDs:", series_IDs)

# Takes one series ID, Returns all file paths belonging to that series
files = reader.GetGDCMSeriesFileNames(folder, series_IDs[0])
reader.SetFileNames(files) # Loads the list of slice files into the reader

image = reader.Execute() #This does the heavy work: reads all slices, sorts them, stacks them, applies spacing/orientation. Output: a 3D image object.

# Convert to NumPy
import numpy as np

volume = sitk.GetArrayFromImage(image)  # sitk.GetArrayFromImage(image): converts the SimpleITK image object into a NumPy array. The resulting array will have dimensions corresponding to the number of slices, height, and width of the image volume. 

print("Shape:", volume.shape) # Shape: (z, y, x) = z slices, each slice is x × y pixels

# Get geometry
spacing = image.GetSpacing()   # (dx, dy, dz) = physical distance between adjacent pixels in each dimension, in millimeters. This is important for accurate measurements and analysis of the image data. The spacing values are typically stored in the DICOM metadata and are used to convert pixel coordinates to real-world coordinates.
origin = image.GetOrigin() # (x0, y0, z0) = physical coordinates of the center of the first pixel in the image. This is important for accurately locating features in the image and for registering the image with other datasets. The origin values are typically stored in the DICOM metadata and are used to convert pixel coordinates to real-world coordinates.
direction = image.GetDirection() # direction = image.GetDirection(): retrieves the direction cosines of the image, which describe the orientation of the image axes in physical space. The direction cosines are typically stored in the DICOM metadata and are used to convert pixel coordinates to real-world coordinates, especially when the image is not aligned with the standard axes (e.g., if it is rotated or tilted). The direction cosines are represented as a 3x3 matrix, where each column corresponds to a unit vector along one of the image axes (x, y, z) in physical space.

print("Spacing:", spacing)
print("Origin:", origin)
print("Direction:", direction)

# Visualize one slice
import matplotlib.pyplot as plt

plt.imshow(volume[volume.shape[0] // 2], cmap='gray') # volume[volume.shape[0] // 2]: selects the middle slice of the volume for visualization. The expression volume.shape[0] // 2 calculates the index of the middle slice along the z-axis (the first dimension of the volume array). This allows you to visualize a representative slice from the middle of the 3D image volume. (// Floor Division): This operator performs division and then rounds the result down to the nearest whole number (the floor). The return type is an integer if both operands are integers, or a float if at least one operand is a float.
plt.title("Middle slice")
plt.show()

# Read RTPLAN (for dwell positions)
for f in os.listdir(folder):
    path = os.path.join(folder, f)
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    
    if ds.Modality == "RTPLAN":
        plan = pydicom.dcmread(path)
        print("Found RTPLAN:", f)

# Read RTDOSE
for f in os.listdir(folder):
    path = os.path.join(folder, f)
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    
    if ds.Modality == "RTDOSE":
        dose_ds = pydicom.dcmread(path)
        dose = dose_ds.pixel_array * dose_ds.DoseGridScaling # dose_ds.pixel_array: retrieves the pixel data from the RTDOSE DICOM file as a NumPy array. This array contains the raw dose values for each voxel in the 3D dose distribution. The shape of this array will typically be (z, y, x), where z is the number of slices, and y and x are the dimensions of each slice.
        print("Dose shape:", dose.shape)