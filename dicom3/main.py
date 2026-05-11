import matplotlib
matplotlib.use('TkAgg')

from dicom_loader import load_dicom, extract_dwell_points, extract_dwell_points_with_dwell_time_and_local_direction
from ui import start_ui
import dose_contribution as dc
import numpy as np

def main():
    folder = r"C:\Users\jichen\Downloads\T00060\T00060"

    volume, spacing, origin, direction, rtstruct, rtplan, rtdose = load_dicom(folder)

    # print("Volume shape:", volume.shape)
    print("Intensity range:", volume.min(), volume.max())

    extract_dwell_points(rtplan) # old function that prints dwell points with repeats, for comparison

    dwells, count = extract_dwell_points_with_dwell_time_and_local_direction(rtplan = rtplan, local_directions = False) # new function that prints dwell points without repeats
    print(f"Extracted {count} dwell positions:")
    
    dwell_positions = np.array([d[3] for d in dwells]) # extract the dwell positions (x, y, z) from the dwells list
    norm_dwell_local_direction = np.array([d[4] for d in dwells]) # extract the local direction vectors from the dwells list
    dwell_times = np.array([d[2] for d in dwells]) # extract the dwell times from the dwells list
    channels = np.array([d[1] for d in dwells]) # extract the channel numbers from the dwells list

    start_ui(volume, spacing, origin, dwell_positions)
    
    distance, cosine_direction_to_voxel, angle_to_voxel = dc.dose_contribution(dwell_positions, norm_dwell_local_direction, 5, rtdose, volume, spacing, origin)
    print("Distance shape:", distance.shape)
    print("Distance:", distance)
    print("Cosine direction shape:", cosine_direction_to_voxel.shape)
    print("Cosine direction:", cosine_direction_to_voxel)
    print("Angle to voxel shape:", angle_to_voxel.shape)
    print("Angle to voxel:", angle_to_voxel)

if __name__ == "__main__": # Only run the code below if this file is being executed directly, not imported as a module in another file.
    main()
