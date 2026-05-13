import matplotlib
matplotlib.use('TkAgg')

from dicom_loader import load_dicom, extract_dwell_points, extract_dwell_points_with_dwell_time_and_local_direction
from ui import start_ui
import dose_contribution as dc
import numpy as np

def main():
    folder = r"C:\Users\jichen\Downloads\T00060\T00060"
    S_k = 1 # air kerma strength in cGy*cm^2/h
    Length_source = 3.5 # active length of the source in mm
    Lambda = 1.113 # Dose-rate constant, in cGy/h/U, where U is the unit of air kerma strength (cGy*cm^2/h)

    volume, spacing, origin, direction, rtstruct, rtplan, rtdose = load_dicom(folder)

    # print("Volume shape:", volume.shape)
    print("Intensity range:", volume.min(), volume.max())

    extract_dwell_points(rtplan) # old function that prints dwell points with repeats, for comparison

    dwells, count = extract_dwell_points_with_dwell_time_and_local_direction(rtplan = rtplan, local_directions = False) # new function that prints dwell points without repeats
    print(f"Extracted {count} dwell positions:")
    
    dwell_positions = np.array([d[3] for d in dwells]) # extract the dwell positions (x, y, z) from the dwells list
    norm_dwell_direction = np.array([d[4] for d in dwells]) # extract the normalized direction vectors from the dwells list
    dwell_times = np.array([d[2] for d in dwells]) # extract the dwell times from the dwells list
    channels = np.array([d[1] for d in dwells]) # extract the channel numbers from the dwells list

    start_ui(volume, spacing, origin, dwell_positions)
    
    count = 2 # for testing, only calculate dose for the first 2 dwell positions to save time

    distance, cosine_direction_to_voxel, angle_to_voxel, beta_value, dose_rate = dc.dose_contribution(dwell_pos=dwell_positions,                                                                                                
                                                                                                      norm_dwell_dir=norm_dwell_direction, 
                                                                                                      dwell_count=count, 
                                                                                                      volume=volume, 
                                                                                                      spacing=spacing, 
                                                                                                      origin=origin, 
                                                                                                      L=Length_source,
                                                                                                      S_k=S_k,
                                                                                                      Lambda=Lambda)
    print("Distance shape:", distance.shape)
    print("Distance:", distance)
    print("Cosine direction shape:", cosine_direction_to_voxel.shape)
    print("Cosine direction:", cosine_direction_to_voxel)
    print("Angle to voxel shape:", angle_to_voxel.shape)
    print("Angle to voxel:", angle_to_voxel)
    print("Beta angle shape:", beta_value.shape)
    print("Beta angle:", beta_value)
    print("Dose rate shape:", dose_rate.shape)
    print("Dose rate:", dose_rate)
    
    dose = np.zeros((count, *volume.shape), dtype=np.float32)
    for i in range(count):
        dose[i] = dose_rate[i] * dwell_times[i] # dose = dose rate * time

    print("Dose shape:", dose.shape)
    print("Dose:", dose)

if __name__ == "__main__": # Only run the code below if this file is being executed directly, not imported as a module in another file.
    main()
