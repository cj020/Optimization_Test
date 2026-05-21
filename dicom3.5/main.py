import matplotlib
matplotlib.use('Agg') # Use the 'Agg' backend for matplotlib to avoid issues with GUI in headless environments (like servers or when running without a display). This allows us to generate plots without needing a graphical interface.

from dicom_loader import load_dicom, extract_dwell_points_with_dwell_time_and_local_direction
import dose_contribution as dc
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import os

def get_rtdose_grid_info(folder):
    """Re-read RTDOSE file to get grid geometry."""
    for root, dirs, files in os.walk(folder):
        for f in files:
            path = os.path.join(root, f)
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True)
                if getattr(ds, "Modality", None) == "RTDOSE":
                    dose_ds = pydicom.dcmread(path)
                    return dose_ds
            except Exception:
                continue
    return None

def get_source_strength(rtplan):
    """Extract ReferenceAirKermaRate from RTPLAN (in µGy·m²/h = cGy·cm²/h = U)."""
    if rtplan is None:
        return None
    for src in rtplan.SourceSequence:
        rakr = getattr(src, "ReferenceAirKermaRate", None)
        if rakr is not None:
            return float(rakr)
    return None

def main():
    folder = r"C:\Users\jichen\Downloads\T00060\T00060"
    
    Length_source = 3.5 # active length of the source in mm
    Lambda = 1.113  # Dose-rate constant in cGy/(h·U), where U = µGy·m²/h = cGy·cm²/h

    volume, spacing, origin, direction, rtstruct, rtplan, rtdose = load_dicom(folder)

    print(f"\nCT Volume shape: {volume.shape}")
    print(f"CT Spacing (dx, dy, dz): {spacing} mm")
    print(f"CT Origin (x0, y0, z0): {origin} mm")
    print("Intensity range:", volume.min(), volume.max())

    # Get source strength from RTPLAN (DICOM ReferenceAirKermaRate is in µGy·m²/h = U)
    S_k = get_source_strength(rtplan)
    if S_k is None:
        print("WARNING: Could not find ReferenceAirKermaRate in RTPLAN. Using S_k=1.")
        S_k = 1.0
    print(f"\nSource strength S_k = {S_k:.4f} U (µGy·m²/h = cGy·cm²/h)")
    print(f"Dose-rate constant Lambda = {Lambda:.4f} cGy/(h·U)")
    
    dwells, count = extract_dwell_points_with_dwell_time_and_local_direction(rtplan = rtplan, local_directions = False) # new function that prints dwell points without repeats
    print(f"Extracted {count} dwell positions:")
    
    dwell_positions = np.array([d[3] for d in dwells]) # extract the dwell positions (x, y, z) from the dwells list
    norm_dwell_direction = np.array([d[4] for d in dwells]) # extract the normalized direction vectors from the dwells list
    dwell_times = np.array([d[2] for d in dwells]) # extract the dwell times from the dwells list
    # channels = np.array([d[1] for d in dwells]) # extract the channel numbers from the dwells list

    print(f"Total irradiation time: {np.sum(dwell_times):.2f} s")
    print(f"Non-zero dwell positions: {np.sum(dwell_times > 0)}")

    # start_ui(volume, spacing, origin, dwell_positions)
    
    # Get RTDOSE grid info for computing on same grid
    dose_ds = get_rtdose_grid_info(folder)
    if dose_ds is None:
        print("ERROR: No RTDOSE file found. Cannot proceed.")
        return
    
    # RTDOSE grid coordinates
    dose_origin = np.array(dose_ds.ImagePositionPatient, dtype=float) # (x0, y0, z0) of the RTDOSE grid in mm
    dy_dose, dx_dose = [float(x) for x in dose_ds.PixelSpacing] # note the order: DICOM PixelSpacing is (row_spacing, column_spacing) = (dy, dx)
    z_offsets = np.array(dose_ds.GridFrameOffsetVector, dtype=float)
    dz_dose = z_offsets[1] - z_offsets[0] if len(z_offsets) > 1 else 1.0

    nz_d, ny_d, nx_d = rtdose.shape
    dose_spacing = (dx_dose, dy_dose, dz_dose)

    # Voxel center coordinates on the RTDOSE grid
    ref_x = np.arange(nx_d) * dx_dose + dose_origin[0]
    ref_y = np.arange(ny_d) * dy_dose + dose_origin[1]
    ref_z = z_offsets + dose_origin[2]

    print(f"\nRTDOSE grid: {rtdose.shape}")
    print(f"RTDOSE spacing (dx, dy, dz): ({dx_dose}, {dy_dose}, {dz_dose}) mm")
    print(f"RTDOSE origin: {dose_origin}")


    # Calculate TG-43 dose directly on the RTDOSE grid
    print("\n" + "="*60)
    print("TG-43 DOSE CALCULATION (on RTDOSE grid)")
    print("="*60)

    total_dose = dc.dose_contribution(
        dwell_pos=dwell_positions,
        norm_dwell_dir=norm_dwell_direction,
        dwell_times=dwell_times,
        volume=np.zeros((nz_d, ny_d, nx_d)),  # dummy volume with RTDOSE shape
        spacing=dose_spacing,
        origin=(dose_origin[0], dose_origin[1], dose_origin[2]),
        L=Length_source,
        S_k=S_k,
        Lambda=Lambda
    )

    print(f"\nCalculated dose shape: {total_dose.shape}")
    print(f"Max calculated dose: {np.max(total_dose):.6f} cGy")
    nonzero_mask = total_dose > 0
    if np.any(nonzero_mask):
        print(f"Mean calculated dose (non-zero): {np.mean(total_dose[nonzero_mask]):.6f} cGy")

    # the following needed to be fixed later
    print("Distance shape:", distance.shape)
    # print("Distance:", distance)
    print("Cosine direction shape:", cosine_direction_to_voxel.shape)
    # print("Cosine direction:", cosine_direction_to_voxel)
    print("Angle to voxel shape:", angle_to_voxel.shape)
    # print("Angle to voxel:", angle_to_voxel)
    print("Beta angle shape:", beta_value.shape)
    # print("Beta angle:", beta_value)
    print("Dose rate shape:", dose_rate.shape)
    # print("Dose rate:", dose_rate)
    
    dose = np.zeros((count, *volume.shape), dtype=np.float32)
    for i in range(count):
        dose[i] = dose_rate[i] * dwell_times[i] # dose = dose rate * time

    print("Dose shape:", dose.shape)
    print("Dose:", dose)

    print("Average dose rate:", np.average(dose_rate)) # print the average dose for testing
    print("Maximum dose rate:", np.max(dose_rate)) # print the maximum dose for testing
    print("Minimum dose rate:", np.min(dose_rate)) # print the minimum dose for testing
    print("Average dose:", np.average(dose)) # print the average dose for testing
    print("Maximum dose:", np.max(dose)) # print the maximum dose for testing
    print("Minimum dose:", np.min(dose)) 

def new_func():
    return# print the minimum dose for testing

if __name__ == "__main__": # Only run the code below if this file is being executed directly, not imported as a module in another file.
    main()
