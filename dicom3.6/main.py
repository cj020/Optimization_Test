import matplotlib
matplotlib.use('Agg') # Use the 'Agg' backend for matplotlib to avoid issues with GUI in headless environments (like servers or when running without a display). This allows us to generate plots without needing a graphical interface.

from dicom_loader import load_dicom, extract_dwell_points_with_dwell_time_and_direction, get_source_strength
import dose_contribution as dc
from ui import start_ui
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import os

def main():
    folder = r"C:\Users\jichen\Downloads\T00060\T00060"
    
    Length_source = 3.5 # active length of the source in mm
    Lambda = 1.113  # Dose-rate constant in cGy/(h·U), where U = µGy·m²/h = cGy·cm²/h

    volume, spacing, origin, direction, rtstruct, rtplan, rtdose, dose_ds = load_dicom(folder)

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
    
    dwells, count = extract_dwell_points_with_dwell_time_and_direction(rtplan = rtplan, local_directions = False) # new function that prints dwell points without repeats
    print(f"Extracted {count} dwell positions:")
    
    dwell_positions = np.array([d[3] for d in dwells]) # extract the dwell positions (x, y, z) from the dwells list
    norm_dwell_direction = np.array([d[4] for d in dwells]) # extract the normalized direction vectors from the dwells list
    dwell_times = np.array([d[2] for d in dwells]) # extract the dwell times from the dwells list
    # channels = np.array([d[1] for d in dwells]) # extract the channel numbers from the dwells list

    print(f"Total irradiation time: {np.sum(dwell_times):.2f} s")
    print(f"Non-zero dwell positions: {np.sum(dwell_times > 0)}")

    start_ui(volume, spacing, origin, dwell_positions)
    
    # Get RTDOSE grid info for computing on same grid
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

    # Reference dose (RTDOSE is in Gy — keep as-is since gamma uses relative dose)
    dose_ref = rtdose.astype(np.float32)
    print(f"\nMax reference dose (RTDOSE): {np.max(dose_ref):.6f} Gy")
    print(f"Mean reference dose (non-zero): {np.mean(dose_ref[dose_ref > 0]):.6f} Gy")

    # Gamma index comparison
    print("\n" + "="*60)
    print("GAMMA INDEX COMPARISON (3%/3mm)")
    print("="*60)

    gamma, pass_rate = dc.gamma_index_3d(
        dose_vol_1=total_dose,  # calculated dose in cGy
        dose_vol_2=dose_ref,    # reference dose in Gy (relative dose comparison)
        spacing=dose_spacing,
        gamma_dist=3.0,
        gamma_percentage=3.0,
        cut_off=0.1
    )  

    print(f"\n{'='*60}")
    print(f"GAMMA INDEX RESULTS")
    print(f"{'='*60}")

    print(f"Pass rate (3%/3mm): {pass_rate:.2f}%")
    valid_gamma = gamma[gamma > 0]
    if len(valid_gamma) > 0:
        print(f"Mean gamma: {np.mean(valid_gamma):.4f}")
        print(f"Max gamma: {np.max(valid_gamma):.4f}")
        print(f"Gamma <= 1: {np.sum(valid_gamma <= 1.0)} / {len(valid_gamma)} voxels")

    # Save comparison figure using relative dose
    mid_z = nz_d // 2
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Normalize for display
    rel_ref_display = dose_ref / np.max(dose_ref) * 100.0
    rel_calc_display = total_dose / np.max(total_dose) * 100.0 if np.max(total_dose) > 0 else total_dose

    ax = axes[0]
    im = ax.imshow(rel_ref_display[mid_z], cmap='jet', aspect='equal', vmin=0, vmax=100)
    ax.set_title("Reference (RTDOSE) - Relative")
    plt.colorbar(im, ax=ax, label='% of max')

    ax = axes[1]
    im = ax.imshow(rel_calc_display[mid_z], cmap='jet', aspect='equal', vmin=0, vmax=100)
    ax.set_title("Calculated (TG-43) - Relative")
    plt.colorbar(im, ax=ax, label='% of max')

    ax = axes[2]
    im = ax.imshow(gamma[mid_z], cmap='RdYlGn_r', vmin=0, vmax=2, aspect='equal')
    ax.set_title(f"Gamma Index (pass: {pass_rate:.1f}%)")
    plt.colorbar(im, ax=ax, label='Gamma')

    plt.tight_layout()
    plt.savefig('gamma_comparison.png', dpi=150)
    print(f"\nComparison figure saved to: gamma_comparison.png")

    # Save the dose arrays for further analysis
    np.save('dose_calculated_tg43.npy', total_dose)
    np.save('dose_reference_rtdose.npy', dose_ref)
    np.save('gamma_map.npy', gamma)
    print("Dose arrays saved to .npy files.")

if __name__ == "__main__": # Only run the code below if this file is being executed directly, not imported as a module in another file.
    main()
