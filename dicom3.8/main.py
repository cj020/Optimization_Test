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
    # Patient folder may contain two RTDOSE objects (native TPS + resampled on CT grid).
    # This pipeline compares TG-43 calculated dose only against the native RTDOSE grid.
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

    # skip interactive CT viewer when using Agg (batch / headless runs)
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
    dose_ref = rtdose.astype(np.float32)

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

    total_dose_cgy = dc.dose_contribution(
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

    # TG-43 output is in cGy; RTDOSE is in Gy — convert for comparison
    dose_calc = (total_dose_cgy / 100.0).astype(np.float32)

    print(f"\nCalculated dose shape: {dose_calc.shape}")
    print(f"Max calculated dose: {np.max(dose_calc):.6f} Gy")
    nonzero_mask = dose_calc > 0
    if np.any(nonzero_mask):
        print(f"Mean calculated dose (non-zero): {np.mean(dose_calc[nonzero_mask]):.6f} Gy")
    if np.max(dose_ref) > 0:
        print(f"Max dose ratio (calc/ref): {np.max(dose_calc) / np.max(dose_ref):.3f}")

    print(f"\nMax reference dose (native RTDOSE): {np.max(dose_ref):.6f} Gy")
    print(f"Mean reference dose (non-zero): {np.mean(dose_ref[dose_ref > 0]):.6f} Gy")
    
    # Gamma index comparison
    print("\n" + "="*60)
    print("GAMMA INDEX COMPARISON (3%/3mm)")
    print("="*60)

    # dose_vol_1 = reference (TPS), dose_vol_2 = evaluation (TG-43)
    gamma, pass_rate = dc.gamma_index_3d(
        dose_vol_1=dose_ref,    # reference dose in Gy (relative dose comparison)
        dose_vol_2=dose_calc,   # calculated dose in cGy
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

    # Save comparison figure at the z-slice of maximum reference dose
    max_ref_idx = np.unravel_index(np.argmax(dose_ref), dose_ref.shape)
    best_z = max_ref_idx[0]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Normalize for display
    rel_ref_display = dose_ref / np.max(dose_ref) * 100.0
    rel_calc_display = dose_calc / np.max(dose_calc) * 100.0 if np.max(dose_calc) > 0 else dose_calc

    ax = axes[0, 0]
    im = ax.imshow(rel_ref_display[best_z], cmap='jet', aspect='equal', vmin=0, vmax=100)
    ax.set_title(f"Reference (RTDOSE) - z={best_z}")
    plt.colorbar(im, ax=ax, label='% of max')

    ax = axes[0, 1]
    im = ax.imshow(rel_calc_display[best_z], cmap='jet', aspect='equal', vmin=0, vmax=100)
    ax.set_title(f"Calculated (TG-43) - z={best_z}")
    plt.colorbar(im, ax=ax, label='% of max')

    ax = axes[0, 2]
    im = ax.imshow(gamma[best_z], cmap='RdYlGn_r', vmin=0, vmax=2, aspect='equal')
    ax.set_title(f"Gamma Index (pass: {pass_rate:.1f}%)")
    plt.colorbar(im, ax=ax, label='Gamma')

    # Row 2: Line profiles through max-ref location
    peak_y, peak_x = max_ref_idx[1], max_ref_idx[2]
    ax = axes[1, 0]
    ax.plot(rel_ref_display[best_z, peak_y, :], 'b-', label='Reference')
    ax.plot(rel_calc_display[best_z, peak_y, :], 'r--', label='Calculated')
    ax.set_title(f"X-profile at y={peak_y}, z={best_z}")
    ax.set_xlabel('x index')
    ax.set_ylabel('% of max')
    ax.legend()
    ax.set_ylim(0, 110)

    ax = axes[1, 1]
    ax.plot(rel_ref_display[best_z, :, peak_x], 'b-', label='Reference')
    ax.plot(rel_calc_display[best_z, :, peak_x], 'r--', label='Calculated')
    ax.set_title(f"Y-profile at x={peak_x}, z={best_z}")
    ax.set_xlabel('y index')
    ax.set_ylabel('% of max')
    ax.legend()
    ax.set_ylim(0, 110)

    # Z-profile through peak
    ax = axes[1, 2]
    ax.plot(rel_ref_display[:, peak_y, peak_x], 'b-', label='Reference')
    ax.plot(rel_calc_display[:, peak_y, peak_x], 'r--', label='Calculated')
    ax.set_title(f"Z-profile at x={peak_x}, y={peak_y}")
    ax.set_xlabel('z index')
    ax.set_ylabel('% of max')
    ax.legend()
    ax.set_ylim(0, 110)

    plt.tight_layout()
    plt.savefig('gamma_comparison.png', dpi=150)
    print(f"\nComparison figure saved to: gamma_comparison.png")

    # Save the dose arrays for further analysis
    np.save('dose_calculated_tg43.npy', dose_calc)
    np.save('dose_reference_rtdose.npy', dose_ref)
    np.save('gamma_map.npy', gamma)
    print("Dose arrays saved to .npy files.")

if __name__ == "__main__": # Only run the code below if this file is being executed directly, not imported as a module in another file.
    main()
