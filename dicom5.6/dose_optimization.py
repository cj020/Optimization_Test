import matplotlib
matplotlib.use('Agg')

from dicom_loader import (
    load_dicom,
    extract_dwell_points_with_dwell_time_and_direction,
    get_source_strength,
    identify_structures,
    build_structure_masks,
)
import numpy as np
import matplotlib.pyplot as plt


def main():
    folder = r"C:\Users\jichen\Downloads\T00060\T00060"

    Length_source = 3.5   # active length of the source in mm
    Lambda = 1.113        # Dose-rate constant in cGy/(h·U)

    # ------------------------------------------------------------------
    # 1. Load all DICOM data (CT, RTSTRUCT, RTPLAN, RTDOSE)
    # ------------------------------------------------------------------
    volume, spacing, origin, direction, rtstruct, rtplan, rtdose, dose_ds = load_dicom(folder)

    print(f"\nCT Volume shape: {volume.shape}")
    print(f"CT Spacing (dx, dy, dz): {spacing} mm")
    print(f"CT Origin  (x0, y0, z0): {origin} mm")

    # ------------------------------------------------------------------
    # 2. Extract source / dwell information from RTPLAN
    # ------------------------------------------------------------------
    S_k = get_source_strength(rtplan)
    if S_k is None:
        print("WARNING: Could not find ReferenceAirKermaRate. Using S_k=1.")
        S_k = 1.0
    print(f"\nSource strength S_k = {S_k:.4f} U")

    dwells, count = extract_dwell_points_with_dwell_time_and_direction(
        rtplan=rtplan, local_directions=False
    )
    dwell_positions = np.array([d[3] for d in dwells])
    dwell_times = np.array([d[2] for d in dwells])
    print(f"Extracted {count} dwell positions, total irradiation time: {np.sum(dwell_times):.2f} s")

    # ------------------------------------------------------------------
    # 3. RTDOSE grid parameters
    # ------------------------------------------------------------------
    if dose_ds is None:
        print("ERROR: No RTDOSE file found. Cannot proceed.")
        return

    dose_origin = np.array(dose_ds.ImagePositionPatient, dtype=float)
    dy_dose, dx_dose = [float(x) for x in dose_ds.PixelSpacing]
    z_offsets = np.array(dose_ds.GridFrameOffsetVector, dtype=float)
    dz_dose = z_offsets[1] - z_offsets[0] if len(z_offsets) > 1 else 1.0

    dose_spacing = np.array([dx_dose, dy_dose, dz_dose])
    dose_shape = rtdose.shape  # (nz, ny, nx)
    dose_ref = rtdose.astype(np.float32)

    print(f"\nRTDOSE grid shape : {dose_shape}")
    print(f"RTDOSE spacing    : ({dx_dose}, {dy_dose}, {dz_dose}) mm")
    print(f"RTDOSE origin     : {dose_origin}")

    # ------------------------------------------------------------------
    # 4. Parse RTSTRUCT — identify CTV, PTV, and OAR
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STRUCTURE SET ANALYSIS")
    print("=" * 70)

    if rtstruct is None:
        print("ERROR: No RTSTRUCT loaded. Cannot identify structures.")
        return

    structures, ctv_names, ptv_names, oar_names = identify_structures(rtstruct)

    print(f"\nTotal structures found: {len(structures)}")
    print(f"  CTV structures ({len(ctv_names)}): {ctv_names}")
    print(f"  PTV structures ({len(ptv_names)}): {ptv_names}")
    print(f"  OAR structures ({len(oar_names)}): {oar_names}")

    print(f"\n{'Name':<30} {'Type':<15} {'Class':<8} {'Contour slices':>15}")
    print("-" * 70)
    for s in structures:
        print(f"{s['name']:<30} {s['interpreted_type']:<15} {s['classification']:<8} {len(s['contours']):>15}")

    # ------------------------------------------------------------------
    # 5. Build binary masks on the RTDOSE grid for CTV, PTV, and OAR
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RASTERIZING STRUCTURE MASKS ON RTDOSE GRID")
    print("=" * 70)

    masks = build_structure_masks(
        structures,
        grid_origin=dose_origin,
        grid_spacing=dose_spacing,
        grid_shape=dose_shape,
        classifications=["CTV", "PTV", "OAR"],
    )

    # ------------------------------------------------------------------
    # 6. Summary: dose statistics inside each structure
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DOSE STATISTICS PER STRUCTURE (from loaded RTDOSE)")
    print("=" * 70)

    print(f"\n{'Structure':<30} {'Class':<6} {'Voxels':>8} "
          f"{'Dmin (Gy)':>10} {'Dmean (Gy)':>11} {'Dmax (Gy)':>10} {'D95 (Gy)':>10}")
    print("-" * 90)

    for s in structures:
        if s["name"] not in masks:
            continue
        m = masks[s["name"]]
        n_vox = int(m.sum())
        if n_vox == 0:
            print(f"{s['name']:<30} {s['classification']:<6} {0:>8}   (empty mask)")
            continue
        dose_in_struct = dose_ref[m]
        d_min = float(np.min(dose_in_struct))
        d_mean = float(np.mean(dose_in_struct))
        d_max = float(np.max(dose_in_struct))
        d95 = float(np.percentile(dose_in_struct, 5))  # D95 = dose covering 95% of the volume
        print(f"{s['name']:<30} {s['classification']:<6} {n_vox:>8} "
              f"{d_min:>10.4f} {d_mean:>11.4f} {d_max:>10.4f} {d95:>10.4f}")

    # ------------------------------------------------------------------
    # 7. Visualization: overlay masks on a dose slice
    # ------------------------------------------------------------------
    max_ref_idx = np.unravel_index(np.argmax(dose_ref), dose_ref.shape)
    best_z = max_ref_idx[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    im = ax.imshow(dose_ref[best_z], cmap='jet', aspect='equal')
    ax.set_title(f"RTDOSE — z-slice {best_z}")
    plt.colorbar(im, ax=ax, label='Dose (Gy)')

    ax = axes[1]
    ax.imshow(dose_ref[best_z], cmap='gray', aspect='equal', alpha=0.5)
    colors_cycle = plt.cm.tab10.colors
    legend_handles = []
    ci = 0
    for s in structures:
        if s["name"] not in masks:
            continue
        m = masks[s["name"]]
        if not np.any(m[best_z]):
            continue
        color = colors_cycle[ci % len(colors_cycle)]
        contour_overlay = np.ma.masked_where(~m[best_z], np.ones_like(m[best_z], dtype=float))
        ax.contour(m[best_z].astype(float), levels=[0.5], colors=[color], linewidths=1.5)
        label = f"{s['name']} ({s['classification']})"
        legend_handles.append(plt.Line2D([0], [0], color=color, lw=2, label=label))
        ci += 1

    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper right', fontsize=7)
    ax.set_title(f"Structure contours — z-slice {best_z}")

    plt.tight_layout()
    plt.savefig('dose_structures_overview.png', dpi=150)
    print(f"\nOverview figure saved to: dose_structures_overview.png")

    print("\n" + "=" * 70)
    print("DOSE OPTIMIZATION SETUP COMPLETE")
    print("=" * 70)
    print(f"  {len(ctv_names)} CTV(s), {len(ptv_names)} PTV(s), and "
          f"{len(oar_names)} OAR(s) identified and rasterized")
    print("  Ready for dose optimization.\n")


if __name__ == "__main__":
    main()
