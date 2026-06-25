import numpy as np
from dicom_loader import (
    load_dicom,
    extract_dwell_points_with_dwell_time_and_direction,
    get_source_strength,
    identify_structures,
    build_structure_masks,
)

def main():
    # Patient folder may contain two RTDOSE objects (native TPS + resampled on CT grid).
    # This pipeline compares TG-43 calculated dose only against the native RTDOSE grid.
    folder = r"C:\Users\jichen\Downloads\T00060\T00060"
    
    Length_source = 3.5 # active length of the source in mm
    Lambda = 1.113  # Dose-rate constant in cGy/(h·U), where U = µGy·m²/h = cGy·cm²/h

    # ------------------------------------------------------------------
    # 1. Load all DICOM data
    # ------------------------------------------------------------------

    volume, spacing, origin, direction, rtstruct, rtplan, rtdose, dose_ds = load_dicom(folder)

    ct_shape = volume.shape          # (nz, ny, nx)
    ct_spacing = np.array(spacing)   # (dx, dy, dz) mm
    ct_origin = np.array(origin)     # (x0, y0, z0) mm
   
    print(f"\n{'='*70}")
    print("CT VOLUME")
    print(f"{'='*70}")
    print(f"  Shape (nz, ny, nx) : {ct_shape}")
    print(f"  Spacing (dx,dy,dz) : {ct_spacing} mm")
    print(f"  Origin  (x0,y0,z0) : {ct_origin} mm")
    print(f"  HU range           : [{volume.min()}, {volume.max()}]")

    # CT voxel-center world coordinates
    nx, ny, nz = ct_shape[2], ct_shape[1], ct_shape[0]
    ct_x = ct_origin[0] + np.arange(nx) * ct_spacing[0]
    ct_y = ct_origin[1] + np.arange(ny) * ct_spacing[1]
    ct_z = ct_origin[2] + np.arange(nz) * ct_spacing[2]

    print(f"\n  X range: [{ct_x[0]:.2f}, {ct_x[-1]:.2f}] mm  ({nx} voxels)")
    print(f"  Y range: [{ct_y[0]:.2f}, {ct_y[-1]:.2f}] mm  ({ny} voxels)")
    print(f"  Z range: [{ct_z[0]:.2f}, {ct_z[-1]:.2f}] mm  ({nz} voxels)")
  
    # ------------------------------------------------------------------
    # 2. Parse RTSTRUCT — PTV and OAR on the CT grid
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("STRUCTURE SET (RTSTRUCT → CT grid)")
    print(f"{'='*70}")

    if rtstruct is None:
        print("  ERROR: No RTSTRUCT loaded.")
        return

    structures, ptv_names, oar_names = identify_structures(rtstruct)

    print(f"\n  Total structures : {len(structures)}")
    print(f"  PTV ({len(ptv_names)})          : {ptv_names}")
    print(f"  OAR ({len(oar_names)})          : {oar_names}")

    print(f"\n  {'Name':<30} {'Type':<15} {'Class':<8} {'Slices':>8}")
    print(f"  {'-'*30} {'-'*15} {'-'*8} {'-'*8}")
    for s in structures:
        print(f"  {s['name']:<30} {s['interpreted_type']:<15} "
              f"{s['classification']:<8} {len(s['contours']):>8}")

    # Rasterize PTV and OAR masks on the CT grid
    print(f"\n  Rasterizing structure masks on CT grid {ct_shape} ...")
    masks = build_structure_masks(
        structures,
        grid_origin=ct_origin,
        grid_spacing=ct_spacing,
        grid_shape=ct_shape,
        classifications=["PTV", "OAR"],
    )

    # Summary of mask coverage
    voxel_vol_mm3 = float(np.prod(ct_spacing))
    print(f"\n  {'Structure':<30} {'Class':<6} {'Voxels':>10} {'Volume (cm³)':>14}")
    print(f"  {'-'*30} {'-'*6} {'-'*10} {'-'*14}")
    for s in structures:
        if s["name"] not in masks:
            continue
        m = masks[s["name"]]
        n_vox = int(m.sum())
        vol_cc = n_vox * voxel_vol_mm3 / 1000.0
        print(f"  {s['name']:<30} {s['classification']:<6} {n_vox:>10} {vol_cc:>14.2f}")

    # ------------------------------------------------------------------
    # 3. Extract dwell information from RTPLAN
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("RTPLAN — DWELL POSITIONS")
    print(f"{'='*70}")

    if rtplan is None:
        print("  ERROR: No RTPLAN loaded.")
        return

    S_k = get_source_strength(rtplan)
    if S_k is None:
        print("  WARNING: ReferenceAirKermaRate not found. Using S_k = 1.0")
        S_k = 1.0
    print(f"\n  Source strength S_k = {S_k:.4f} U (µGy·m²/h)")

    dwells, count = extract_dwell_points_with_dwell_time_and_direction(
        rtplan=rtplan, local_directions=False
    )

    dwell_positions = np.array([d[3] for d in dwells])       # (N, 3) world coords
    dwell_times = np.array([d[2] for d in dwells])           # (N,)   seconds
    dwell_directions = np.array([d[4] for d in dwells])      # (N, 3) unit vectors
    channels = np.array([d[1] for d in dwells])              # (N,)   channel IDs
    
    print(f"\n  Total dwell positions   : {count}")
    print(f"  Non-zero dwell times    : {np.sum(dwell_times > 0)}")
    print(f"  Total irradiation time  : {np.sum(dwell_times):.2f} s")
    print(f"  Unique channels         : {len(np.unique(channels))}")

    print(f"\n  Dwell X range: [{dwell_positions[:,0].min():.2f}, {dwell_positions[:,0].max():.2f}] mm")
    print(f"  Dwell Y range: [{dwell_positions[:,1].min():.2f}, {dwell_positions[:,1].max():.2f}] mm")
    print(f"  Dwell Z range: [{dwell_positions[:,2].min():.2f}, {dwell_positions[:,2].max():.2f}] mm")

    # ------------------------------------------------------------------
    # 4. RTDOSE (loaded for future reference, not used in optimization)
    # ------------------------------------------------------------------
    if rtdose is not None:
        print(f"\n{'='*70}")
        print("RTDOSE (loaded for reference — not used in optimization)")
        print(f"{'='*70}")
        print(f"  Shape : {rtdose.shape}")
        print(f"  Max   : {rtdose.max():.4f} Gy")
    else:
        print("\n  No RTDOSE loaded (optional for optimization).")

    # ------------------------------------------------------------------
    # 5. Summary — all data ready for optimization
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("OPTIMIZATION DATA READY")
    print(f"{'='*70}")
    print(f"  CT grid          : {ct_shape}  spacing {tuple(ct_spacing)} mm")
    print(f"  PTV mask(s)      : {ptv_names}")
    print(f"  OAR mask(s)      : {oar_names}")
    print(f"  Dwell positions  : {dwell_positions.shape[0]}")
    print(f"  Source strength   : {S_k:.4f} U")
    print()


if __name__ == "__main__":
    main()