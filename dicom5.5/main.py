import numpy as np
from dicom_loader import (
    load_dicom,
    extract_dwell_points_with_dwell_time_and_direction,
    get_source_strength,
    identify_structures,
    build_structure_masks,
)
from needle_mesh import mesh_needle_in_ptv
from dose_contribution import dose_contribution
from cost_function import cost_function, print_cost_report, PRESCRIPTION_PARAMS
from plan_evaluation import evaluate_plan, print_evaluation_report
from ipsa import run_ipsa

def main():
    # Patient folder may contain two RTDOSE objects (native TPS + resampled on CT grid).
    # This pipeline compares TG-43 calculated dose only against the native RTDOSE grid.
    folder = r"C:\Users\jichen\Documents\T00060\T00060"
    
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
    print("STRUCTURE SET (RTSTRUCT on CT grid)")
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
    print(f"\n  {'Structure':<30} {'Class':<6} {'Voxels':>10} {'Volume (cm3)':>14}")
    print(f"  {'-'*30} {'-'*6} {'-'*10} {'-'*14}")
    for s in structures:
        if s["name"] not in masks:
            continue
        m = masks[s["name"]]
        n_vox = int(m.sum())
        vol_cc = n_vox * voxel_vol_mm3 / 1000.0
        print(f"  {s['name']:<30} {s['classification']:<6} {n_vox:>10} {vol_cc:>14.2f}")

    # Build combined PTV mask (union of all PTV structures)
    ptv_mask = np.zeros(ct_shape, dtype=bool)
    for name in ptv_names:
        if name in masks:
            ptv_mask |= masks[name]
    print(f"\n  Combined PTV mask: {int(ptv_mask.sum())} voxels")


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
    print(f"\n  Source strength S_k = {S_k:.4f} U (µGy*m2/h)")

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
    # 4. Mesh each needle within PTV (+5 mm margin) at 1 mm spacing
    # ------------------------------------------------------------------
    
    print(f"\n{'='*70}")
    print("NEEDLE MESHING (PTV + 5 mm margin, 1 mm spacing)")
    print(f"{'='*70}")

    unique_channels = np.unique(channels)
    needle_meshes = {}   # channel_id -> mesh_points (K, 3)

    print(f"\n  {'Channel':<10} {'Dwells':>7} {'PTV entry (mm)':>15} "
          f"{'PTV exit (mm)':>14} {'PTV length':>11} {'Mesh pts':>9} "
          f"{'Mesh length':>12}")
    print(f"  {'-'*10} {'-'*7} {'-'*15} {'-'*14} {'-'*11} {'-'*9} {'-'*12}")

    total_mesh_points = 0

    for ch_id in unique_channels:
        ch_mask = channels == ch_id
        ch_dwells = dwell_positions[ch_mask]

        mesh_pts, t_entry, t_exit = mesh_needle_in_ptv(
            ch_dwells, ptv_mask, ct_origin, ct_spacing,
            margin=5.0, mesh_step=1.0,
        )

        needle_meshes[int(ch_id)] = mesh_pts

        if len(mesh_pts) == 0:
            print(f"  {int(ch_id):<10} {len(ch_dwells):>7}   (no PTV intersection)")
            continue

        ptv_len = t_exit - t_entry
        mesh_len = np.linalg.norm(mesh_pts[-1] - mesh_pts[0])
        total_mesh_points += len(mesh_pts)

        print(f"  {int(ch_id):<10} {len(ch_dwells):>7} {t_entry:>15.2f} "
              f"{t_exit:>14.2f} {ptv_len:>11.2f} {len(mesh_pts):>9} "
              f"{mesh_len:>12.2f}")

    print(f"\n  Total mesh points across all needles: {total_mesh_points}")

    # ------------------------------------------------------------------
     # 5. TG-43 dose calculation (uniform initial dwell times, on CT grid)
    # ------------------------------------------------------------------
    
    print(f"\n{'='*70}")
    print("TG-43 DOSE CALCULATION (uniform initial dwell times, on CT grid)")
    print(f"{'='*70}")

    active_mask = dwell_times > 0
    n_active = int(active_mask.sum())
    uniform_time = float(np.mean(dwell_times[active_mask]))
    initial_dwell_times = np.where(active_mask, uniform_time, 0.0)

    print(f"\n  Active dwell positions     : {n_active}")
    print(f"  Uniform dwell time         : {uniform_time:.4f} s")
    print(f"  Total irradiation time     : {initial_dwell_times.sum():.2f} s "
          f"(original: {dwell_times.sum():.2f} s)")

    total_dose_cgy = dose_contribution(
        dwell_pos=dwell_positions,
        norm_dwell_dir=dwell_directions,
        dwell_times=initial_dwell_times,
        volume=volume,
        spacing=tuple(ct_spacing),
        origin=tuple(ct_origin),
        L=Length_source,
        S_k=S_k,
        Lambda=Lambda,
    )

    dose_gy = total_dose_cgy / 100.0   # cGy -> Gy

    print(f"\n  Dose grid shape : {dose_gy.shape}")
    print(f"  Max dose        : {dose_gy.max():.4f} Gy")
    nonzero = dose_gy[dose_gy > 0]
    if len(nonzero) > 0:
        print(f"  Mean (>0)       : {nonzero.mean():.4f} Gy")
        print(f"  Non-zero voxels : {len(nonzero)}")

    # Dose statistics inside each structure
    print(f"\n  {'Structure':<30} {'Class':<6} "
          f"{'Dmin (Gy)':>10} {'Dmean (Gy)':>11} {'Dmax (Gy)':>10} {'D95 (Gy)':>10}")
    print(f"  {'-'*30} {'-'*6} {'-'*10} {'-'*11} {'-'*10} {'-'*10}")
    for s in structures:
        if s["name"] not in masks:
            continue
        m = masks[s["name"]]
        n_vox = int(m.sum())
        if n_vox == 0:
            print(f"  {s['name']:<30} {s['classification']:<6}   (empty mask)")
            continue
        d = dose_gy[m]
        d95 = float(np.percentile(d, 5))   # dose covering 95% of volume
        print(f"  {s['name']:<30} {s['classification']:<6} "
              f"{d.min():>10.4f} {d.mean():>11.4f} {d.max():>10.4f} {d95:>10.4f}")

    # ------------------------------------------------------------------       
    # 6. Cost function evaluation (initial plan)
    # ------------------------------------------------------------------

    J, breakdown = cost_function(dose_gy, masks, ptv_names, oar_names)
    print_cost_report(J, breakdown)

    # ------------------------------------------------------------------
    # 7. IPSA dose optimisation (100 iterations)
    # ------------------------------------------------------------------

    optimized_dwell_times, ipsa_history = run_ipsa(
        dwell_positions, dwell_directions, initial_dwell_times,
        initial_dose_gy=dose_gy,
        masks=masks, ptv_names=ptv_names, oar_names=oar_names,
        volume=volume, spacing=tuple(ct_spacing), origin=tuple(ct_origin),
        L=Length_source, S_k=S_k, Lambda=Lambda,
    )

    # ------------------------------------------------------------------
    # 8. Recompute dose with optimised dwell times
    # ------------------------------------------------------------------

    print(f"\n{'='*70}")
    print("TG-43 DOSE CALCULATION (optimised dwell times, on CT grid)")
    print(f"{'='*70}")

    opt_dose_cgy = dose_contribution(
        dwell_pos=dwell_positions,
        norm_dwell_dir=dwell_directions,
        dwell_times=optimized_dwell_times,
        volume=volume,
        spacing=tuple(ct_spacing),
        origin=tuple(ct_origin),
        L=Length_source,
        S_k=S_k,
        Lambda=Lambda,
    )

    opt_dose_gy = opt_dose_cgy / 100.0

    print(f"\n  Dose grid shape : {opt_dose_gy.shape}")
    print(f"  Max dose        : {opt_dose_gy.max():.4f} Gy")
    opt_nonzero = opt_dose_gy[opt_dose_gy > 0]
    if len(opt_nonzero) > 0:
        print(f"  Mean (>0)       : {opt_nonzero.mean():.4f} Gy")
        print(f"  Non-zero voxels : {len(opt_nonzero)}")

    print(f"\n  {'Structure':<30} {'Class':<6} "
          f"{'Dmin (Gy)':>10} {'Dmean (Gy)':>11} {'Dmax (Gy)':>10} {'D95 (Gy)':>10}")
    print(f"  {'-'*30} {'-'*6} {'-'*10} {'-'*11} {'-'*10} {'-'*10}")
    for s in structures:
        if s["name"] not in masks:
            continue
        m = masks[s["name"]]
        n_vox = int(m.sum())
        if n_vox == 0:
            print(f"  {s['name']:<30} {s['classification']:<6}   (empty mask)")
            continue
        d = opt_dose_gy[m]
        d95 = float(np.percentile(d, 5))
        print(f"  {s['name']:<30} {s['classification']:<6} "
              f"{d.min():>10.4f} {d.mean():>11.4f} {d.max():>10.4f} {d95:>10.4f}")

    # ------------------------------------------------------------------
    # 9. Cost function evaluation (optimised plan)
    # ------------------------------------------------------------------

    J_opt, breakdown_opt = cost_function(opt_dose_gy, masks, ptv_names, oar_names)
    print_cost_report(J_opt, breakdown_opt)

    # ------------------------------------------------------------------
    # 10. Clinical plan evaluation (DVH criteria vs prescription)
    # ------------------------------------------------------------------

    eval_report = evaluate_plan(
        opt_dose_gy, masks, ptv_names, oar_names,
        spacing_mm=ct_spacing,
        D_p=PRESCRIPTION_PARAMS["D_p"],
    )
    print_evaluation_report(eval_report, D_p=PRESCRIPTION_PARAMS["D_p"])

    # ------------------------------------------------------------------
    # 11. RTDOSE (loaded for future reference, not used in optimization)
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
    # 12. Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  CT grid          : {ct_shape}  spacing {tuple(ct_spacing)} mm")
    print(f"  PTV mask(s)      : {ptv_names}")
    print(f"  OAR mask(s)      : {oar_names}")
    print(f"  Dwell positions  : {dwell_positions.shape[0]}")
    print(f"  Source strength   : {S_k:.4f} U")
    print(f"\n  Initial (uniform) dose : max {dose_gy.max():.4f} Gy   cost J = {J:.2f}")
    print(f"  Optimised dose         : max {opt_dose_gy.max():.4f} Gy   cost J = {J_opt:.2f}")
    print(f"  Cost reduction         : {(1 - J_opt / J) * 100:.1f}%")
    n_fail = sum(
        1 for role in eval_report.values()
        for c in role.get("checks", [])
        if c["status"] == "FAIL"
    )
    n_checks = sum(len(role.get("checks", [])) for role in eval_report.values())
    print(f"  Clinical criteria      : {n_checks - n_fail}/{n_checks} met "
          f"({n_fail} failed)")
    print()


if __name__ == "__main__":
    main()