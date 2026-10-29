import numpy as np
from dicom_loader import (
    load_dicom,
    extract_dwell_points_with_dwell_time_and_direction,
    extract_channel_geometry,
    get_source_strength,
    identify_structures,
    build_structure_masks,
    resample_volume_to_spacing,
)
from needle_mesh import (
    assemble_mesh_dwells,
    count_mesh_labels,
    label_mesh_points,
    mesh_channel_in_search_space,
    structure_search_bounds,
)
from dose_contribution import dose_contribution
from plan_evaluation import evaluate_plan, print_evaluation_report
from simultaneous_loss import evaluate_simultaneous_loss, print_simultaneous_loss
from ipsa import run_ipsa
from ui import start_ui

def main():
    # Patient folder may contain two RTDOSE objects (native TPS + resampled on CT grid).
    # This pipeline compares TG-43 calculated dose only against the native RTDOSE grid.
    folder = r"C:\Users\jichen\Documents\Cases\us test pt"
    
    Length_source = 3.5 # active length of the source in mm
    Lambda = 1.113  # Dose-rate constant in cGy/(h·U), where U = µGy·m²/h = cGy·cm²/h

    # ------------------------------------------------------------------
    # 1. Load all DICOM data
    # ------------------------------------------------------------------

    volume, spacing, origin, direction, rtstruct, rtplan, rtdose, dose_ds, modality = load_dicom(folder)

    # Native TRUS is typically ~0.2 mm in-plane. TG-43 / IPSA run on a 1 mm
    # calculation grid, matching typical CT voxel size.
    if modality == "US":
        print("\n  US native grid is finer than 1 mm; downsampling for dose calculation.")
        volume, spacing, origin = resample_volume_to_spacing(
            volume, spacing, origin, target_mm=1.0
        )
        
    vol_shape = volume.shape          # (nz, ny, nx)
    vol_spacing = np.array(spacing)   # (dx, dy, dz) mm
    vol_origin = np.array(origin)     # (x0, y0, z0) mm
   
    print(f"\n{'='*70}")
    print(f"{modality} VOLUME")
    print(f"{'='*70}")
    print(f"  Shape (nz, ny, nx) : {vol_shape}")
    print(f"  Spacing (dx,dy,dz) : {vol_spacing} mm")
    print(f"  Origin  (x0,y0,z0) : {vol_origin} mm")
    print(f"  Value range        : [{volume.min()}, {volume.max()}]")

    # Voxel-center world coordinates
    nx, ny, nz = vol_shape[2], vol_shape[1], vol_shape[0]
    vol_x = vol_origin[0] + np.arange(nx) * vol_spacing[0]
    vol_y = vol_origin[1] + np.arange(ny) * vol_spacing[1]
    vol_z = vol_origin[2] + np.arange(nz) * vol_spacing[2]

    print(f"\n  X range: [{vol_x[0]:.2f}, {vol_x[-1]:.2f}] mm  ({nx} voxels)")
    print(f"  Y range: [{vol_y[0]:.2f}, {vol_y[-1]:.2f}] mm  ({ny} voxels)")
    print(f"  Z range: [{vol_z[0]:.2f}, {vol_z[-1]:.2f}] mm  ({nz} voxels)")
  
    # ------------------------------------------------------------------
    # 2. Parse RTSTRUCT — CTV, PTV, and OAR on the image grid
    # ------------------------------------------------------------------
    
    print(f"\n{'='*70}")
    print(f"STRUCTURE SET (RTSTRUCT on {modality} grid)")
    print(f"{'='*70}")

    if rtstruct is None:
        print("  ERROR: No RTSTRUCT loaded.")
        return

    structures, ctv_names, ptv_names, oar_names = identify_structures(rtstruct)

    print(f"\n  Total structures : {len(structures)}")
    print(f"  CTV ({len(ctv_names)})          : {ctv_names}")
    print(f"  PTV ({len(ptv_names)})          : {ptv_names}")
    print(f"  OAR ({len(oar_names)})          : {oar_names}")

    print(f"\n  {'Name':<30} {'Type':<15} {'Class':<8} {'Slices':>8}")
    print(f"  {'-'*30} {'-'*15} {'-'*8} {'-'*8}")
    for s in structures:
        print(f"  {s['name']:<30} {s['interpreted_type']:<15} "
              f"{s['classification']:<8} {len(s['contours']):>8}")

    # Rasterize CTV, PTV, and OAR masks on the image grid
    print(f"\n  Rasterizing structure masks on {modality} grid {vol_shape} ...")

    # masks function returns a 3D Boolean NumPy array that indicates whether each voxel belongs to a particular anatomical structure.
    masks = build_structure_masks(
        structures,
        grid_origin=vol_origin,
        grid_spacing=vol_spacing,
        grid_shape=vol_shape,
        classifications=["CTV", "PTV", "OAR"],
    )

    # Summary of mask coverage
    voxel_vol_mm3 = float(np.prod(vol_spacing))
    print(f"\n  {'Structure':<30} {'Class':<6} {'Voxels':>10} {'Volume (cm3)':>14}")
    print(f"  {'-'*30} {'-'*6} {'-'*10} {'-'*14}")
    for s in structures:
        if s["name"] not in masks:
            continue
        m = masks[s["name"]]
        n_vox = int(m.sum())
        vol_cc = n_vox * voxel_vol_mm3 / 1000.0
        print(f"  {s['name']:<30} {s['classification']:<6} {n_vox:>10} {vol_cc:>14.2f}")

    # Search space for meshing: bounding box of all CTV, PTV, and OAR
    search_names = [
        name for name in (ctv_names + ptv_names + oar_names) if name in masks
    ]
    structure_classes = {s["name"]: s["classification"] for s in structures}


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
    # 4. Mesh each channel along the needle line inside the PTV+OAR box
    # ------------------------------------------------------------------
    
    print(f"\n{'='*70}")
    print("NEEDLE MESHING (channel line ∩ CTV/PTV/OAR bounding box, 1 mm)")
    print(f"{'='*70}")

    plan_channels = extract_channel_geometry(rtplan)
    if not plan_channels:
        print("  ERROR: No channel start/end points found in RTPLAN.")
        return

    box_min, box_max = structure_search_bounds(
        masks, vol_origin, vol_spacing, names=search_names,
    )
    if box_min is None:
        print("  ERROR: Search space is empty (no CTV/PTV/OAR voxels).")
        return

    print(f"\n  Search space (AABB of CTV + PTV + OAR):")
    print(f"    X : [{box_min[0]:.2f}, {box_max[0]:.2f}] mm")
    print(f"    Y : [{box_min[1]:.2f}, {box_max[1]:.2f}] mm")
    print(f"    Z : [{box_min[2]:.2f}, {box_max[2]:.2f}] mm")
    print(f"    Size: {box_max[0] - box_min[0]:.2f} × "
          f"{box_max[1] - box_min[1]:.2f} × "
          f"{box_max[2] - box_min[2]:.2f} mm")
    print(f"  Channels from RTPLAN (dwell positions ignored): {len(plan_channels)}")

    needle_meshes = {}   # channel_id -> mesh_points (K, 3)
    needle_labels = {}   # channel_id -> list of label dicts

    print(f"\n  {'Channel':<10} {'CPs':>5} {'Needle (mm)':>12} "
          f"{'t_enter':>10} {'t_exit':>10} {'Mesh pts':>9} "
          f"{'CTV/PTV':>8} {'OAR':>6} {'Out':>6}")
    print(f"  {'-'*10} {'-'*5} {'-'*12} {'-'*10} {'-'*10} {'-'*9} "
          f"{'-'*8} {'-'*6} {'-'*6}")

    total_mesh_points = 0

    for ch in plan_channels:
        ch_id = int(ch["channel_id"])
        needle_len = float(np.linalg.norm(ch["end"] - ch["start"]))
        mesh_pts, t_entry, t_exit = mesh_channel_in_search_space(
            ch["start"], ch["end"], box_min, box_max, mesh_step=1.0,
        )
        labels = label_mesh_points(
            mesh_pts, masks, vol_origin, vol_spacing, structure_classes,
        )
        needle_meshes[ch_id] = mesh_pts
        needle_labels[ch_id] = labels

        if len(mesh_pts) == 0:
            print(f"  {ch_id:<10} {ch['n_control_points']:>5} {needle_len:>12.2f} "
                  f"  (no search-space intersection)")
            continue

        n_target = sum(1 for lab in labels if lab["in_ctv"] or lab["in_ptv"])
        n_oar = sum(1 for lab in labels if lab["in_oar"])
        n_out = sum(1 for lab in labels if lab["region"] == "OUTSIDE")
        total_mesh_points += len(mesh_pts)

        print(f"  {ch_id:<10} {ch['n_control_points']:>5} {needle_len:>12.2f} "
              f"{t_entry:>10.2f} {t_exit:>10.2f} {len(mesh_pts):>9} "
              f"{n_target:>8} {n_oar:>6} {n_out:>6}")

    print(f"\n  Total mesh points across all needles: {total_mesh_points}")

    mesh_dwells = assemble_mesh_dwells(needle_meshes, needle_labels)
    mesh_positions = mesh_dwells["positions"]
    mesh_directions = mesh_dwells["directions"]
    mesh_channels = mesh_dwells["channel_ids"]
    mesh_labels = mesh_dwells["labels"]
    n_mesh = len(mesh_positions)

    region_counts, organ_counts = count_mesh_labels(mesh_labels)
    print(f"\n  Mesh labels by region (OAR vertices are kept for optimisation):")
    for region in sorted(region_counts):
        print(f"    {region:<12}: {region_counts[region]}")
    print(f"  Mesh labels by organ (a vertex may count in more than one):")
    for organ in sorted(organ_counts):
        print(f"    {organ:<30}: {organ_counts[organ]}")

    if n_mesh == 0:
        print("  ERROR: No mesh points to use as candidate dwells.")
        return

    # ------------------------------------------------------------------
    # 5. TG-43 dose calculation (uniform time on every mesh vertex)
    # ------------------------------------------------------------------
    
    print(f"\n{'='*70}")
    print(f"TG-43 DOSE CALCULATION (uniform time on all mesh vertices, on {modality} grid)")
    print(f"{'='*70}")

    orig_active = dwell_times[dwell_times > 0]
    orig_total = float(orig_active.sum()) if len(orig_active) else 0.0
    orig_mean = float(orig_active.mean()) if len(orig_active) else 1.0
    orig_max = float(orig_active.max()) if len(orig_active) else 1.0
    # Same total irradiation time as the original plan, spread evenly over
    # every mesh vertex. IPSA then turns unused vertices off (time -> 0)
    # and slides remaining activity along each catheter.
    uniform_time = orig_total / n_mesh if n_mesh else 0.0
    initial_dwell_times = np.full(n_mesh, uniform_time, dtype=np.float64)

    print(f"\n  Candidate mesh vertices    : {n_mesh}")
    print(f"  Original active dwells     : {len(orig_active)}")
    print(f"  Original mean dwell time   : {orig_mean:.4f} s")
    print(f"  Uniform mesh dwell time    : {uniform_time:.4f} s")
    print(f"  Total irradiation time     : {initial_dwell_times.sum():.2f} s "
          f"(original: {orig_total:.2f} s)")

    total_dose_cgy = dose_contribution(
        dwell_pos=mesh_positions,
        norm_dwell_dir=mesh_directions,
        dwell_times=initial_dwell_times,
        volume=volume,
        spacing=tuple(vol_spacing),
        origin=tuple(vol_origin),
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
    # 6. Cost-function parameters (GUI) and initial evaluation
    # ------------------------------------------------------------------

    rx_params = start_ui(
        volume, tuple(vol_spacing), tuple(vol_origin), mesh_positions,
        dose_gy=dose_gy, masks=masks, ctv_names=ctv_names, oar_names=oar_names,
        mesh_labels=mesh_labels,
    )

    L, breakdown = evaluate_simultaneous_loss(
        dose_gy, initial_dwell_times, masks, ctv_names, oar_names, rx_params,
    )
    print_simultaneous_loss(L, breakdown, rx_params)

    # ------------------------------------------------------------------
    # 7. IPSA: dwell time + mesh-position optimisation
    # ------------------------------------------------------------------

    optimized_dwell_times, ipsa_history = run_ipsa(
        mesh_positions, mesh_directions, initial_dwell_times,
        initial_dose_gy=dose_gy,
        masks=masks, ctv_names=ctv_names, oar_names=oar_names,
        volume=volume, spacing=tuple(vol_spacing), origin=tuple(vol_origin),
        L=Length_source, S_k=S_k, Lambda=Lambda,
        rx_params=rx_params,
        channel_ids=mesh_channels,
        ipsa_params={
            # Bounds / step size from the original plan, not the small
            # per-vertex uniform start, so activity can concentrate.
            "t_max": 3.0 * orig_max,
            "perturbation_std": 0.3 * orig_mean,
        },
    )

    active_mask = optimized_dwell_times > 0
    active_labels = [lab for lab, on in zip(mesh_labels, active_mask) if on]
    active_region, active_organ = count_mesh_labels(active_labels)
    print(f"\n  Active mesh vertices by region (OAR not excluded):")
    for region in sorted(active_region):
        print(f"    {region:<12}: {active_region[region]}")
    print(f"  Active mesh vertices by organ:")
    for organ in sorted(active_organ):
        print(f"    {organ:<30}: {active_organ[organ]}")

    # ------------------------------------------------------------------
    # 8. Recompute dose with optimised mesh dwells
    # ------------------------------------------------------------------

    print(f"\n{'='*70}")
    print(f"TG-43 DOSE CALCULATION (optimised mesh dwells, on {modality} grid)")
    print(f"{'='*70}")

    opt_dose_cgy = dose_contribution(
        dwell_pos=mesh_positions,
        norm_dwell_dir=mesh_directions,
        dwell_times=optimized_dwell_times,
        volume=volume,
        spacing=tuple(vol_spacing),
        origin=tuple(vol_origin),
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
    # 9. Simultaneous-loss evaluation (optimised plan)
    # ------------------------------------------------------------------

    L_opt, breakdown_opt = evaluate_simultaneous_loss(
        opt_dose_gy, optimized_dwell_times, masks, ctv_names, oar_names,
        rx_params,
    )
    print_simultaneous_loss(L_opt, breakdown_opt, rx_params)

    # ------------------------------------------------------------------
    # 10. Clinical plan evaluation (DVH criteria vs prescription)
    # ------------------------------------------------------------------

    eval_report = evaluate_plan(
        opt_dose_gy, masks, ctv_names, ptv_names, oar_names,
        spacing_mm=vol_spacing,
        D_p=rx_params["D_p"],
    )
    print_evaluation_report(eval_report, D_p=rx_params["D_p"])

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
    print(f"  {modality} grid          : {vol_shape}  spacing {tuple(vol_spacing)} mm")
    print(f"  CTV mask(s)      : {ctv_names}")
    print(f"  PTV mask(s)      : {ptv_names}")
    print(f"  OAR mask(s)      : {oar_names}")
    print(f"  Mesh candidates  : {n_mesh}  "
          f"(CTV/PTV {sum(1 for lab in mesh_labels if lab['in_ctv'] or lab['in_ptv'])}, "
          f"OAR {sum(1 for lab in mesh_labels if lab['in_oar'])}, "
          f"outside {sum(1 for lab in mesh_labels if lab['region'] == 'OUTSIDE')})")
    print(f"  Active after IPSA: {int(np.sum(optimized_dwell_times > 0))}")
    print(f"  Source strength  : {S_k:.4f} U")
    print(f"\n  Initial (uniform mesh) : max {dose_gy.max():.4f} Gy   L = {L:.2f}  "
          f"(J_dose {breakdown['J_dose']:.2f}, P_l2 {breakdown['P_l2']:.2f})")
    print(f"  Optimised             : max {opt_dose_gy.max():.4f} Gy   L = {L_opt:.2f}  "
          f"(J_dose {breakdown_opt['J_dose']:.2f}, P_l2 {breakdown_opt['P_l2']:.2f})")
    print(f"  Loss reduction        : {(1 - L_opt / L) * 100:.1f}%")
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