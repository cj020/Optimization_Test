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
    EXTENDED_CTV_NAME,
    EXTENDED_PTV_NAME,
    assemble_mesh_dwells,
    count_mesh_labels,
    extended_structure_mask,
    filter_mesh_dwells,
    label_mesh_points,
    mesh_channel_in_search_space,
    points_in_mask,
    structure_search_bounds,
)
from dose_contribution import dose_contribution
from plan_evaluation import evaluate_plan, print_evaluation_report
from simultaneous_loss import evaluate_simultaneous_loss, print_simultaneous_loss
from ipsa import run_ipsa
from cost_function import copy_prescription_params

Length_source = 3.5  # active length of the source in mm
Lambda = 1.113       # Dose-rate constant in cGy/(h·U), where U = µGy·m²/h = cGy·cm²/h

# Candidate dwell options for optimisation
CANDIDATES_ALL = "all"                    # every mesh vertex in the search area
CANDIDATES_EXTENDED_PTV = "extended_ptv"  # only vertices inside the Extended PTV
CANDIDATES_EXTENDED_CTV = "extended_ctv"  # only vertices inside the Extended CTV
EXTENDED_PTV_MARGIN_MM = 5.0              # Extended PTV = PTV + this margin beyond its boundary
EXTENDED_CTV_MARGIN_MM = 5.0              # Extended CTV = CTV + this margin beyond its boundary

# Image / dose-calculation grid. 1 mm is the current default (US only).
GRID_1MM = 1.0
GRID_3MM = 3.0


class PipelineError(RuntimeError):
    """A case cannot be planned (missing RTSTRUCT / RTPLAN, empty mesh, ...)."""


def _drop_empty_other(masks, structures):
    """Discard empty masks of unclassified ('OTHER') structures, e.g. point markers."""
    classes = {s["name"]: s.get("classification") for s in structures}
    return {name: m for name, m in masks.items()
            if classes.get(name) != "OTHER" or m.any()}


def _print_dose_table(dose_gy, structures, masks):
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


def resample_field_to_grid(field, src_origin, src_spacing, dst_origin, dst_spacing, dst_shape):
    """Interpolate a 3-D field from one voxel grid onto another (world mm).

    Used to overlay a coarse dose grid on the native image for display.
    """
    from scipy.ndimage import map_coordinates

    field = np.asarray(field, dtype=np.float64)
    src_origin = np.asarray(src_origin, dtype=np.float64).reshape(3)
    src_spacing = np.asarray(src_spacing, dtype=np.float64).reshape(3)
    dst_origin = np.asarray(dst_origin, dtype=np.float64).reshape(3)
    dst_spacing = np.asarray(dst_spacing, dtype=np.float64).reshape(3)
    dst_shape = tuple(int(n) for n in dst_shape)
    if (field.shape == dst_shape
            and np.allclose(src_origin, dst_origin)
            and np.allclose(src_spacing, dst_spacing)):
        return field.copy()

    nz, ny, nx = dst_shape
    z = dst_origin[2] + np.arange(nz) * dst_spacing[2]
    y = dst_origin[1] + np.arange(ny) * dst_spacing[1]
    x = dst_origin[0] + np.arange(nx) * dst_spacing[0]
    iz = (z - src_origin[2]) / src_spacing[2]
    iy = (y - src_origin[1]) / src_spacing[1]
    ix = (x - src_origin[0]) / src_spacing[0]
    zz, yy, xx = np.meshgrid(iz, iy, ix, indexing="ij")
    coords = np.vstack([zz.ravel(), yy.ravel(), xx.ravel()])
    out = map_coordinates(field, coords, order=1, mode="nearest")
    return out.reshape(dst_shape)


def make_calc_grid(case, grid_mm=GRID_1MM):
    """Resample the native image and re-rasterize structures for dose / IPSA / criteria.

    Display keeps ``case['volume']`` at native spacing. This grid is used only
    for TG-43, optimisation, and the clinical evaluation.
    """
    volume, spacing, origin = resample_volume_to_spacing(
        case["volume"], case["spacing"], case["origin"], target_mm=grid_mm,
    )
    spacing = np.asarray(spacing, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    print(f"\n  Calculation grid ({grid_mm:g} mm target) for {case['modality']}:")
    print(f"    Native  {case['volume'].shape} @ {tuple(float(v) for v in case['spacing'])} mm")
    print(f"    Calc    {volume.shape} @ {tuple(float(v) for v in spacing)} mm")
    print(f"    Rasterizing structure masks on the calculation grid {volume.shape} ...")
    masks = _drop_empty_other(build_structure_masks(
        case["structures"],
        grid_origin=origin,
        grid_spacing=spacing,
        grid_shape=volume.shape,
    ), case["structures"])
    voxel_vol_mm3 = float(np.prod(spacing))
    print(f"    {'Structure':<30} {'Class':<6} {'Voxels':>10} {'Volume (cm3)':>14}")
    print(f"    {'-'*30} {'-'*6} {'-'*10} {'-'*14}")
    for s in case["structures"]:
        if s["name"] not in masks:
            continue
        n_vox = int(masks[s["name"]].sum())
        print(f"    {s['name']:<30} {s['classification']:<6} {n_vox:>10} "
              f"{n_vox * voxel_vol_mm3 / 1000.0:>14.2f}")
    return volume, spacing, origin, masks


def _print_dose_summary(dose_gy):
    print(f"\n  Dose grid shape : {dose_gy.shape}")
    print(f"  Max dose        : {dose_gy.max():.4f} Gy")
    nonzero = dose_gy[dose_gy > 0]
    if len(nonzero) > 0:
        print(f"  Mean (>0)       : {nonzero.mean():.4f} Gy")
        print(f"  Non-zero voxels : {len(nonzero)}")


def load_case(folder):
    """Load DICOM on the native image grid for display.

    Dose calculation, IPSA, and clinical criteria use a separate 1 mm or
    3 mm grid created later in ``build_mesh`` / ``optimize_case``.

    Returns
    -------
    case : dict
        Native volume, structure masks, RTPLAN dwells, and related metadata.
    """
    # Patient folder may contain two RTDOSE objects (native TPS + resampled on CT grid).
    # This pipeline compares TG-43 calculated dose only against the native RTDOSE grid.

    # ------------------------------------------------------------------
    # 1. Load all DICOM data (native spacing — used for display)
    # ------------------------------------------------------------------

    volume, spacing, origin, direction, rtstruct, rtplan, rtdose, dose_ds, modality = load_dicom(folder)

    vol_shape = volume.shape          # (nz, ny, nx)
    vol_spacing = np.array(spacing)   # (dx, dy, dz) mm
    vol_origin = np.array(origin)     # (x0, y0, z0) mm

    print(f"\n{'='*70}")
    print(f"{modality} VOLUME (native, for display)")
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
        raise PipelineError("No RTSTRUCT found in the selected folder.")

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
    # All structures are rasterized (not only CTV/PTV/OAR) so any of them can
    # be chosen as an optimization objective in the settings dialog.
    masks = _drop_empty_other(build_structure_masks(
        structures,
        grid_origin=vol_origin,
        grid_spacing=vol_spacing,
        grid_shape=vol_shape,
    ), structures)

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

    # Extended PTV (PTV + margin), used to restrict candidate dwells
    ext_ptv_mask = extended_structure_mask(
        masks, ptv_names, vol_spacing, margin_mm=EXTENDED_PTV_MARGIN_MM,
    )
    if ext_ptv_mask is None:
        print(f"\n  {EXTENDED_PTV_NAME}: no PTV voxels — not available.")
    else:
        print(f"\n  {EXTENDED_PTV_NAME} (PTV + {EXTENDED_PTV_MARGIN_MM:g} mm): "
              f"{int(ext_ptv_mask.sum())} voxels, "
              f"{ext_ptv_mask.sum() * voxel_vol_mm3 / 1000.0:.2f} cm3")

    # Extended CTV (CTV + margin), used to restrict candidate dwells
    ext_ctv_mask = extended_structure_mask(
        masks, ctv_names, vol_spacing, margin_mm=EXTENDED_CTV_MARGIN_MM,
    )
    if ext_ctv_mask is None:
        print(f"  {EXTENDED_CTV_NAME}: no CTV voxels — not available.")
    else:
        print(f"  {EXTENDED_CTV_NAME} (CTV + {EXTENDED_CTV_MARGIN_MM:g} mm): "
              f"{int(ext_ctv_mask.sum())} voxels, "
              f"{ext_ctv_mask.sum() * voxel_vol_mm3 / 1000.0:.2f} cm3")

    # ------------------------------------------------------------------
    # 3. Extract dwell information from RTPLAN
    # ------------------------------------------------------------------

    print(f"\n{'='*70}")
    print("RTPLAN — DWELL POSITIONS")
    print(f"{'='*70}")

    if rtplan is None:
        raise PipelineError("No RTPLAN found in the selected folder.")

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
    channels = np.array([d[1] for d in dwells])              # (N,)   channel IDs

    print(f"\n  Total dwell positions   : {count}")
    print(f"  Non-zero dwell times    : {np.sum(dwell_times > 0)}")
    print(f"  Total irradiation time  : {np.sum(dwell_times):.2f} s")
    print(f"  Unique channels         : {len(np.unique(channels))}")

    print(f"\n  Dwell X range: [{dwell_positions[:,0].min():.2f}, {dwell_positions[:,0].max():.2f}] mm")
    print(f"  Dwell Y range: [{dwell_positions[:,1].min():.2f}, {dwell_positions[:,1].max():.2f}] mm")
    print(f"  Dwell Z range: [{dwell_positions[:,2].min():.2f}, {dwell_positions[:,2].max():.2f}] mm")

    return {
        "folder": folder,
        "modality": modality,
        "volume": volume,
        "spacing": vol_spacing,
        "origin": vol_origin,
        "structures": structures,
        "structure_classes": structure_classes,
        "search_names": search_names,
        "ctv_names": ctv_names,
        "ptv_names": ptv_names,
        "oar_names": oar_names,
        "masks": masks,
        "extended_ptv_mask": ext_ptv_mask,
        "extended_ptv_margin_mm": EXTENDED_PTV_MARGIN_MM,
        "extended_ctv_mask": ext_ctv_mask,
        "extended_ctv_margin_mm": EXTENDED_CTV_MARGIN_MM,
        "rtdose": rtdose,
        "rtplan": rtplan,
        "S_k": S_k,
        "plan_dwell_positions": dwell_positions,
        "plan_dwell_times": dwell_times,
        "plan_dwell_channels": channels,
    }


def build_candidate_mesh(case, candidate_region=CANDIDATES_ALL,
                         ptv_margin_mm=EXTENDED_PTV_MARGIN_MM,
                         ctv_margin_mm=EXTENDED_CTV_MARGIN_MM,
                         **_ignored):
    """Step 4 only: candidate dwell positions on the needle meshes (no dose).

    Fast; used both by ``build_mesh`` and to preview the candidates in 3-D
    before running the optimization. ``candidate_region`` / margins are as
    in ``build_mesh``; other keywords (e.g. ``grid_mm``) are ignored.

    Returns
    -------
    candidates : dict
        ``positions`` (N, 3), ``directions`` (N, 3), ``channel_ids`` (N,),
        ``local_index`` (N,), ``labels`` (list of N dicts), the
        ``extended_ptv_mask`` / ``extended_ctv_mask`` used, and the settings.
    """
    vol_spacing = case["spacing"]
    vol_origin = case["origin"]
    masks = case["masks"]
    structure_classes = case["structure_classes"]
    search_names = case["search_names"]
    rtplan = case["rtplan"]

    # ------------------------------------------------------------------
    # 4. Mesh each channel along the needle line inside the PTV+OAR box
    # ------------------------------------------------------------------

    print(f"\n{'='*70}")
    print("NEEDLE MESHING (channel line ∩ CTV/PTV/OAR bounding box, 1 mm)")
    print(f"{'='*70}")

    plan_channels = extract_channel_geometry(rtplan)
    if not plan_channels:
        raise PipelineError("No channel start/end points found in RTPLAN.")

    box_min, box_max = structure_search_bounds(
        masks, vol_origin, vol_spacing, names=search_names,
    )
    if box_min is None:
        raise PipelineError("Search space is empty (no CTV/PTV/OAR voxels).")

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

    # Extended PTV / Extended CTV labels on every vertex, then optionally restrict.
    known = (CANDIDATES_ALL, CANDIDATES_EXTENDED_PTV, CANDIDATES_EXTENDED_CTV)
    if candidate_region not in known:
        raise PipelineError(f"Unknown candidate region: {candidate_region!r}")
    if ptv_margin_mm == case.get("extended_ptv_margin_mm"):
        ext_ptv_mask = case.get("extended_ptv_mask")
    else:
        ext_ptv_mask = extended_structure_mask(
            masks, case["ptv_names"], vol_spacing, margin_mm=ptv_margin_mm,
        )
    if ctv_margin_mm == case.get("extended_ctv_margin_mm"):
        ext_ctv_mask = case.get("extended_ctv_mask")
    else:
        ext_ctv_mask = extended_structure_mask(
            masks, case["ctv_names"], vol_spacing, margin_mm=ctv_margin_mm,
        )
    in_ext_ptv = points_in_mask(mesh_dwells["positions"], ext_ptv_mask, vol_origin, vol_spacing)
    in_ext_ctv = points_in_mask(mesh_dwells["positions"], ext_ctv_mask, vol_origin, vol_spacing)
    for lab, in_p, in_c in zip(mesh_dwells["labels"], in_ext_ptv, in_ext_ctv):
        lab["in_extended_ptv"] = bool(in_p)
        lab["in_extended_ctv"] = bool(in_c)
    print(f"\n  Vertices inside {EXTENDED_PTV_NAME} (PTV + {ptv_margin_mm:g} mm): "
          f"{int(in_ext_ptv.sum())} / {len(in_ext_ptv)}")
    print(f"  Vertices inside {EXTENDED_CTV_NAME} (CTV + {ctv_margin_mm:g} mm): "
          f"{int(in_ext_ctv.sum())} / {len(in_ext_ctv)}")

    if candidate_region == CANDIDATES_EXTENDED_PTV:
        if ext_ptv_mask is None:
            raise PipelineError(
                f"No PTV structure found; cannot restrict dwells to the {EXTENDED_PTV_NAME}."
            )
        mesh_dwells = filter_mesh_dwells(mesh_dwells, in_ext_ptv)
        print(f"  Candidate dwells restricted to the {EXTENDED_PTV_NAME}: "
              f"{len(mesh_dwells['positions'])} vertices kept")
    elif candidate_region == CANDIDATES_EXTENDED_CTV:
        if ext_ctv_mask is None:
            raise PipelineError(
                f"No CTV structure found; cannot restrict dwells to the {EXTENDED_CTV_NAME}."
            )
        mesh_dwells = filter_mesh_dwells(mesh_dwells, in_ext_ctv)
        print(f"  Candidate dwells restricted to the {EXTENDED_CTV_NAME}: "
              f"{len(mesh_dwells['positions'])} vertices kept")
    else:
        print("  Candidate dwells: all mesh vertices in the search area")

    mesh_positions = mesh_dwells["positions"]
    mesh_directions = mesh_dwells["directions"]
    mesh_channels = mesh_dwells["channel_ids"]
    mesh_local_index = mesh_dwells["local_index"]
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
        raise PipelineError("No mesh points to use as candidate dwells.")

    return {
        "positions": mesh_positions,
        "directions": mesh_directions,
        "channel_ids": mesh_channels,
        "local_index": mesh_local_index,
        "labels": mesh_labels,
        "extended_ptv_mask": ext_ptv_mask,
        "extended_ctv_mask": ext_ctv_mask,
        "candidate_region": candidate_region,
        "ptv_margin_mm": ptv_margin_mm,
        "ctv_margin_mm": ctv_margin_mm,
    }


def build_mesh(case, candidate_region=CANDIDATES_ALL,
               ptv_margin_mm=EXTENDED_PTV_MARGIN_MM,
               ctv_margin_mm=EXTENDED_CTV_MARGIN_MM,
               grid_mm=GRID_1MM):
    """Steps 4-5: mesh the RTPLAN channels and compute the uniform-time TG-43 dose.

    Parameters
    ----------
    candidate_region : str
        ``CANDIDATES_ALL`` keeps every mesh vertex in the search area;
        ``CANDIDATES_EXTENDED_PTV`` keeps only vertices inside the
        Extended PTV (PTV plus ``ptv_margin_mm`` beyond its boundary);
        ``CANDIDATES_EXTENDED_CTV`` keeps only vertices inside the
        Extended CTV (CTV plus ``ctv_margin_mm`` beyond its boundary).
    grid_mm : float
        Target voxel size (mm) for TG-43 / IPSA / clinical criteria
        (``GRID_1MM`` or ``GRID_3MM``). The displayed image stays native.

    Returns
    -------
    mesh : dict
        Candidate dwells on the needle meshes, their uniform start times,
        and the corresponding initial dose.
    """
    structures = case["structures"]
    S_k = case["S_k"]
    dwell_times = case["plan_dwell_times"]

    cand = build_candidate_mesh(case, candidate_region, ptv_margin_mm, ctv_margin_mm)
    mesh_positions = cand["positions"]
    mesh_directions = cand["directions"]
    mesh_channels = cand["channel_ids"]
    mesh_local_index = cand["local_index"]
    mesh_labels = cand["labels"]
    ext_ptv_mask = cand["extended_ptv_mask"]
    ext_ctv_mask = cand["extended_ctv_mask"]
    n_mesh = len(mesh_positions)

    calc_volume, calc_spacing, calc_origin, calc_masks = make_calc_grid(case, grid_mm)

    # ------------------------------------------------------------------
    # 5. TG-43 dose calculation (uniform time on every mesh vertex)
    # ------------------------------------------------------------------

    print(f"\n{'='*70}")
    print(f"TG-43 DOSE CALCULATION (uniform time on all mesh vertices, "
          f"{grid_mm:g} mm calculation grid)")
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
        volume=calc_volume,
        spacing=tuple(calc_spacing),
        origin=tuple(calc_origin),
        L=Length_source,
        S_k=S_k,
        Lambda=Lambda,
    )

    dose_gy = total_dose_cgy / 100.0   # cGy -> Gy

    _print_dose_summary(dose_gy)
    _print_dose_table(dose_gy, structures, calc_masks)

    return {
        "orig_mean": orig_mean,
        "orig_max": orig_max,
        "orig_total": orig_total,
        "mesh_positions": mesh_positions,
        "mesh_directions": mesh_directions,
        "mesh_channels": mesh_channels,
        "mesh_local_index": mesh_local_index,
        "mesh_labels": mesh_labels,
        "initial_dwell_times": initial_dwell_times,
        "initial_dose_gy": dose_gy,
        "candidate_region": candidate_region,
        "ptv_margin_mm": ptv_margin_mm,
        "ctv_margin_mm": ctv_margin_mm,
        "extended_ptv_mask": ext_ptv_mask,
        "extended_ctv_mask": ext_ctv_mask,
        "grid_mm": float(grid_mm),
        "calc_volume": calc_volume,
        "calc_spacing": calc_spacing,
        "calc_origin": calc_origin,
        "calc_masks": calc_masks,
    }


def optimize_case(case, rx_params=None, ipsa_params=None, mesh_params=None):
    """Steps 4-12: needle mesh, initial loss, IPSA, optimised dose, evaluation.

    ``mesh_params`` may set ``candidate_region`` (``CANDIDATES_ALL``,
    ``CANDIDATES_EXTENDED_PTV``, or ``CANDIDATES_EXTENDED_CTV``),
    ``ptv_margin_mm`` / ``ctv_margin_mm``, and ``grid_mm`` (1 or 3)
    for ``build_mesh``. Dose, IPSA, and clinical criteria use that
    calculation grid; the displayed image stays at native spacing.

    Returns
    -------
    result : dict
        The candidate mesh, optimised dwell times on it, optimised dose,
        losses, and the clinical evaluation report.
    """
    rx_params = copy_prescription_params(rx_params)
    mesh = build_mesh(case, **(mesh_params or {}))

    volume = mesh["calc_volume"]
    vol_spacing = mesh["calc_spacing"]
    vol_origin = mesh["calc_origin"]
    masks = mesh["calc_masks"]
    structures = case["structures"]
    ctv_names = case["ctv_names"]
    ptv_names = case["ptv_names"]
    oar_names = case["oar_names"]
    modality = case["modality"]
    S_k = case["S_k"]
    mesh_positions = mesh["mesh_positions"]
    mesh_directions = mesh["mesh_directions"]
    mesh_channels = mesh["mesh_channels"]
    mesh_labels = mesh["mesh_labels"]
    initial_dwell_times = mesh["initial_dwell_times"]
    dose_gy = mesh["initial_dose_gy"]
    n_mesh = len(mesh_positions)

    # ------------------------------------------------------------------
    # 6. Initial evaluation with the cost-function parameters
    # ------------------------------------------------------------------

    L, breakdown = evaluate_simultaneous_loss(
        dose_gy, initial_dwell_times, masks, ctv_names, oar_names, rx_params,
    )
    print_simultaneous_loss(L, breakdown, rx_params)

    # ------------------------------------------------------------------
    # 7. IPSA: dwell time + mesh-position optimisation
    # ------------------------------------------------------------------

    ip = {
        # Bounds / step size from the original plan, not the small
        # per-vertex uniform start, so activity can concentrate.
        "t_max": 3.0 * mesh["orig_max"],
        "perturbation_std": 0.3 * mesh["orig_mean"],
    }
    if ipsa_params:
        ip.update(ipsa_params)

    optimized_dwell_times, ipsa_history = run_ipsa(
        mesh_positions, mesh_directions, initial_dwell_times,
        initial_dose_gy=dose_gy,
        masks=masks, ctv_names=ctv_names, oar_names=oar_names,
        volume=volume, spacing=tuple(vol_spacing), origin=tuple(vol_origin),
        L=Length_source, S_k=S_k, Lambda=Lambda,
        rx_params=rx_params,
        channel_ids=mesh_channels,
        ipsa_params=ip,
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
    print(f"TG-43 DOSE CALCULATION (optimised mesh dwells, "
          f"{mesh['grid_mm']:g} mm calculation grid)")
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

    _print_dose_summary(opt_dose_gy)
    _print_dose_table(opt_dose_gy, structures, masks)

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
        protocol=rx_params.get("protocol"),
    )
    print_evaluation_report(eval_report, D_p=rx_params["D_p"],
                            protocol=rx_params.get("protocol"))

    opt_dose_gy_display = resample_field_to_grid(
        opt_dose_gy, vol_origin, vol_spacing,
        case["origin"], case["spacing"], case["volume"].shape,
    )

    # ------------------------------------------------------------------
    # 11. RTDOSE (loaded for future reference, not used in optimization)
    # ------------------------------------------------------------------

    rtdose = case["rtdose"]
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
    n_fail = sum(
        1 for role in eval_report.values()
        for c in role.get("checks", [])
        if c["status"] == "FAIL"
    )
    n_checks = sum(len(role.get("checks", [])) for role in eval_report.values())

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Native image     : {case['volume'].shape}  "
          f"spacing {tuple(float(v) for v in case['spacing'])} mm")
    print(f"  {modality} calc grid     : {volume.shape}  spacing {tuple(vol_spacing)} mm"
          f"  (target {mesh['grid_mm']:g} mm)")
    print(f"  CTV mask(s)      : {ctv_names}")
    print(f"  PTV mask(s)      : {ptv_names}")
    print(f"  OAR mask(s)      : {oar_names}")
    if mesh["candidate_region"] == CANDIDATES_EXTENDED_PTV:
        print(f"  Candidate region : {EXTENDED_PTV_NAME} (PTV + {mesh['ptv_margin_mm']:g} mm)")
    elif mesh["candidate_region"] == CANDIDATES_EXTENDED_CTV:
        print(f"  Candidate region : {EXTENDED_CTV_NAME} (CTV + {mesh['ctv_margin_mm']:g} mm)")
    else:
        print(f"  Candidate region : all mesh vertices in the search area")
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
    reduction = (1 - L_opt / L) * 100 if abs(L) > 1e-12 else 0.0
    print(f"  Loss reduction        : {reduction:.1f}%")
    print(f"  Clinical criteria      : {n_checks - n_fail}/{n_checks} met "
          f"({n_fail} failed)")
    print()

    return {
        **mesh,
        "rx_params": rx_params,
        "optimized_dwell_times": optimized_dwell_times,
        "opt_dose_gy": opt_dose_gy,
        "opt_dose_gy_display": opt_dose_gy_display,
        "ipsa_history": ipsa_history,
        "L_initial": L,
        "L_opt": L_opt,
        "breakdown_initial": breakdown,
        "breakdown_opt": breakdown_opt,
        "loss_reduction_pct": reduction,
        "eval_report": eval_report,
        "n_checks": n_checks,
        "n_fail": n_fail,
    }
