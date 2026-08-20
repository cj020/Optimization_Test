import os
import pydicom
import numpy as np
from matplotlib.path import Path

# Filenames suggesting a derived dose grid (e.g. resampled to CT). Not used for TG-43 vs TPS comparison.
_DERIVED_RTDOSE_NAME_KEYWORDS = ("resampled", "resample", "dose_resampled")

# folder = r"C:\Users\jichen\Downloads\T00060\T00060"

def is_derived_rtdose_path(path):
    """True if the path looks like a derived/resampled RTDOSE, not the native TPS export."""
    name = os.path.basename(path).lower()
    return any(k in name for k in _DERIVED_RTDOSE_NAME_KEYWORDS)


def select_native_rtdose(rtdose_paths, brachy_plan_sop_uid=None, rtplan_path=None):
    """
    Pick the native TPS RTDOSE for reference comparison (TG-43 vs RTDOSE).

    Skips derived grids (e.g. DOSE_resampled.dcm). Does not compare two RTDOSE files
    against each other — only one reference volume is returned.
    """
    if not rtdose_paths:
        return None, []

    native_paths = [p for p in rtdose_paths if not is_derived_rtdose_path(p)]
    skipped = [p for p in rtdose_paths if is_derived_rtdose_path(p)]

    for p in skipped:
        print(f"  Skipping derived RTDOSE (not used for comparison): {os.path.basename(p)}")

    if not native_paths:
        print("  WARNING: No native RTDOSE after excluding derived files; using first available.")
        native_paths = list(rtdose_paths)

    # Prefer native file whose ReferencedRTPlanSequence matches the brachy RTPLAN
    if brachy_plan_sop_uid:
        for p in native_paths:
            try:
                ds = pydicom.dcmread(p, stop_before_pixels=True)
                ref_seq = getattr(ds, "ReferencedRTPlanSequence", None)
                if ref_seq:
                    ref_uid = getattr(ref_seq[0], "ReferencedSOPInstanceUID", None)
                    if ref_uid == brachy_plan_sop_uid:
                        print(f"  Selected native RTDOSE (plan UID match): {os.path.basename(p)}")
                        return p, skipped
            except Exception:
                continue

    if len(native_paths) == 1:
        print(f"  Selected native RTDOSE: {os.path.basename(native_paths[0])}")
        return native_paths[0], skipped

    # Multiple native candidates: spatial match to brachy dwell centroid
    if rtplan_path:
        print(f"  {len(native_paths)} native RTDOSE file(s); trying spatial match...")
        try:
            plan_ds = pydicom.dcmread(rtplan_path, stop_before_pixels=True)
            dwell_pts = []
            for app in plan_ds.ApplicationSetupSequence:
                for ch in app.ChannelSequence:
                    for cp in ch.BrachyControlPointSequence:
                        if hasattr(cp, "ControlPoint3DPosition"):
                            dwell_pts.append(np.array(cp.ControlPoint3DPosition, dtype=float))
            if dwell_pts:
                centroid = np.mean(dwell_pts, axis=0)
                best_dist = float("inf")
                best_path = None
                for p in native_paths:
                    try:
                        ds = pydicom.dcmread(p)
                        dose_arr = ds.pixel_array * ds.DoseGridScaling
                        origin = np.array(ds.ImagePositionPatient, dtype=float)
                        dy, dx = [float(v) for v in ds.PixelSpacing]
                        z_off = np.array(ds.GridFrameOffsetVector, dtype=float)
                        mi = np.unravel_index(np.argmax(dose_arr), dose_arr.shape)
                        max_pos = np.array([
                            origin[0] + mi[2] * dx,
                            origin[1] + mi[1] * dy,
                            origin[2] + z_off[mi[0]],
                        ])
                        d = np.linalg.norm(max_pos - centroid)
                        print(f"    {os.path.basename(p)}: max-dose dist to dwell centroid = {d:.1f} mm")
                        if d < best_dist:
                            best_dist = d
                            best_path = p
                    except Exception:
                        continue
                if best_path:
                    print(f"  Selected native RTDOSE (spatial): {os.path.basename(best_path)}")
                    return best_path, skipped
        except Exception as e:
            print(f"  Spatial matching failed: {e}")

    # Prefer standard RTDOSE*.dcm naming over other natives
    for p in native_paths:
        if os.path.basename(p).upper().startswith("RTDOSE"):
            print(f"  Selected native RTDOSE (filename): {os.path.basename(p)}")
            return p, skipped

    print(f"  WARNING: Using first native RTDOSE: {os.path.basename(native_paths[0])}")
    return native_paths[0], skipped


# ============================================================================
# Image volume construction — generic voxel grid, then CT / US geometry
# ============================================================================

def _slice_sort_key(ds):
    """Scalar used to order slices along the stack (patient z, then instance)."""
    iop = getattr(ds, "ImageOrientationPatient", None)
    ipp = getattr(ds, "ImagePositionPatient", None)
    if iop is not None and ipp is not None:
        iop = np.array(iop, dtype=float)
        ipp = np.array(ipp, dtype=float)
        normal = np.cross(iop[:3], iop[3:6])
        nrm = np.linalg.norm(normal)
        if nrm > 0:
            return float(np.dot(ipp, normal / nrm))
    if ipp is not None:
        return float(np.array(ipp, dtype=float)[2])
    if hasattr(ds, "SliceLocation"):
        return float(ds.SliceLocation)
    return float(getattr(ds, "InstanceNumber", 0) or 0)


def _slice_pixel_array(ds):
    """Return a 2D (rows, cols) array from a single-frame image dataset."""
    arr = ds.pixel_array
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = np.mean(arr[..., :3], axis=-1)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D image slice, got shape {arr.shape}")
    return arr


def _largest_series(paths):
    """Keep the SeriesInstanceUID group with the most files."""
    groups = {}
    for p in paths:
        ds = pydicom.dcmread(p, stop_before_pixels=True)
        uid = str(getattr(ds, "SeriesInstanceUID", "unknown"))
        groups.setdefault(uid, []).append(p)
    return max(groups.values(), key=len)


def select_image_series(ct_files, us_files):
    """
    Choose the image series used to build the voxel grid.

    If both CT and US are present, the larger series is used.
    """
    if ct_files:
        ct_files = _largest_series(ct_files)
    if us_files:
        us_files = _largest_series(us_files)

    if ct_files and us_files:
        print(f"  Found both CT ({len(ct_files)} file(s)) and US ({len(us_files)} file(s)); "
              f"using the larger series.")
        if len(us_files) >= len(ct_files):
            return us_files, "US"
        return ct_files, "CT"
    if ct_files:
        return ct_files, "CT"
    if us_files:
        return us_files, "US"
    raise ValueError("No CT or Ultrasound image series found")


def build_voxel_volume(image_files):
    """
    Build a generic 3D voxel array from an image series (CT or Ultrasound).

    Pixel data is stacked only. Spacing, origin, and direction are not applied
    here — those are read afterwards from CT- or US-specific tags.

    Returns
    -------
    volume : np.ndarray
        Shape (nz, ny, nx).
    datasets : list[pydicom.Dataset]
        Slice datasets in the same order as `volume` (one entry for a
        multi-frame file).
    """
    if not image_files:
        raise ValueError("No image files to build volume")

    first = pydicom.dcmread(image_files[0], stop_before_pixels=True) # read the first DICOM file
    n_frames = int(getattr(first, "NumberOfFrames", 1) or 1) # distinguish between a collection of single-frame files and a single multi-frame DICOM file

    # Single multi-frame file (some US volumes)
    if len(image_files) == 1 and n_frames > 1:
        ds = pydicom.dcmread(image_files[0])
        arr = ds.pixel_array
        if arr.ndim == 4 and arr.shape[-1] in (3, 4):
            arr = np.mean(arr[..., :3], axis=-1)
        if arr.ndim != 3:
            raise ValueError(f"Expected multi-frame volume (nz, ny, nx), got shape {arr.shape}")
        print(f"  Stacked multi-frame volume: {arr.shape}  dtype={arr.dtype}")
        return arr, [ds]

    loaded = []
    for p in image_files:
        ds = pydicom.dcmread(p)
        arr = _slice_pixel_array(ds)
        loaded.append((_slice_sort_key(ds), ds, arr))

    loaded.sort(key=lambda t: t[0])
    shapes = {t[2].shape for t in loaded}
    if len(shapes) != 1:
        raise ValueError(f"Inconsistent slice shapes in image series: {shapes}")

    volume = np.stack([t[2] for t in loaded], axis=0)
    datasets = [t[1] for t in loaded]
    print(f"  Stacked {len(datasets)} slices into volume {volume.shape}  dtype={volume.dtype}")
    return volume, datasets


def _inplane_spacing_from_pixelspacing(ds):
    """DICOM PixelSpacing is (row spacing, column spacing) = (dy, dx) in mm."""
    dy, dx = [float(v) for v in ds.PixelSpacing]
    return dx, dy


def _inplane_spacing_from_us_regions(ds):
    """
    Ultrasound region calibration (0018,6011).

    PhysicalDeltaX / PhysicalDeltaY are typically in cm when
    PhysicalUnitsXDirection / YDirection == 3.
    """
    regions = getattr(ds, "SequenceOfUltrasoundRegions", None)
    if not regions:
        raise ValueError("Ultrasound image has neither PixelSpacing nor SequenceOfUltrasoundRegions")
    region = regions[0]
    dx = float(region.PhysicalDeltaX)
    dy = float(region.PhysicalDeltaY)
    ux = int(getattr(region, "PhysicalUnitsXDirection", 3) or 3)
    uy = int(getattr(region, "PhysicalUnitsYDirection", 3) or 3)
    # DICOM ultrasound units: 3 = cm
    if ux == 3:
        dx *= 10.0
    if uy == 3:
        dy *= 10.0
    return dx, dy


def _direction_from_orientation(ds):
    iop = getattr(ds, "ImageOrientationPatient", None)
    if iop is None:
        return (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    iop = np.array(iop, dtype=float)
    row_cos = iop[:3]
    col_cos = iop[3:6]
    slice_cos = np.cross(row_cos, col_cos)
    nrm = np.linalg.norm(slice_cos)
    if nrm > 0:
        slice_cos = slice_cos / nrm
    return tuple(float(v) for v in np.concatenate([row_cos, col_cos, slice_cos]))


def _origin_from_ipp(ds):
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is not None:
        return tuple(float(v) for v in ipp)
    return (0.0, 0.0, 0.0)


def _slice_spacing_mm(datasets, fallback_ds):
    """Through-plane spacing (mm) from consecutive ImagePositionPatient, else SliceThickness."""
    positions = []
    for ds in datasets:
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None:
            positions.append(np.array(ipp, dtype=float))
    if len(positions) >= 2:
        diffs = [np.linalg.norm(positions[i + 1] - positions[i]) for i in range(len(positions) - 1)]
        dz = float(np.median(diffs))
        if dz > 0:
            return dz
    for attr in ("SpacingBetweenSlices", "SliceThickness"):
        val = getattr(fallback_ds, attr, None)
        if val is not None:
            dz = abs(float(val))
            if dz > 0:
                return dz
    return 1.0


def read_ct_geometry(datasets):
    """Read spacing (dx, dy, dz), origin (x, y, z), and direction from CT headers."""
    first = datasets[0]
    if not hasattr(first, "PixelSpacing") or first.PixelSpacing is None:
        raise ValueError("CT series is missing PixelSpacing")
    dx, dy = _inplane_spacing_from_pixelspacing(first)
    dz = _slice_spacing_mm(datasets, first)
    origin = _origin_from_ipp(first)
    if origin == (0.0, 0.0, 0.0) and not hasattr(first, "ImagePositionPatient"):
        print("  WARNING: CT series has no ImagePositionPatient; origin set to (0, 0, 0)")
    direction = _direction_from_orientation(first)
    return (dx, dy, dz), origin, direction


def read_us_geometry(datasets):
    """Read spacing (dx, dy, dz), origin (x, y, z), and direction from Ultrasound headers."""
    first = datasets[0]
    if hasattr(first, "PixelSpacing") and first.PixelSpacing is not None:
        dx, dy = _inplane_spacing_from_pixelspacing(first)
    else:
        dx, dy = _inplane_spacing_from_us_regions(first)
        print(f"  US in-plane spacing from SequenceOfUltrasoundRegions: dx={dx:.4f}, dy={dy:.4f} mm")
    dz = _slice_spacing_mm(datasets, first)
    origin = _origin_from_ipp(first)
    if origin == (0.0, 0.0, 0.0) and not hasattr(first, "ImagePositionPatient"):
        print("  WARNING: US series has no ImagePositionPatient; origin set to (0, 0, 0)")
    direction = _direction_from_orientation(first)
    return (dx, dy, dz), origin, direction


def apply_ct_rescale(volume, datasets):
    """Convert stored CT pixel values to HU using RescaleSlope / RescaleIntercept."""
    first = datasets[0]
    slope = float(getattr(first, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(first, "RescaleIntercept", 0.0) or 0.0)
    if slope == 1.0 and intercept == 0.0:
        return volume
    print(f"  Applying CT rescale: HU = {slope} * stored + {intercept}")
    return volume.astype(np.float64) * slope + intercept


def load_dicom(folder, skip_keywords=None):
    """
    Load a brachytherapy study: image volume (CT or US) + RTSTRUCT + RTPLAN + RTDOSE.

    Voxel construction is modality-agnostic (stack slices into a 3D array).
    Spacing, origin, and direction are then read from CT or Ultrasound tags.

    Parameters
    ----------
    folder : str
        Root folder of patient data
    skip_keywords : list[str] or None
        Folder-path substrings to ignore (e.g. ["Contours"])

    Returns
    -------
    volume : np.ndarray
        Image volume (z, y, x)
    spacing : tuple
            (dx, dy, dz) in mm
    origin : tuple
            (x0, y0, z0) in mm
    direction : tuple
            9-element direction cosines
    rtstruct : pydicom.Dataset or None
    rtplan : pydicom.Dataset or None
    rtdose : np.ndarray or None
    dose_ds : pydicom.Dataset or None
    modality : str
        "CT" or "US"
    """
    if skip_keywords is None:
        skip_keywords = ["Contours"]

    ct_series_files = []
    us_series_files = []
    rtstruct_path = None
    rtplan_paths = []
    rtdose_paths = []

    print("\n Scanning DICOM files...\n")

    # =========================
    # 1. Scan recursively
    # =========================
    for root, dirs, files in os.walk(folder):
        if any(k in root for k in skip_keywords):
            continue

        for f in files:
            path = os.path.join(root, f)

            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True) # stop_before_pixels=True speeds up reading by not loading pixel data, which is not needed for modality checking.
            except Exception:
                continue

            modality = getattr(ds, "Modality", None) # getattr is used to safely access the Modality attribute, returning None if it doesn't exist instead of raising an error.

            # CT/US slices
            if modality == "CT":
                ct_series_files.append(path)

            elif modality == "US":
                us_series_files.append(path)

            # RTSTRUCT (only keep first found)
            elif modality == "RTSTRUCT" and rtstruct_path is None:
                rtstruct_path = path

            # RTPLAN
            elif modality == "RTPLAN":
                rtplan_paths.append(path)

            # RTDOSE
            elif modality == "RTDOSE":
                rtdose_paths.append(path)

            print(f"{str(modality):10} | {path}")
    
    # Select the brachytherapy RTPLAN (has ApplicationSetupSequence)
    rtplan_path = None
    brachy_plan_sop_uid = None
    for p in rtplan_paths:
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True)
            if hasattr(ds, "ApplicationSetupSequence"):
                rtplan_path = p
                brachy_plan_sop_uid = getattr(ds, "SOPInstanceUID", None)
                print(f"\n  Selected brachy RTPLAN: {p}")
                print(f"  SOP Instance UID: {brachy_plan_sop_uid}")
                break
        except Exception:
            continue
    if rtplan_path is None and rtplan_paths:
        rtplan_path = rtplan_paths[0]
        print(f"\n  No brachy RTPLAN found, using first: {rtplan_path}")

    # Reference RTDOSE: native TPS grid only (TG-43 comparison target; not vs resampled dose)
    print(f"\n  RTDOSE selection ({len(rtdose_paths)} file(s) in folder):")
    rtdose_path, _skipped_derived = select_native_rtdose(
        rtdose_paths,
        brachy_plan_sop_uid=brachy_plan_sop_uid,
        rtplan_path=rtplan_path,
    )
    
    print(f"\n  Summary: {len(ct_series_files)} CT, {len(us_series_files)} US, "
          f"{len(rtplan_paths)} RTPLAN(s), {len(rtdose_paths)} RTDOSE(s) found")
    if rtdose_path:
        print(f"  Reference for TG-43 comparison: {os.path.basename(rtdose_path)}")
    
    # =========================
    # 2. Generic voxel volume, then CT / US geometry
    # =========================
    image_files, image_modality = select_image_series(ct_series_files, us_series_files)
    print(f"\n Building {image_modality} voxel volume ({len(image_files)} file(s))...")

    volume, slice_datasets = build_voxel_volume(image_files)

    if image_modality == "CT":
        spacing, origin, direction = read_ct_geometry(slice_datasets)
        volume = apply_ct_rescale(volume, slice_datasets)
    elif image_modality == "US":
        spacing, origin, direction = read_us_geometry(slice_datasets)
    else:
        raise ValueError(f"Unsupported image modality: {image_modality}")

    print(f"{image_modality} shape:", volume.shape)
    print("Spacing:", spacing)
    print("Origin:", origin)
    print("Direction:", direction)

    # =========================
    # 3. Load RTSTRUCT
    # =========================
    rtstruct = None
    if rtstruct_path:
        print("\n Loading RTSTRUCT")
        rtstruct = pydicom.dcmread(rtstruct_path)

    # =========================
    # 4. Load RTPLAN
    # =========================
    rtplan = None
    if rtplan_path:
        print("\n Loading RTPLAN")
        rtplan = pydicom.dcmread(rtplan_path)

    # =========================
    # 5. Load RTDOSE
    # =========================
    rtdose = None
    dose_ds = None
    if rtdose_path:
        print("\n Loading RTDOSE")
        dose_ds = pydicom.dcmread(rtdose_path)
        rtdose = dose_ds.pixel_array * dose_ds.DoseGridScaling

        print("Dose shape:", rtdose.shape)  # (nz, ny, nx)
        
        # rtdose is a 3D array with dimensions (z, y, x) corresponding to the dose grid. The spacing and origin of the dose grid can be obtained from the DICOM tags as follows:
        dy, dx = dose_ds.PixelSpacing
        z_offsets = dose_ds.GridFrameOffsetVector
        dz = z_offsets[1] - z_offsets[0]
        dose_spacing = (dz, dy, dx)  # note the order (dz, dy, dx)
        
        print("Dose spacing:", dose_spacing)

        dose_origin = dose_ds.ImagePositionPatient # (x0, y0, z0), the physical (world/patient) coordinate of the first voxel in the dose grid

        print("Dose origin:", dose_origin)

    return volume, spacing, origin, direction, rtstruct, rtplan, rtdose, dose_ds, image_modality

def extract_dwell_points(rtplan): # the old extract_dwell_points function, without extracting the dwell time
    
    dwell_positions = []  # list to store dwell positions (x, y, z)
    channel_numbers = []  # list to store corresponding channel numbers
    channel_total_times = [] # list to store total time of each channel
    control_point_indices = []  # list to store corresponding control point indices
    control_point_times = [] # list to store corresponding control point times weighted
    count = 0 # counter for dwell positions

    print("\nDwell points with repeat:\n")

    if rtplan is None:
        print("No RTPLAN available")
        return dwell_positions, count, channel_numbers, channel_total_times, control_point_indices, control_point_times

    try:
        for app in rtplan.ApplicationSetupSequence:
            for channel in app.ChannelSequence: # A channel = catheter path.
                for cp in channel.BrachyControlPointSequence:

                    if hasattr(cp, "ControlPoint3DPosition"): # hasattr check to avoid missing attribute
                        pos = cp.ControlPoint3DPosition  # ControlPoint3DPosition returns: (x, y, z) in mm, in the patient coordinate system (world coordinates)
                        
                        dwell_positions.append(pos)
                        channel_numbers.append(channel.ChannelNumber)
                        control_point_indices.append(cp.ControlPointIndex)
                        channel_total_times.append(channel.ChannelTotalTime) 
                        
                        # time weight (if exists)
                        time = getattr(cp, "CumulativeTimeWeight", None)
                        control_point_times.append(time)
                        
                        print(f"Dwell {count}: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}, Channel={channel.ChannelNumber}, ChannelTotalTime={channel.ChannelTotalTime}, ControlPoint={cp.ControlPointIndex}, TimeWeight={time}")
                        count += 1

    except Exception as e:
        print("Error reading RTPLAN:", e)

        print(f"\nTotal dwell positions: {count}")
    return dwell_positions, count, channel_numbers, channel_total_times,control_point_indices, control_point_times

def extract_dwell_points_with_dwell_time_and_direction(rtplan, local_directions = False): # extract dwell positions along with their corresponding dwell times calculated from the cumulative time weights and channel total time, an the local direction.
    """
    Extract dwell positions with irradiation time, time weight, and direction.
    Prints grouped by channel index.
    """
    
    dwells = []
    count = 0

    print("\n" + "="*80)
    print("DWELL POSITION EXTRACTION")
    print("="*80)
    print("\nDwell points without repeat:\n")

    if rtplan is None:
        print("No RTPLAN available")
        return dwells, count

    try:
        for app in rtplan.ApplicationSetupSequence:

            for channel in app.ChannelSequence: # A channel = catheter path.

                cps = channel.BrachyControlPointSequence
                channel_num = channel.ChannelNumber
                channel_time = float(getattr(channel, "ChannelTotalTime", 1.0))

                # Final CumulativeTimeWeight for normalization.
                # ChannelTotalTime is the actual total treatment time; CWT values
                # must be divided by w_final so dwell times sum to ChannelTotalTime.
                w_final = float(getattr(cps[-1], "CumulativeTimeWeight", 1.0))
                if w_final <= 0:
                    w_final = 1.0

                
                # Compute global direction (first to last control point) for the entire channel, as a fallback if local direction is not calculated
                cp_first = cps[0]
                cp_last = cps[len(cps) - 1]
                vec_first_to_last = np.array(cp_last.ControlPoint3DPosition, dtype=float) - np.array(cp_first.ControlPoint3DPosition, dtype=float)
                norm_vec_first_to_last = np.linalg.norm(vec_first_to_last)
                    
                if norm_vec_first_to_last > 0:
                    norm_vec_direction = vec_first_to_last / norm_vec_first_to_last
                else:
                    norm_vec_direction = np.array([0.0, 0.0, 0.0])

                # Print channel header
                print(f"\n{'-'*80}")
                print(f"  Channel {channel_num}  |  ChannelTotalTime = {channel_time:.4f} s  |  CWT_final = {w_final:.4f}")                
                print(f"{'-'*80}")
                print(f"  {'#':<5} {'X (mm)':>9} {'Y (mm)':>9} {'Z (mm)':>9}"
                      f"  {'TimeWeight':>11} {'Irrad. Time (s)':>16}"
                      f"  {'Dir_x':>6} {'Dir_y':>6} {'Dir_z':>6}")
                print(f"  {'-'*5} {'-'*9} {'-'*9} {'-'*9}"
                      f"  {'-'*11} {'-'*16}"
                      f"  {'-'*6} {'-'*6} {'-'*6}")

                dwell_idx_in_channel = 0 # counter for dwell positions within the current channel
                
                for i in range(0, len(cps) - 1, 2):

                    cp_start = cps[i]
                    cp_end = cps[i + 1]

                    if not hasattr(cp_start, "ControlPoint3DPosition"):
                        continue

                    # --- position (world coordinates)
                    pos = np.array(cp_start.ControlPoint3DPosition, dtype=float) # (x, y, z) in mm, in the patient coordinate system (world coordinates)

                    # --- time weight difference
                    w1 = getattr(cp_start, "CumulativeTimeWeight", None)
                    w2 = getattr(cp_end, "CumulativeTimeWeight", None)

                    if w1 is None or w2 is None:
                        time_weight = None
                        dwell_time = None
                    else:
                        time_weight = (w2 - w1) / w_final
                        dwell_time = time_weight * channel_time
                 
                    # Direction calculation ---if local direction
                    if local_directions == True: # calculate local direction vector using neighboring control points, with special handling for first and last dwells

                        # first dwell
                        if i == 0:
                            p0 = np.array(cps[i].ControlPoint3DPosition, dtype=float)
                            p1 = np.array(cps[i + 2].ControlPoint3DPosition, dtype=float)
                            cp_local_direction_vec = p1 - p0

                        # last dwell
                        elif i >= len(cps) - 2:
                            p0 = np.array(cps[i - 2].ControlPoint3DPosition, dtype=float)
                            p1 = np.array(cps[i].ControlPoint3DPosition, dtype=float)
                            cp_local_direction_vec = p1 - p0

                        # interior dwell
                        else:
                            p0 = np.array(cps[i - 2].ControlPoint3DPosition, dtype=float)
                            p1 = np.array(cps[i + 2].ControlPoint3DPosition, dtype=float)
                            cp_local_direction_vec = p1 - p0

                        local_direction_vec_norm = np.linalg.norm(cp_local_direction_vec)

                        if local_direction_vec_norm > 0:
                            norm_cp_direction = cp_local_direction_vec / local_direction_vec_norm
                        else:
                            norm_cp_direction = np.array([0.0, 0.0, 0.0])
                    
                    else: # calculate direction vector by approximating the direction of the catheter
                        norm_cp_direction = norm_vec_direction

                    # --- store one true dwell
                    dwells.append([
                        count,
                        channel_num,
                        dwell_time,
                        pos, 
                        norm_cp_direction
                    ])

                    # Print dwell info
                    tw_str = f"{time_weight:.4f}" if time_weight is not None else "N/A"
                    dt_str = f"{dwell_time:.4f}" if dwell_time is not None else "N/A"

                    print(f"  {dwell_idx_in_channel:<5} {pos[0]:>9.2f} {pos[1]:>9.2f} {pos[2]:>9.2f}"
                          f"  {tw_str:>11} {dt_str:>16}"
                          f"  {norm_cp_direction[0]:>6.3f} {norm_cp_direction[1]:>6.3f} {norm_cp_direction[2]:>6.3f}")

                    count += 1
                    dwell_idx_in_channel += 1

    except Exception as e:
        print("Error reading RTPLAN:", e)

    print(f"\n{'='*80}")
    print(f"Total dwell points across all channels: {count}")
    print(f"{'='*80}\n")

    return dwells, count

def get_rtdose_grid_info(folder, skip_keywords=None):
    """Load the same native RTDOSE dataset used as the reference grid."""
    if skip_keywords is None:
        skip_keywords = ["Contours"]
    rtdose_paths = []
    for root, dirs, files in os.walk(folder):
        if any(k in root for k in skip_keywords):
            continue
        for f in files:
            path = os.path.join(root, f)
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True)
                if getattr(ds, "Modality", None) == "RTDOSE":
                    rtdose_paths.append(path)
            except Exception:
                continue
    rtdose_path, _ = select_native_rtdose(rtdose_paths)
    if rtdose_path:
        return pydicom.dcmread(rtdose_path)
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



# ============================================================================
# RTSTRUCT parsing — structure extraction, CTV/PTV/OAR classification, masks
# ============================================================================

# Name fragments that suggest a CTV (case-insensitive matching)
_CTV_NAME_HINTS = ("ctv", "prostate")
# Name fragments that suggest a PTV / other expanded target
_PTV_NAME_HINTS = ("ptv", "gtv", "target", "itv", "igtv")
# Name fragments commonly associated with OAR
_OAR_NAME_HINTS = (
    "bladder", "rectum", "bowel", "sigmoid", "femur", "femoral",
    "spinal", "cord", "brainstem", "parotid", "lung", "heart",
    "kidney", "liver", "esophagus", "optic", "eye", "lens",
    "cochlea", "urethra", "vagina", "skin", "body", "external",
    "ring", "shell", "wall", "mucosa", "intestine", "colon",
    "stomach", "duodenum", "penile", "bulb",
)


def parse_rtstruct(rtstruct):
    """
    Parse an RTSTRUCT dataset into a list of structure info dicts.

    Each dict contains:
        roi_number  : int          — DICOM ROI Number
        name        : str          — Structure name from StructureSetROISequence
        interpreted_type : str     — RTROIInterpretedType (e.g. "PTV", "ORGAN", "AVOIDANCE", "")
        color       : list[int]    — RGB display color [r, g, b]
        contours    : list[dict]   — per-slice contours, each with keys:
            "z"      : float       — z coordinate (mm) of the contour plane
            "points" : np.ndarray  — (N, 2) array of (x, y) polygon vertices
    """
    if rtstruct is None:
        return []

    # Build lookup: ROI Number → name
    roi_name_map = {}
    for roi in rtstruct.StructureSetROISequence:
        roi_name_map[int(roi.ROINumber)] = roi.ROIName

    # Build lookup: Referenced ROI Number → interpreted type, such as PTV, organ.
    roi_type_map = {}
    obs_seq = getattr(rtstruct, "RTROIObservationsSequence", [])
    for obs in obs_seq:
        ref_num = int(obs.ReferencedROINumber)
        roi_type_map[ref_num] = getattr(obs, "RTROIInterpretedType", "")

    # Build lookup: Referenced ROI Number → color. It is used mainly for visualization (drawing prostate, bladder, rectum with different colors).
    roi_color_map = {}
    contour_seq = getattr(rtstruct, "ROIContourSequence", [])
    for roi_contour in contour_seq:
        ref_num = int(roi_contour.ReferencedROINumber)
        roi_color_map[ref_num] = [int(c) for c in getattr(roi_contour, "ROIDisplayColor", [255, 0, 0])]

    # Extract contour geometry per structure
    structures = []
    for roi_contour in contour_seq:
        ref_num = int(roi_contour.ReferencedROINumber)
        name = roi_name_map.get(ref_num, f"ROI_{ref_num}")
        interpreted_type = roi_type_map.get(ref_num, "")
        color = roi_color_map.get(ref_num, [255, 0, 0])

        contours = []
        contour_data_seq = getattr(roi_contour, "ContourSequence", [])
        for contour_item in contour_data_seq:
            n_pts = int(contour_item.NumberOfContourPoints)
            raw = np.array(contour_item.ContourData, dtype=float).reshape(-1, 3)
            z_val = float(raw[0, 2])
            
            # get the coordinate of the point on the contour with z
            contours.append({
                "z": z_val,
                "points": raw[:, :2],   # (N, 2) — x, y only
                "n_points": n_pts,
            })

        structures.append({
            "roi_number": ref_num,
            "name": name,
            "interpreted_type": interpreted_type,
            "color": color,
            "contours": contours,
        })

    return structures


def classify_structure(structure):
    """
    Classify a structure as "CTV", "PTV", "OAR", or "OTHER".

    Explicit DICOM types (CTV/PTV/GTV) are authoritative.  For generic
    types like "ORGAN", name-based heuristics decide — e.g. "Prostate"
    is treated as a CTV in brachytherapy even though its DICOM type
    is ORGAN.
    """
    itype = structure["interpreted_type"].upper().strip()
    name_lower = structure["name"].lower()

    # Explicit DICOM target types
    if itype == "CTV":
        return "CTV"
    if itype in ("PTV", "GTV"):
        return "PTV"

    # Name-based CTV hints (checked before PTV so "CTV" wins over broader targets)
    for hint in _CTV_NAME_HINTS:
        if hint in name_lower:
            return "CTV"

    # Name-based PTV / expanded-target hints
    for hint in _PTV_NAME_HINTS:
        if hint in name_lower:
            return "PTV"
    
    if itype in ("ORGAN", "AVOIDANCE", "OAR"):
        return "OAR"

    for hint in _OAR_NAME_HINTS:
        if hint in name_lower:
            return "OAR"

    return "OTHER"


def identify_structures(rtstruct):
    """
    High-level convenience: parse RTSTRUCT and classify every structure.

    Returns
    -------
    structures : list[dict]
        Each dict is the parse_rtstruct output augmented with a "classification" key.
    ctv_names  : list[str]
    ptv_names  : list[str]
    oar_names  : list[str]
    """
    structures = parse_rtstruct(rtstruct)
    ctv_names = []
    ptv_names = []
    oar_names = []

    for s in structures:
        cls = classify_structure(s)
        s["classification"] = cls
        if cls == "CTV":
            ctv_names.append(s["name"])
        elif cls == "PTV":
            ptv_names.append(s["name"])
        elif cls == "OAR":
            oar_names.append(s["name"])

    return structures, ctv_names, ptv_names, oar_names


def contour_to_mask(structure, grid_origin, grid_spacing, grid_shape):
    """
    Rasterize a structure's contour polygons into a 3D binary mask on the given grid.

    Parameters
    ----------
    structure : dict
        One element from parse_rtstruct / identify_structures output.
    grid_origin : array-like, shape (3,)
        (x0, y0, z0) of the first voxel in the target grid (e.g. RTDOSE grid).
    grid_spacing : array-like, shape (3,)
        (dx, dy, dz) voxel size in mm.
    grid_shape : tuple (nz, ny, nx)
        Shape of the target 3D grid.

    Returns
    -------
    mask : np.ndarray, dtype bool, shape (nz, ny, nx)
    """
    nz, ny, nx = grid_shape
    ox, oy, oz = float(grid_origin[0]), float(grid_origin[1]), float(grid_origin[2])
    dx, dy, dz = float(grid_spacing[0]), float(grid_spacing[1]), float(grid_spacing[2])

    mask = np.zeros(grid_shape, dtype=bool)

    # Pre-compute z coordinates of every grid slice
    z_coords = oz + np.arange(nz) * dz
    z_tolerance = abs(dz) / 2.0

    # x, y pixel center coordinates
    x_centers = ox + np.arange(nx) * dx
    y_centers = oy + np.arange(ny) * dy
    xx, yy = np.meshgrid(x_centers, y_centers)  # (ny, nx) each
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])  # (ny*nx, 2)

    for contour in structure["contours"]:
        cz = contour["z"]
        pts = contour["points"]  # (N, 2) — x, y polygon

        if len(pts) < 3:
            continue

        # Find the grid z-index closest to this contour's z
        z_idx = np.argmin(np.abs(z_coords - cz))
        if abs(z_coords[z_idx] - cz) > z_tolerance:
            continue

        poly_path = Path(pts)
        inside = poly_path.contains_points(grid_points).reshape(ny, nx)
        # XOR-accumulate so nested contours (holes) toggle correctly
        mask[z_idx] ^= inside

    return mask


def build_structure_masks(structures, grid_origin, grid_spacing, grid_shape, classifications=None):
    """
    Build binary masks for a list of structures on the given grid.

    Parameters
    ----------
    structures : list[dict]
        Output from identify_structures.
    grid_origin, grid_spacing, grid_shape :
        Target grid parameters (e.g. from the RTDOSE grid).
    classifications : list[str] or None
        If given, only build masks for structures whose "classification" is in this list.
        E.g. ["CTV", "PTV", "OAR"] to skip "OTHER".

    Returns
    -------
    masks : dict[str, np.ndarray]
        {structure_name: bool mask} for each selected structure.
    """
    masks = {}
    for s in structures:
        if classifications is not None and s.get("classification") not in classifications:
            continue
        print(f"  Rasterizing '{s['name']}' ({s.get('classification', '?')}) "
              f"— {len(s['contours'])} contour slices ...")
        masks[s["name"]] = contour_to_mask(s, grid_origin, grid_spacing, grid_shape)
        n_voxels = int(masks[s["name"]].sum())
        print(f"    -> {n_voxels} voxels in mask")
    return masks