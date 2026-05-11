import os
import pydicom
import SimpleITK as sitk
import numpy as np

# folder = r"C:\Users\jichen\Downloads\T00060\T00060"

def load_dicom(folder, skip_keywords=None):
    """
    Robust DICOM loader for CT + RTSTRUCT + RTPLAN + RTDOSE.

    Parameters
    ----------
    folder : str
        Root folder of patient data
    skip_keywords : list[str] or None
        Folder names to ignore (e.g. ["Contours"])

    Returns
    -------
    volume : np.ndarray
        CT volume (z, y, x)
    spacing : tuple
    origin : tuple
    direction : tuple
    rtstruct : pydicom.Dataset or None
    rtplan : pydicom.Dataset or None
    rtdose : np.ndarray or None
    """
    if skip_keywords is None:
        skip_keywords = ["Contours", "ct"]  # optional safety filter

    ct_series_files = []
    rtstruct_path = None
    rtplan_path = None
    rtdose_path = None

    print("\n🔍 Scanning DICOM files...\n")

    # =========================
    # 1. Scan recursively
    # =========================
    # for root, dirs, files in os.walk(folder):
    #     print("\n📁 ROOT:", root)
    #     print("📂 DIRS:", dirs)
    #     print("📄 FILES:", files[:10])  # show only first 10 files

    for root, dirs, files in os.walk(folder): # os.walk() recursively traverses the folder, returning the current directory (root), subdirectories (dirs), and files (files) at each level.

        # skip unwanted folders
        if any(k in root for k in skip_keywords):
            continue

        for f in files:
            path = os.path.join(root, f)

            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True) # stop_before_pixels=True speeds up reading by not loading pixel data, which is not needed for modality checking.
            except Exception:
                continue

            modality = getattr(ds, "Modality", None) # getattr is used to safely access the Modality attribute, returning None if it doesn't exist instead of raising an error.

            # CT slices
            if modality == "CT":
                ct_series_files.append(path)

            # RTSTRUCT (only keep first found)
            elif modality == "RTSTRUCT" and rtstruct_path is None:
                rtstruct_path = path

            # RTPLAN
            elif modality == "RTPLAN" and rtplan_path is None:
                rtplan_path = path

            # RTDOSE
            elif modality == "RTDOSE" and rtdose_path is None:
                rtdose_path = path

            print(f"{modality:10} | {path}")
    
    # =========================
    # 2. Load CT properly
    # =========================
    print("\n🧠 Loading CT volume...")

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(folder)

    if not series_ids:
        raise ValueError("No CT series found")

    series_files = reader.GetGDCMSeriesFileNames(folder, series_ids[0])
    reader.SetFileNames(series_files)

    image = reader.Execute()

    volume = sitk.GetArrayFromImage(image) # (nz, ny, nx)

    spacing = image.GetSpacing() # (dx, dy, dz) in mm
    origin = image.GetOrigin() # (x0, y0, z0), the physical (world/patient) coordinate of the first voxel in the CT volume.
    direction = image.GetDirection() 

    print("CT shape:", volume.shape)
    print("Spacing:", spacing)
    print("Origin:", origin)

    # =========================
    # 3. Load RTSTRUCT
    # =========================
    rtstruct = None
    if rtstruct_path:
        print("\n📌 Loading RTSTRUCT")
        rtstruct = pydicom.dcmread(rtstruct_path)

    # =========================
    # 4. Load RTPLAN
    # =========================
    rtplan = None
    if rtplan_path:
        print("📌 Loading RTPLAN")
        rtplan = pydicom.dcmread(rtplan_path)

    # =========================
    # 5. Load RTDOSE
    # =========================
    rtdose = None
    if rtdose_path:
        print("📌 Loading RTDOSE")
        dose_ds = pydicom.dcmread(rtdose_path)
        rtdose = dose_ds.pixel_array * dose_ds.DoseGridScaling

        print("Dose shape:", rtdose.shape) # (nz, ny, nx)
        
        # rtdose is a 3D array with dimensions (z, y, x) corresponding to the dose grid. The spacing and origin of the dose grid can be obtained from the DICOM tags as follows:
        dy, dx = dose_ds.PixelSpacing
        z_offsets = dose_ds.GridFrameOffsetVector
        dz = z_offsets[1] - z_offsets[0]
        dose_spacing = (dz, dy, dx)  # note the order (dz, dy, dx)
        
        print("Dose spacing:", dose_spacing)

        dose_origin = dose_ds.ImagePositionPatient # (x0, y0, z0), the physical (world/patient) coordinate of the first voxel in the dose grid

        print("Dose origin:", dose_origin)

    return volume, spacing, origin, direction, rtstruct, rtplan, rtdose

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
                        pos = cp.ControlPoint3DPosition  # (x, y, z) in mm, in the patient coordinate system (world coordinates)
                        
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

def extract_dwell_points_with_dwell_time_and_local_direction(rtplan, local_directions = False): # extract dwell positions along with their corresponding dwell times calculated from the cumulative time weights and channel total time, an the local direction.

    dwells = []
    count = 0

    print("\nDwell points without repeat:\n")

    if rtplan is None:
        print("No RTPLAN available")
        return dwells, count

    try:
        for app in rtplan.ApplicationSetupSequence:

            for channel in app.ChannelSequence: # A channel = catheter path.

                cps = channel.BrachyControlPointSequence

                # IMPORTANT: process in pairs
                for i in range(0, len(cps) - 1, 2):

                    cp_start = cps[i]
                    cp_end = cps[i + 1]

                    if not hasattr(cp_start, "ControlPoint3DPosition"):
                        continue

                    # --- position (world coordinates)
                    pos = np.array(cp_start.ControlPoint3DPosition, dtype=float)

                    # --- time weight difference
                    w1 = getattr(cp_start, "CumulativeTimeWeight", None)
                    w2 = getattr(cp_end, "CumulativeTimeWeight", None)

                    if w1 is None or w2 is None:
                        dwell_time = None
                    else:
                        weight = w2 - w1
                        channel_time = float(getattr(channel, "ChannelTotalTime", 1.0))
                        dwell_time = weight * channel_time

                    # The first and last control points
                    cp_first = cps[0]
                    cp_last = cps[len(cps) - 1]
                    vec_first_to_last = np.array(cp_last.ControlPoint3DPosition, dtype=float) - np.array(cp_first.ControlPoint3DPosition, dtype=float)
                    norm_vec_first_to_last = np.linalg.norm(vec_first_to_last)
                    
                    if norm_vec_first_to_last > 0:
                        norm_vec_direction = (
                            vec_first_to_last / norm_vec_first_to_last
                        )
                    else:
                        norm_vec_direction = np.array([0.0, 0.0, 0.0])

                    # --- local direction
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
                            norm_cp_direction = (
                                cp_local_direction_vec / local_direction_vec_norm
                            )
                        else:
                            norm_cp_direction = np.array([0.0, 0.0, 0.0])
                    
                    else: # calculate direction vector by approximating the direction of the catheter
                        norm_cp_direction = norm_vec_direction

                    # --- store one true dwell
                    dwells.append([
                        count,
                        channel.ChannelNumber,
                        dwell_time,
                        pos, 
                        norm_cp_direction
                    ])

                    print(
                        f"Dwell {count}: "
                        f"Channel={channel.ChannelNumber}, "
                        f"Time={dwell_time}, "
                        f"Pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), "
                        f"LocalDirection=({norm_cp_direction[0]:.2f}, {norm_cp_direction[1]:.2f}, {norm_cp_direction[2]:.2f})"
                    )

                    count += 1

    except Exception as e:
        print("Error reading RTPLAN:", e)

    print(f"\nTotal dwell points: {count}")

    return dwells, count