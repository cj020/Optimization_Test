import os
import pydicom
import SimpleITK as sitk

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

def extract_dwell_positions(rtplan):
    dwell_positions = [] # here the dwell positions will be stored as (x, y, z) in mm (world coordinates)
    
    count = 0 # counter for dwell positions

    if rtplan is None:
        print("No RTPLAN available")
        return dwell_positions, count

    try:
        for app in rtplan.ApplicationSetupSequence:
            for channel in app.ChannelSequence:
                for cp in channel.BrachyControlPointSequence:

                    if hasattr(cp, "ControlPoint3DPosition"): # hasattr check to avoid missing attribute
                        pos = cp.ControlPoint3DPosition  # (x, y, z) in mm, in the patient coordinate system (world coordinates)
                        dwell_positions.append(pos)

                        print(f"Dwell {count}: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}")
                        count += 1

    except Exception as e:
        print("Error reading RTPLAN:", e)

    print(f"\nTotal dwell positions: {count}")
    return dwell_positions, count