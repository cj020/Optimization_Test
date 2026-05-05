import os
import pydicom
import SimpleITK as sitk

def load_ct(folder):
    for f in os.listdir(folder): # os.listdir() returns a list of all files in the folder.
        path = os.path.join(folder, f) 
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True) 
            print(f, ds.Modality) 
        except:
            pass
    
    # Read CT series
    # Do NOT just load files randomly. You must sort them properly
    reader = sitk.ImageSeriesReader() 
    series_IDs = reader.GetGDCMSeriesIDs(folder) 

    if not series_IDs:
        raise ValueError("No DICOM series found in folder")

    print("Series IDs:", series_IDs)

    # Takes one series ID, Returns all file paths belonging to that series
    files = reader.GetGDCMSeriesFileNames(folder, series_IDs[0])
    reader.SetFileNames(files) 

    image = reader.Execute() 

    # Convert to NumPy
    volume = sitk.GetArrayFromImage(image)  

    print("Shape:", volume.shape) 


    # Get geometry
    spacing = image.GetSpacing()   
    origin = image.GetOrigin() 
    direction = image.GetDirection() 

    print("Spacing:", spacing)
    print("Origin:", origin)
    print("Direction:", direction)

    # Read RTPLAN (for dwell positions)
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        
        if ds.Modality == "RTPLAN":
            rtplan = pydicom.dcmread(path)
            print("Found RTPLAN:", f)

    # Read RTDOSE
    # for f in os.listdir(folder):
    #     path = os.path.join(folder, f)
    #     ds = pydicom.dcmread(path, stop_before_pixels=True)
        
    #     if ds.Modality == "RTDOSE":
    #         dose_ds = pydicom.dcmread(path)
    #         dose = dose_ds.pixel_array * dose_ds.DoseGridScaling 
    #         print("Dose shape:", dose.shape)

    return volume, spacing, origin, direction, rtplan

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
                        pos = cp.ControlPoint3DPosition  # (x, y, z)
                        dwell_positions.append(pos)

                        print(f"Dwell {count}: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}")
                        count += 1

    except Exception as e:
        print("Error reading RTPLAN:", e)

    print(f"\nTotal dwell positions: {count}")
    return dwell_positions, count