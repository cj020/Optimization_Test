import matplotlib
matplotlib.use('TkAgg')

from dicom_loader import load_dicom, extract_dwell_positions
from ui import start_ui
import dose_contribution as dc

def main():
    folder = r"C:\Users\jichen\Downloads\T00060\T00060"

    volume, spacing, origin, direction, rtstruct, rtplan, rtdose = load_dicom(folder)

    # print("Volume shape:", volume.shape)
    print("Intensity range:", volume.min(), volume.max())

    dwell_positions, count = extract_dwell_positions(rtplan)
    print(f"Extracted {count} dwell positions:")

    start_ui(volume, spacing, origin, dwell_positions)
    
    distance = dc.dose_contribution(dwell_positions, count, rtdose, volume, spacing, origin)
    print("Distance shape:", distance.shape)
    
if __name__ == "__main__": 
    main()
