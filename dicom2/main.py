import matplotlib
matplotlib.use('TkAgg')

from dicom_loader import load_ct, extract_dwell_positions
from ui import start_ui

def main():
    folder = r"C:\Users\jichen\Downloads\T00060\T00060"

    volume, spacing, origin, direction, rtplan = load_ct(folder)

    print("Volume shape:", volume.shape)
    print("Intensity range:", volume.min(), volume.max())

    dwell_positions, count = extract_dwell_positions(rtplan)
    print(f"Extracted {count} dwell positions:")

    start_ui(volume, spacing, origin, dwell_positions)
    
if __name__ == "__main__": 
    main()
