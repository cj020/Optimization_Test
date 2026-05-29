import matplotlib
matplotlib.use('TkAgg')

from dicom_loader import load_ct
from ui import start_ui

def main():
    folder = r"C:\Users\jichen\Downloads\T00060-20260422T210320Z-3-001\T00060"

    volume, spacing, origin, direction = load_ct(folder)

    print("Volume shape:", volume.shape)
    print("Intensity range:", volume.min(), volume.max())

    start_ui(volume)

if __name__ == "__main__": # if __name__ == "__main__": is a common Python idiom that checks if the script is being run directly (as the main program) rather than imported as a module in another script. If this condition is true, it calls the main() function to execute the code that loads the CT data and starts the user interface. This allows the script to be used both as a standalone program and as an importable module without executing the main code when imported.
    main()