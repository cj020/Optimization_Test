import sys

from pipeline import load_case, optimize_case


def run_cli(folder):
    """Headless run: load the case and optimise with the default parameters."""
    case = load_case(folder)
    optimize_case(case)


def main():
    # GUI: the window opens empty; use File > Import DICOM Folder... to load a case.
    # Batch: python main.py --cli "C:\path\to\patient folder"
    if len(sys.argv) >= 3 and sys.argv[1] == "--cli":
        run_cli(sys.argv[2])
        return

    from ui import run_app
    sys.exit(run_app())


if __name__ == "__main__":
    main()
