import sys

from pipeline import CANDIDATES_ALL, CANDIDATES_EXTENDED_PTV, load_case, optimize_case


def run_cli(folder, candidate_region=CANDIDATES_ALL):
    """Headless run: load the case and optimise with the default parameters."""
    case = load_case(folder)
    optimize_case(case, mesh_params={"candidate_region": candidate_region})


def main():
    # GUI: the window opens empty; use File > Import DICOM Folder... to load a case.
    # Batch: python main.py --cli "C:\path\to\patient folder" [--extended-ptv]
    #   --extended-ptv restricts candidate dwells to the Extended PTV (PTV + 5 mm).
    if len(sys.argv) >= 3 and sys.argv[1] == "--cli":
        region = CANDIDATES_EXTENDED_PTV if "--extended-ptv" in sys.argv[3:] else CANDIDATES_ALL
        run_cli(sys.argv[2], candidate_region=region)
        return

    from ui import run_app
    sys.exit(run_app())


if __name__ == "__main__":
    main()
