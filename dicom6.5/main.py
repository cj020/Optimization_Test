import sys

from pipeline import (
    CANDIDATES_ALL,
    CANDIDATES_EXTENDED_PTV,
    GRID_1MM,
    GRID_3MM,
    load_case,
    optimize_case,
)


def run_cli(folder, candidate_region=CANDIDATES_ALL, grid_mm=GRID_1MM):
    """Headless run: load the case and optimise with the default parameters."""
    case = load_case(folder, grid_mm=grid_mm)
    optimize_case(case, mesh_params={"candidate_region": candidate_region})


def main():
    # GUI: the window opens empty; use File > Import DICOM Folder... to load a case.
    # Batch: python main.py --cli "C:\path\to\patient folder" [--extended-ptv] [--grid 3]
    #   --extended-ptv restricts candidate dwells to the Extended PTV (PTV + 5 mm).
    #   --grid 3 resamples CT and US to 3 mm (default is the current 1 mm US grid).
    if len(sys.argv) >= 3 and sys.argv[1] == "--cli":
        extra = sys.argv[3:]
        region = CANDIDATES_EXTENDED_PTV if "--extended-ptv" in extra else CANDIDATES_ALL
        grid_mm = GRID_1MM
        if "--grid" in extra:
            i = extra.index("--grid")
            if i + 1 < len(extra) and extra[i + 1] in ("3", "3mm"):
                grid_mm = GRID_3MM
        run_cli(sys.argv[2], candidate_region=region, grid_mm=grid_mm)
        return

    from ui import run_app
    sys.exit(run_app())


if __name__ == "__main__":
    main()
