import sys

import vtk
from PyQt5.QtWidgets import QApplication

from main_window import MainWindow


def main() -> None:
    vtk.vtkObject.GlobalWarningDisplayOff()

    app = QApplication(sys.argv)
    app.setApplicationName("DICOM Viewer")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
