"""Main application window – menu bar, 2x2 viewer grid, status bar."""

from __future__ import annotations

import os
import traceback

import vtk
from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QGridLayout,
    QFileDialog,
    QMessageBox,
    QAction,
    QStatusBar,
)
from PyQt5.QtCore import Qt, QTimer

from slice_view import SliceViewWidget, Orientation
from volume_view import VolumeViewWidget
from dicom_loader import load_dicom_folder
from exporter import export_dicom


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DICOM Viewer")
        self.resize(1200, 800)

        self._image_data: vtk.vtkImageData | None = None

        self._build_menu()
        self._build_views()

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready – use File ▸ Import DICOM to load a study.")

        # Defer VTK interactor initialization until the window is on-screen.
        QTimer.singleShot(0, self._initialize_vtk)

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------
    def _build_menu(self) -> None:
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")

        import_action = QAction("&Import DICOM Folder…", self)
        import_action.setShortcut("Ctrl+O")
        import_action.triggered.connect(self._on_import)
        file_menu.addAction(import_action)

        self._export_action = QAction("&Export DICOM Series…", self)
        self._export_action.setShortcut("Ctrl+S")
        self._export_action.setEnabled(False)
        self._export_action.triggered.connect(self._on_export)
        file_menu.addAction(self._export_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    # ------------------------------------------------------------------
    # Views (2 × 2 grid)
    # ------------------------------------------------------------------
    def _build_views(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout(central)
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setSpacing(4)

        self.axial_view = SliceViewWidget(Orientation.AXIAL)
        self.sagittal_view = SliceViewWidget(Orientation.SAGITTAL)
        self.coronal_view = SliceViewWidget(Orientation.CORONAL)
        self.volume_view = VolumeViewWidget()

        grid.addWidget(self.axial_view, 0, 0)
        grid.addWidget(self.sagittal_view, 0, 1)
        grid.addWidget(self.coronal_view, 1, 0)
        grid.addWidget(self.volume_view, 1, 1)

        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

    def _initialize_vtk(self) -> None:
        self.axial_view.initialize()
        self.sagittal_view.initialize()
        self.coronal_view.initialize()
        self.volume_view.initialize()

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------
    def _on_import(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select DICOM Folder", "", QFileDialog.ShowDirsOnly
        )
        if not folder:
            return

        self.status_bar.showMessage(f"Loading {folder} …")
        self.setCursor(Qt.WaitCursor)
        try:
            image_data = load_dicom_folder(folder)
        except Exception as exc:
            self.unsetCursor()
            self.status_bar.showMessage("Import failed.")
            QMessageBox.critical(
                self,
                "Import Error",
                f"Failed to load DICOM data:\n\n{exc}\n\n{traceback.format_exc()}",
            )
            return

        self._image_data = image_data
        self._feed_views(image_data)

        basename = os.path.basename(folder)
        self.setWindowTitle(f"DICOM Viewer – {basename}")
        self._export_action.setEnabled(True)
        dims = image_data.GetDimensions()
        self.status_bar.showMessage(
            f"Loaded {basename}  ({dims[0]}×{dims[1]}×{dims[2]} voxels)"
        )
        self.unsetCursor()

    def _feed_views(self, image_data: vtk.vtkImageData) -> None:
        self.axial_view.set_image_data(image_data)
        self.sagittal_view.set_image_data(image_data)
        self.coronal_view.set_image_data(image_data)
        self.volume_view.set_image_data(image_data)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def _on_export(self) -> None:
        if self._image_data is None:
            return

        folder = QFileDialog.getExistingDirectory(
            self, "Select Export Folder", "", QFileDialog.ShowDirsOnly
        )
        if not folder:
            return

        self.status_bar.showMessage(f"Exporting to {folder} …")
        self.setCursor(Qt.WaitCursor)
        try:
            export_dicom(self._image_data, folder)
        except Exception as exc:
            self.unsetCursor()
            self.status_bar.showMessage("Export failed.")
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export DICOM:\n\n{exc}\n\n{traceback.format_exc()}",
            )
            return

        self.unsetCursor()
        self.status_bar.showMessage(f"Exported to {folder}")
        QMessageBox.information(self, "Export Complete", f"DICOM series written to:\n{folder}")
