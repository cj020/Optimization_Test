"""2-D slice viewer widget for axial / sagittal / coronal orientations."""

from __future__ import annotations

from enum import IntEnum

import vtk
from PyQt5.QtWidgets import QFrame, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


class Orientation(IntEnum):
    AXIAL = 2     # XY plane, scroll along Z
    CORONAL = 1   # XZ plane, scroll along Y
    SAGITTAL = 0  # YZ plane, scroll along X


_LABELS = {
    Orientation.AXIAL: "Axial",
    Orientation.CORONAL: "Coronal",
    Orientation.SAGITTAL: "Sagittal",
}


class SliceViewWidget(QFrame):
    """Embeds a vtkResliceImageViewer inside a QFrame with a label overlay."""

    def __init__(self, orientation: Orientation, parent=None) -> None:
        super().__init__(parent)
        self.orientation = orientation
        self._image_data: vtk.vtkImageData | None = None

        self.setFrameStyle(QFrame.Box | QFrame.Sunken)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.label = QLabel(_LABELS[orientation])
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet(
            "background-color: rgba(0,0,0,180); color: #00ff00; "
            "font-weight: bold; padding: 2px 6px;"
        )
        layout.addWidget(self.label)

        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk_widget, stretch=1)

        self.viewer = vtk.vtkResliceImageViewer()
        self.viewer.SetRenderWindow(self.vtk_widget.GetRenderWindow())
        self.viewer.SetupInteractor(self.vtk_widget)
        self.viewer.SetSliceOrientation(int(orientation))

        self.viewer.GetRenderer().SetBackground(0.0, 0.0, 0.0)
        self.viewer.SetResliceModeToAxisAligned()

    # ------------------------------------------------------------------
    def set_image_data(self, image_data: vtk.vtkImageData) -> None:
        """Assign a new volume and reset the viewer to the middle slice."""
        self._image_data = image_data
        self.viewer.SetInputData(image_data)

        dims = image_data.GetDimensions()
        axis = int(self.orientation)
        mid_slice = dims[axis] // 2
        self.viewer.SetSlice(mid_slice)
        self.viewer.Render()
        self._update_label()

    def _update_label(self) -> None:
        name = _LABELS[self.orientation]
        idx = self.viewer.GetSlice()
        smin = self.viewer.GetSliceMin()
        smax = self.viewer.GetSliceMax()
        self.label.setText(f"{name}  [{idx}/{smax}]")

    # ------------------------------------------------------------------
    def initialize(self) -> None:
        """Must be called once after the widget is visible (after show())."""
        self.vtk_widget.Initialize()
        if self._image_data is not None:
            self.vtk_widget.GetRenderWindow().Render()

        iren = self.vtk_widget.GetRenderWindow().GetInteractor()
        iren.AddObserver("MouseWheelForwardEvent", self._on_scroll)
        iren.AddObserver("MouseWheelBackwardEvent", self._on_scroll)

    def _on_scroll(self, _obj, _event) -> None:
        self._update_label()

    def reset_camera(self) -> None:
        self.viewer.GetRenderer().ResetCamera()
        self.viewer.Render()
