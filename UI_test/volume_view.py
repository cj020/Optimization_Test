"""3-D volume-rendering widget using vtkSmartVolumeMapper (GPU ray-cast)."""

from __future__ import annotations

import vtk
from PyQt5.QtWidgets import QFrame, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


class VolumeViewWidget(QFrame):
    """Renders a vtkImageData volume with ray-cast transfer functions."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFrameStyle(QFrame.Box | QFrame.Sunken)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.label = QLabel("3D Volume")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet(
            "background-color: rgba(0,0,0,180); color: #00ff00; "
            "font-weight: bold; padding: 2px 6px;"
        )
        layout.addWidget(self.label)

        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk_widget, stretch=1)

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.1, 0.1, 0.15)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)

        self._volume: vtk.vtkVolume | None = None

    # ------------------------------------------------------------------
    def set_image_data(self, image_data: vtk.vtkImageData) -> None:
        """Build (or rebuild) the volume-rendering pipeline for *image_data*."""
        if self._volume is not None:
            self.renderer.RemoveVolume(self._volume)

        mapper = vtk.vtkSmartVolumeMapper()
        mapper.SetInputData(image_data)

        color_tf = vtk.vtkColorTransferFunction()
        color_tf.AddRGBPoint(-1000, 0.0, 0.0, 0.0)   # air  -> black
        color_tf.AddRGBPoint(-600, 0.26, 0.15, 0.07)  # lung -> dark brown
        color_tf.AddRGBPoint(-400, 0.55, 0.25, 0.15)  # fat
        color_tf.AddRGBPoint(40, 0.88, 0.60, 0.50)    # soft tissue -> skin
        color_tf.AddRGBPoint(80, 0.90, 0.55, 0.45)
        color_tf.AddRGBPoint(400, 1.0, 0.99, 0.95)    # bone -> white
        color_tf.AddRGBPoint(2000, 1.0, 1.0, 1.0)

        opacity_tf = vtk.vtkPiecewiseFunction()
        opacity_tf.AddPoint(-1000, 0.00)
        opacity_tf.AddPoint(-600, 0.00)
        opacity_tf.AddPoint(-400, 0.05)
        opacity_tf.AddPoint(40, 0.15)
        opacity_tf.AddPoint(80, 0.25)
        opacity_tf.AddPoint(400, 0.70)
        opacity_tf.AddPoint(2000, 1.00)

        gradient_opacity = vtk.vtkPiecewiseFunction()
        gradient_opacity.AddPoint(0, 0.0)
        gradient_opacity.AddPoint(90, 0.5)
        gradient_opacity.AddPoint(100, 1.0)

        vol_property = vtk.vtkVolumeProperty()
        vol_property.SetColor(color_tf)
        vol_property.SetScalarOpacity(opacity_tf)
        vol_property.SetGradientOpacity(gradient_opacity)
        vol_property.ShadeOn()
        vol_property.SetInterpolationTypeToLinear()
        vol_property.SetAmbient(0.4)
        vol_property.SetDiffuse(0.6)
        vol_property.SetSpecular(0.2)

        volume = vtk.vtkVolume()
        volume.SetMapper(mapper)
        volume.SetProperty(vol_property)

        self.renderer.AddVolume(volume)
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
        self._volume = volume

    # ------------------------------------------------------------------
    def initialize(self) -> None:
        """Must be called once after the widget is visible (after show())."""
        self.vtk_widget.Initialize()
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.vtk_widget.GetRenderWindow().GetInteractor().SetInteractorStyle(style)
        self.vtk_widget.GetRenderWindow().Render()

    def reset_camera(self) -> None:
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
