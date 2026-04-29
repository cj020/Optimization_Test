"""Load a folder of DICOM files into a vtkImageData volume."""

from __future__ import annotations

import os

import vtk


def load_dicom_folder(folder_path: str) -> vtk.vtkImageData:
    """Read all DICOM slices in *folder_path* and return a vtkImageData volume.

    Raises
    ------
    FileNotFoundError
        If the folder does not exist.
    RuntimeError
        If the folder contains no readable DICOM files.
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Directory not found: {folder_path}")

    reader = vtk.vtkDICOMImageReader()
    reader.SetDirectoryName(folder_path)
    reader.Update()

    image_data: vtk.vtkImageData = reader.GetOutput()
    dims = image_data.GetDimensions()

    if dims[0] < 2 or dims[1] < 2 or dims[2] < 1:
        raise RuntimeError(
            f"No valid DICOM data found in '{folder_path}'. "
            "Make sure the folder contains DICOM files (.dcm)."
        )

    return image_data
