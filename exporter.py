"""Export a vtkImageData volume back to a DICOM series on disk."""

from __future__ import annotations

import os
import datetime

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def export_dicom(
    image_data: vtk.vtkImageData,
    output_dir: str,
    patient_name: str = "Anonymous",
    series_description: str = "Exported from DICOM Viewer",
) -> None:
    """Write *image_data* as a DICOM series (one file per axial slice).

    Raises
    ------
    OSError
        If the output directory cannot be created.
    """
    os.makedirs(output_dir, exist_ok=True)

    dims = image_data.GetDimensions()       # (cols, rows, slices)
    spacing = image_data.GetSpacing()
    origin = image_data.GetOrigin()

    scalars = image_data.GetPointData().GetScalars()
    np_volume = vtk_to_numpy(scalars)

    if np_volume.dtype not in (np.int16, np.uint16):
        np_volume = np_volume.astype(np.int16)

    np_volume = np_volume.reshape(dims[2], dims[1], dims[0])

    study_uid = generate_uid()
    series_uid = generate_uid()
    frame_of_ref_uid = generate_uid()
    now = datetime.datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S.%f")

    for z_idx in range(dims[2]):
        sop_uid = generate_uid()
        filename = os.path.join(output_dir, f"slice_{z_idx:05d}.dcm")

        file_meta = pydicom.Dataset()
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT
        file_meta.MediaStorageSOPInstanceUID = sop_uid
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\x00" * 128)

        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = sop_uid
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = frame_of_ref_uid

        ds.Modality = "CT"
        ds.Manufacturer = "DICOM Viewer Export"
        ds.PatientName = patient_name
        ds.PatientID = "0000"
        ds.StudyDate = date_str
        ds.StudyTime = time_str
        ds.ContentDate = date_str
        ds.ContentTime = time_str
        ds.SeriesDescription = series_description

        ds.Rows = dims[1]
        ds.Columns = dims[0]
        ds.NumberOfFrames = 1
        ds.InstanceNumber = z_idx + 1
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1  # signed
        ds.RescaleIntercept = "0"
        ds.RescaleSlope = "1"

        ds.PixelSpacing = [f"{spacing[0]:.6f}", f"{spacing[1]:.6f}"]
        ds.SliceThickness = f"{spacing[2]:.6f}"
        ds.ImagePositionPatient = [
            f"{origin[0]:.6f}",
            f"{origin[1]:.6f}",
            f"{origin[2] + z_idx * spacing[2]:.6f}",
        ]
        ds.ImageOrientationPatient = ["1", "0", "0", "0", "1", "0"]

        ds.PixelData = np_volume[z_idx].tobytes()
        ds.save_as(filename)
