"""
CerebraSense DICOM Orchestration Engine
Handles medical imaging format conversion and HIPAA-aligned anonymization.
"""

import os
import pydicom
import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any

def anonymize_dicom_metadata(ds: pydicom.dataset.Dataset) -> pydicom.dataset.Dataset:
    """Strip PHI (Protected Health Information) from DICOM headers."""
    # List of tags to strip or replace
    tags_to_anonymize = [
        "PatientName", "PatientID", "PatientBirthDate", 
        "InstitutionName", "ReferringPhysicianName", "OperatorsName"
    ]
    for tag in tags_to_anonymize:
        if tag in ds:
            ds.data_element(tag).value = "ANONYMIZED"
    return ds

def convert_dicom_folder_to_nifti(
    dicom_dir: Path, 
    output_path: Path
) -> Tuple[Path, Dict[str, Any]]:
    """
    Converts a clinical DICOM folder into a NIfTI volume for model inference.
    Returns the path to the NIfTI and extracted clinical metadata.
    """
    # 1. Load DICOM slices
    files = [dicom_dir / f for f in os.listdir(dicom_dir) if f.endswith(".dcm") or f.endswith(".DCM")]
    if not files:
        raise ValueError(f"No DICOM files found in {dicom_dir}")
        
    slices = [pydicom.dcmread(f) for f in files]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2])) # Sort by Z-axis
    
    # 2. Extract Volume Data
    pixel_data = np.stack([s.pixel_array for s in slices])
    pixel_data = pixel_data.astype(np.float32)
    
    # 3. Handle Metadata
    first_slice = slices[0]
    clinical_meta = {
        "age": str(getattr(first_slice, "PatientAge", "070Y")).strip("Y"),
        "sex": str(getattr(first_slice, "PatientSex", "F")),
        "institution": str(getattr(first_slice, "InstitutionName", "Local Clinic"))
    }
    
    # 4. Create NIfTI
    # Note: For high-quality, we should resolve the affine from DICOM tags
    # Here we use a standard identity or a simple spacing affine
    affine = np.eye(4)
    try:
        spacing = [float(s) for s in first_slice.PixelSpacing] + [float(first_slice.SliceThickness)]
        affine[0,0], affine[1,1], affine[2,2] = spacing[0], spacing[1], spacing[2]
    except: pass
    
    img = nib.Nifti1Image(pixel_data, affine)
    nib.save(img, str(output_path))
    
    return output_path, clinical_meta
