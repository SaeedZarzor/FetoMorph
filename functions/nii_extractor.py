from deps import *
import os
import nibabel as nib
import numpy as np
from pathlib import Path
# from PyQt5.QtWidgets import QMessageBox  # if not supplied by deps

def nifti_extractor(parent, file_path: str, out_dir: str, valid_labels: set[int]):
    # Load canonical RAS+ orientation
    nii = nib.as_closest_canonical(nib.load(file_path))

    # Cast voxel data to int so label comparisons are exact
    img = nii.get_fdata(dtype=np.float32)  # cheap load
    img = img.astype(np.int32, copy=False)

    if not valid_labels:
        QMessageBox.critical(parent, "Export Failed",
                             "No valid labels selected. Please select at least one label before proceeding!")
        return

    print("[NIfTI Extractor] Extracting regions:", valid_labels)

    # Build boolean mask of selected labels
    mask = np.isin(img, list(valid_labels))

    # Find slice indices along axis 1 (Y) that contain any selected label
    # Note: this is axis 1, not Z. Update if you intend Z.
    slice_idxs = np.where(np.any(mask, axis=(0, 2)))[0]

    if slice_idxs.size == 0:
        QMessageBox.information(parent, "[NIfTI Extractor]", "No slices contain the selected mask.")
        return

    # Keep only those slices
    filtered = mask[:, slice_idxs, :].astype(np.uint8, copy=False)

    # Ensure output dir exists
    os.makedirs(out_dir, exist_ok=True)

    # Prepare header and save
    header = nii.header.copy()
    header.set_data_dtype(np.uint8)

    brain_nii = nib.Nifti1Image(filtered, affine=nii.affine, header=header)

    basename = Path(file_path).stem
    out_path = os.path.join(out_dir, f"{basename}_region_extracted.nii.gz")
    nib.save(brain_nii, out_path)
    print(f"[NIfTI Extractor] Saved: {Path(out_path).name}")

    return out_path
