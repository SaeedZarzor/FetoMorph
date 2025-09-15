import os
import nibabel as nib
from skimage import measure
import trimesh
import numpy as np


def nii_to_stl(file_path: str, out_dir: str, valid_labels: set[int],level: float = 0.5)

    img = nib.load(file_path)
    img = nib.as_closest_canonical(nifti_img)
    data = nifti_img.get_fdata()     # Get voxel data (3D NumPy array)


    if not valid_labels:
        print(" [NIfTI2STL] Warning: None of the selected regions are present in this NIfTI file!")
        threshold = np.percentile(data, 50)
        brain_mask = data #> threshold
    else:
        print("[NIfTI2STL] Extracting regions:", valid_labels)
        # Step 2: Create a Mask Including Only These Regions
        brain_mask = np.isin(data, list(valid_labels))

    
    verts, faces, normals, values = measure.marching_cubes(brain_mask, level=0.5)
    stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            stl_mesh.vectors[i][j] = verts[f[j], :]
    
    stl_extracted = os.path.join(out_dir, "output.stl")

    stl_mesh.save(stl_extracted)


