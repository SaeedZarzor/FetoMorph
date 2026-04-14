"""Convert a NIfTI segmentation volume to an STL surface mesh.

Uses marching cubes to extract an iso-surface from selected label
regions and exports the result as an STL file via trimesh.
"""

from deps import *
from skimage import measure


def nii_to_stl(file_path: str, out_dir: str, valid_labels: set[int], level: float = 0.5):
    """Generate an STL mesh from selected regions of a NIfTI file.

    The volume is reoriented to closest-canonical (RAS+) form, a binary
    mask is built from *valid_labels*, and marching cubes extracts a
    triangulated surface that is saved as ``output.stl`` inside
    *out_dir*.

    If *valid_labels* is empty, a warning is printed and the raw voxel
    data is used directly (no label filtering).

    Args:
        file_path: Path to the input NIfTI file (.nii or .nii.gz).
        out_dir: Directory where ``output.stl`` will be written.
        valid_labels: Set of integer label values to include in the
            surface extraction.
        level: Iso-surface level passed to ``marching_cubes``.
            Defaults to 0.5.
    """

    img = nib.load(file_path)
    img = nib.as_closest_canonical(img)
    data = img.get_fdata()     # Get voxel data (3D NumPy array)


    if not valid_labels:
        print(" [NIfTI2STL] Warning: None of the selected regions are present in this NIfTI file!")
        threshold = np.percentile(data, 50)
        brain_mask = data #> threshold
    else:
        print("[NIfTI2STL] Extracting regions:", valid_labels)
        # Step 2: Create a Mask Including Only These Regions
        brain_mask = np.isin(data, list(valid_labels))

    
    verts, faces, normals, values = measure.marching_cubes(brain_mask, level=0.5)
    stl_mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

    stl_extracted = os.path.join(out_dir, "output.stl")
    stl_mesh.export(stl_extracted)


