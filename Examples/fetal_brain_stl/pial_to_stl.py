import nibabel as nib
import pyvista as pv
import numpy as np

def pial_to_stl(pial_path, stl_path):
    # Load FreeSurfer surface
    verts, faces = nib.freesurfer.read_geometry(pial_path)
    
    # faces from nibabel are Nx3 (no leading count per face)
    # PyVista expects faces as: [3, v0, v1, v2, 3, v0, v1, v2, ...]
    faces_pv = []
    for tri in faces:
        faces_pv.append(3)
        faces_pv.extend(tri)
    faces_pv = np.array(faces_pv)

    # Create PolyData
    mesh = pv.PolyData(verts, faces_pv)

    # Save to STL
    mesh.save(stl_path)
    print(f"Saved STL: {stl_path}")

# Example usage
pial_to_stl("rh.pial", "rh_pial.stl")
pial_to_stl("lh.pial", "lh_pial.stl")

rh_mesh = pv.read("rh_pial.stl")
lh_mesh = pv.read("lh_pial.stl")
combined = rh_mesh.merge(lh_mesh)
combined.save("brain_both.stl")
