import time
import uuid
try:
    import nibabel as nib
    import pyvista as pv
    import numpy as np
except Exception as ex:
    print("[FreeSurfer] Missing dependency. Install with:\n  pip install nibabel pyvista")
    raise


def _pial_to_pv_polydata(pial_path: str):
    t0 = time.time()
    print(f"Reading geometry: {pial_path}")
    verts, faces = nib.freesurfer.read_geometry(pial_path)  # faces: (M, 3)

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Unexpected faces array shape from nibabel (expected Nx3).")

    ntri = faces.shape[0]
    faces_pv = np.column_stack(
        (np.full((ntri, 1), 3, dtype=np.int64), faces.astype(np.int64))
    ).ravel(order="C")

    pv.PolyData.use_strict_n_faces(True)
    mesh = pv.PolyData(verts, faces_pv)
    mesh.clean(inplace=True)

    # robust face count
    face_count = getattr(mesh, "n_faces_strict", mesh.n_cells)

    dt = time.time() - t0
    print(f"Loaded mesh: points={mesh.n_points:,}  faces={face_count:,}")
    return mesh



def pial_to_stl(pial_path: str, temp_path: str):
    """Convert a single .pial to STL and save (returns stl_path)."""
    print(f"Converting to STL ...")
    t0 = time.time()
    mesh = _pial_to_pv_polydata(pial_path)
    mesh.save(temp_path)
    dt = time.time() - t0
    print(f"Saved STL: points={mesh.n_points:,}, faces={mesh.n_faces:,}")
    return temp_path


def pial_pair_to_combined_stl(rh_pial: str, lh_pial: str, out_stl: str):
    """Convert rh & lh .pial to a single merged STL and save (returns out_stl)."""
    print(f"Converting pair:\n  RH: {rh_pial}\n  LH: {lh_pial}")
    t0 = time.time()
    m_rh = _pial_to_pv_polydata(rh_pial)
    m_lh = _pial_to_pv_polydata(lh_pial)

    print("Merging hemispheres ...")
    combined = m_rh.merge(m_lh)
    combined.clean(inplace=True)

    combined.save(out_stl)
    dt = time.time() - t0
    print(f"Saved combined STL: "
          f"points={combined.n_points:,}, faces={combined.n_faces:,}")
    return out_stl
