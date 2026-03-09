import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.freesurfer.io import read_geometry


# ---- CONFIG: set your filenames here ----
# Paths are relative to the repository root; adjust if your data
# lives elsewhere.
t2_path = (
    "assets/data/fetal_surface/"
    "25week_1day_sub-CC00923XX16_ses-11631/"
    "sub-CC00923XX16_ses-11631_T2w.nii.gz"
)
seg_path = (
    "assets/data/fetal_surface/"
    "25week_1day_sub-CC00923XX16_ses-11631/"
    "seg.nii.gz"
)
lh_pial_path = (
    "assets/data/fetal_surface/"
    "25week_1day_sub-CC00923XX16_ses-11631/lh.pial"
)
rh_pial_path = (
    "assets/data/fetal_surface/"
    "25week_1day_sub-CC00923XX16_ses-11631/rh.pial"
)
# ----------------------------------------


def load_vol(path: str):
    img = nib.load(path)
    data = img.get_fdata()
    aff = img.affine
    return img, data, aff


def frac_vertices_inside(verts_vox: np.ndarray, shape: tuple[int, int, int]) -> float:
    nx, ny, nz = shape
    x, y, z = verts_vox.T
    inside = (
        (x >= 0)
        & (x < nx - 1)
        & (y >= 0)
        & (y < ny - 1)
        & (z >= 0)
        & (z < nz - 1)
    )
    return float(inside.mean())


def main():
    # --- 1) Load volumes ---
    t2_img, t2_data, t2_aff = load_vol(t2_path)
    seg_img, seg_data, seg_aff = load_vol(seg_path)

    print("T2 shape:", t2_data.shape)
    print("SEG shape:", seg_data.shape)
    print("Max |T2_aff - SEG_aff|:", np.abs(t2_aff - seg_aff).max())

    # quick check: seg should be in same space as T2
    if t2_data.shape != seg_data.shape:
        print("WARNING: T2 and SEG shapes differ -> likely different grids.")
    if np.abs(t2_aff - seg_aff).max() > 1e-3:
        print("WARNING: T2 and SEG affines differ noticeably.")

    # --- 2) Load pial surfaces ---
    lh_verts, lh_faces = read_geometry(lh_pial_path)
    rh_verts, rh_faces = read_geometry(rh_pial_path)

    # Assume (for this basic check) that pial coords are in SAME RAS as T2
    t2_aff_inv = np.linalg.inv(t2_aff)
    lh_h = np.c_[lh_verts, np.ones(len(lh_verts))]
    rh_h = np.c_[rh_verts, np.ones(len(rh_verts))]

    lh_vox = (t2_aff_inv @ lh_h.T).T[:, :3]
    rh_vox = (t2_aff_inv @ rh_h.T).T[:, :3]

    lh_frac = frac_vertices_inside(lh_vox, t2_data.shape)
    rh_frac = frac_vertices_inside(rh_vox, t2_data.shape)

    print(f"Fraction of LH pial vertices inside T2 volume: {lh_frac:.3f}")
    print(f"Fraction of RH pial vertices inside T2 volume: {rh_frac:.3f}")

    # --- 3) Simple visual check on one slice (axial mid-slice) ---
    mid_z = t2_data.shape[2] // 2

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(t2_data[:, :, mid_z].T, cmap="gray", origin="lower")

    # plot vertices near this slice (within 0.5 voxel)
    dz = 0.5
    lh_mask = np.abs(lh_vox[:, 2] - mid_z) < dz
    rh_mask = np.abs(rh_vox[:, 2] - mid_z) < dz

    ax.scatter(
        lh_vox[lh_mask, 0],
        lh_vox[lh_mask, 1],
        s=1,
        alpha=0.5,
        label="LH pial",
    )
    ax.scatter(
        rh_vox[rh_mask, 0],
        rh_vox[rh_mask, 1],
        s=1,
        alpha=0.5,
        label="RH pial",
    )

    ax.set_title(f"Axial slice z={mid_z}")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
