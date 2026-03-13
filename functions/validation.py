import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

def validate_nifti_area(out_dir: str):

    filename = "brain_extracted.nii.gz"
    file_path = os.path.join(out_dir, filename)
    if os.path.exists(file_path):
        nifti_img = nib.load(file_path)
        brain_mask = nifti_img.get_fdata()
        
        brain_pixel_counts = [np.sum(brain_mask[:, i, :]) for i in range(brain_mask.shape[1])]
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(brain_pixel_counts)), brain_pixel_counts, marker="o", linestyle="-", color="blue")
        plt.xlabel("Slice Number")
        plt.ylabel("Brain Pixel Count")
        plt.title("Brain Size Distribution Across Slices")
        plt.grid()
        brain_size_distribution = os.path.join(out_dir, "brain_size_distribution.png")
        plt.savefig(brain_size_distribution, dpi=300)
        
        print(f"[Validation] Saved size-distribution plot: {brain_size_distribution}")

        
        print("[NIfTI Area validation] ")
        
        return plt
    else:
        return

def validate_nifti_lGI (out_dir: str):
    filename = "Brain_lGI.xlsx"
    file_path = os.path.join(out_dir, filename)
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
        perimeters = df['Inner_Perimeter_mm'].to_numpy()   # as NumPy array
        convex_hull_perimeters = df['Outer_Perimeter_mm'].to_numpy()   # as NumPy array
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(convex_hull_perimeters)), convex_hull_perimeters, marker="o", linestyle="-", color="blue")
        plt.plot(range(len(convex_hull_perimeters)), perimeters, marker="*", linestyle="-", color="red")
        plt.xlabel("Slice Number")
        plt.ylabel("convex hull (blue) and perimeter (red) in mm")
        plt.title("Convex Hull vs. Perimeter Distribution Across Slices")
        plt.grid()
        brain_convex_perimeter = os.path.join(out_dir, "brain_convex_perimeter.png")
        plt.savefig(brain_convex_perimeter, dpi=300)

        print("[Validation] Saved size-distribution plot: {brain_convex_perimeter}")
        
        print("[NIfTI lGI validation] ")
        
        return plt
    else:
        return




