from deps import *
import nibabel as nib
from scipy.ndimage import binary_opening, binary_closing, label
from functions.Helpers import compute_kernel_convex, defect_mm_per_px_and_fixed



def compute_nifti_allmarks(file_path: str, out_dir: str, min_contour_area: float=30, kernel_size: int=5):
    nifti_img = nib.load(file_path)
    image_data = nifti_img.get_fdata()  # Get voxel data (3D NumPy array)
    affine = nifti_img.affine        # ✅ Needed for spatial info
    header = nifti_img.header        # ✅ Optional but keeps meta info
    # Get voxel size (in mm)
    voxel_size = nifti_img.header.get_zooms()  # (x, y, z) in mm

    # Extract pixel size
    pixel_size_x, pixel_size_y, pixel_size_z = voxel_size[:3]

    print(f"[NIfTI All hallmarks] voxel size: {pixel_size_x:.4f} x {pixel_size_y:.4f} x {pixel_size_z:.4f} mm")


    pixel_area_mm2 = pixel_size_x* pixel_size_z

    unique_labels = np.unique(image_data)
    print("[NIfTI All hallmarks] Unique labels in the image:", unique_labels)

    # Step 1: Define the Selected Regions
    selected_regions = {2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 17}  # The regions you want to include

    # Check if selected regions exist in the image
    valid_labels = selected_regions.intersection(set(unique_labels))
    if not valid_labels:
        print(" [NIfTI All hallmarks] Warning: None of the selected regions are present in this NIfTI file!")
        threshold = np.percentile(image_data, 50)
        brain_mask = image_data #> threshold
    else:
        print("[NIfTI All hallmarks] Extracting regions:", valid_labels)
        # Step 2: Create a Mask Including Only These Regions
        brain_mask = np.isin(image_data, list(valid_labels))


    brain_slices = np.where(np.any(brain_mask, axis=(0, 2)))[0]
    if brain_slices.size == 0:
        QMessageBox.information(self, "[NIfTI All hallmarks]", "No slices contain the selected mask.")
        return
        
   
    out_dir_slices = os.path.join(out_dir, "brain_slices")
    os.makedirs(out_dir_slices, exist_ok=True)
    print(f"[NIfTI All hallmarks] Temp output dir: {out_dir}")



    sheet1=[]
    sum_area = 0
    sum_inner,sum_outer = 0, 0
    valid_slices = []
    saved_pngs = []
    total_depth = []
    
    
    # Loop through all slices and save each as an image
    if len(brain_slices) > 0:
        for idx in brain_slices:  # Iterate over slices (z-dimension)
            slice_mask = brain_mask[:, idx, :].astype(np.uint8)  # Convert to uint8

            nonzero_count = np.count_nonzero(slice_mask)
#            if nonzero_count < 100:
#                print(f"[NIfTI lGI] Slice {idx} ignored: too few non-zero pixels ({nonzero_count})")
#                continue  # Skip this slice
#            else:
            valid_slices.append(idx)
            slice_area = float(np.sum(slice_mask) * pixel_area_mm2)
            sum_area += slice_area
                
            # Inner contour
            inner_contours, _ = cv2.findContours(slice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_contours = [cnt for cnt in inner_contours if cv2.contourArea(cnt) > min_contour_area]
            cnt_mm = [cnt.astype(np.float32) * [pixel_size_x, pixel_size_z] for cnt in filtered_contours]
            inner_perimeter = sum(cv2.arcLength(cnt, True) for cnt in cnt_mm)
                
            # Outer contour using morphological closing
            closed_mask = cv2.morphologyEx(slice_mask, cv2.MORPH_CLOSE, compute_kernel_convex(kernel_size))
            outer_contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_outer_contours = [cnt for cnt in outer_contours if cv2.contourArea(cnt) > min_contour_area]
            outer_cnt_mm = [cnt.astype(np.float32) * [pixel_size_x, pixel_size_z] for cnt in filtered_outer_contours]
            outer_perimeter = sum(cv2.arcLength(cnt, True) for cnt in outer_cnt_mm)
                
            # Save perimeters
            sum_inner += inner_perimeter
            sum_outer += outer_perimeter
            GI_slice = inner_perimeter / outer_perimeter if outer_perimeter > 0 else 0
                
            # Create a grayscale image for visualization
            annotated = np.stack([slice_mask * 255] * 3, axis=-1)  # Convert binary mask to RGB
            annotated = annotated.reshape((annotated.shape[0], annotated.shape[1], 3))
            cv2.drawContours(annotated, filtered_contours, -1, (0, 0, 255), 1)  # Red contours (original)
            cv2.drawContours(annotated, filtered_outer_contours, -1, (0, 255, 0), 1)  # green
            
            depth = []
            if filtered_contours:
                for cnt in filtered_contours:
                    hull = cv2.convexHull(cnt, returnPoints=False)  # Compute convex hull
                    if hull is not None and len(hull) >= 3 and len(cnt) >= 3:
                        defects = cv2.convexityDefects(cnt, hull)
                        if defects is not None:
                            for i in range(defects.shape[0]):
                                s, e, f, d = defects[i, 0]
                                start = tuple(cnt[s][0])
                                end = tuple(cnt[e][0])
                                far = tuple(cnt[f][0])
                                annotated = cv2.line(annotated, start, end, [255, 0, 0], 2)
                                if d>256:
                                    annotated = cv2.circle(annotated, far, 2, [0, 0, 255], -1)
                                    if pixel_size_x!=pixel_size_z:
                                        mm_per_px, mm_per_fixed = defect_mm_per_px_and_fixed(start, end, far, pixel_size_x, pixel_size_z)
                                    else:
                                        mm_per_fixed = pixel_size_x/256

                                    depth_mm = d *mm_per_fixed
                                    depth.append(depth_mm)
                
            mean_depth = (sum(depth)/len(depth)) if depth else None
            total_depth.extend(depth)
            sheet1.append([idx, slice_area, inner_perimeter, outer_perimeter,
                len(depth),                         # n_defects
                (min(depth) if depth else None),    # min_depth_mm
                (max(depth) if depth else None),    # max_depth_mm
                mean_depth                          # mean_depth_mm
                ])
                    
            slice_path = os.path.join(out_dir_slices, f"brain_slice_{idx:03d}.png")
            cv2.imwrite(slice_path, annotated)
            saved_pngs.append(slice_path)



#        brain_length_cm = ((valid_slices[-1] - valid_slices[0] + 1) * pixel_size_y) /10
        brain_volume = (sum_area * pixel_size_y)/1000
        Area = sum_inner * pixel_size_y /100
        GI_total = sum_inner / sum_outer if sum_outer > 0 else 0
        total_depth = [x/10 for x in total_depth]
        for i,v in enumerate(total_depth):
            total_depth[i] = v/10
    
        mean_total = (sum(total_depth)/ len(total_depth))  if total_depth else None
        
        sheet1.append(["Volume cm^3",round(brain_volume,2), "Surface Area cm^2",round(Area,2)])
        sheet1.append(["GI",round(GI_total,2)])
        sheet1.append(["Total_Number_of_Sluci_cz",len(total_depth), "Mean_value_across_slices_cm",round(mean_total, 2)])
        sheet1.append(["Max_sulci_across_slices_cm",round((max(total_depth) if total_depth else None),2),
        "Min_sulci_across_slices_cm",round((min(total_depth) if total_depth else None),2)])

        df = pd.DataFrame(sheet1, columns=["Slice", "Inner_area_mm^2" ,"Inner_Perimeter_mm", "Outer_Perimeter_mm", "Sulci_count", "min_depth_mm", "max_dpeth_mm", "mean_depth_mm"])
        xlsx_path = os.path.join(out_dir, "Brain_Allmarks.xlsx")
        df.to_excel(xlsx_path, index=False)
            

        # Step 3: Apply Mask & Save Extracted Brain NIfTI
        filtered_mask = brain_mask[:, valid_slices, :]
        brain_nii = nib.Nifti1Image(filtered_mask, affine=affine, header=header)
        
        brain_extracted = os.path.join(out_dir, "brain_extracted.nii.gz")
        nib.save(brain_nii, brain_extracted)
        print("[NIfTI All hallmarks] Brain-extracted NIfTI file saved as 'brain_extracted.nii.gz'")
        
        return Area, brain_volume, GI_total, total_depth ,saved_pngs, valid_slices

    else:
        QMessageBox.information(self, "[NIfTI All hallmarks]", "All slices were filtered out (too small).")
        return



def compute_nifti_volume(file_path: str, out_dir: str,):


    nifti_img = nib.load(file_path)
    image_data = nifti_img.get_fdata()  # Get voxel data (3D NumPy array)
    affine = nifti_img.affine        # ✅ Needed for spatial info
    header = nifti_img.header        # ✅ Optional but keeps meta info
    # Get voxel size (in mm)
    voxel_size = nifti_img.header.get_zooms()  # (x, y, z) in mm

    # Extract pixel size
    pixel_size_x, pixel_size_y, pixel_size_z = voxel_size[:3]

    print(f"[NIfTI Volume] voxel size: {pixel_size_x:.4f} x {pixel_size_y:.4f} x {pixel_size_z:.4f} mm")


    pixel_area_mm2 = pixel_size_x* pixel_size_z

    unique_labels = np.unique(image_data)
    print("[NIfTI Volume] Unique labels in the image:", unique_labels)

    # Step 1: Define the Selected Regions
    selected_regions = {3, 4, 5, 6, 14, 15, 16, 17}  # The regions you want to include

    # Check if selected regions exist in the image
    valid_labels = selected_regions.intersection(set(unique_labels))
    if not valid_labels:
        print(" [NIfTI Volume] Warning: None of the selected regions are present in this NIfTI file!")
        threshold = np.percentile(image_data, 50)
        brain_mask = image_data #> threshold
    else:
        print("[NIfTI Volume] Extracting regions:", valid_labels)
        # Step 2: Create a Mask Including Only These Regions
        brain_mask = np.isin(image_data, list(valid_labels))


    brain_slices = np.where(np.any(brain_mask, axis=(0, 2)))[0]
    if brain_slices.size == 0:
        QMessageBox.information(self, "[NIfTI Volume]", "No slices contain the selected mask.")
        return
        
   
    out_dir_slices = os.path.join(out_dir, "brain_slices")
    os.makedirs(out_dir_slices, exist_ok=True)
    print(f"[NIfTI Volume] Temp output dir: {out_dir}")


    sheet1=[]

    sum_area = 0
    valid_slices = []
    saved_pngs = []


    # Loop through all slices and save each as an image
    if len(brain_slices) > 0:
        for idx in brain_slices:  # Iterate over slices (z-dimension)
            slice_mask = brain_mask[:, idx, :].astype(np.uint8)  # Convert to uint8

            nonzero_count = np.count_nonzero(slice_mask)
#            if nonzero_count < 100:
#                print(f"[NIfTI Volume] Slice {idx} ignored: too few non-zero pixels ({nonzero_count})")
#                continue  # Skip this slice
#            else:
            valid_slices.append(idx)
                
            slice_area = float(np.sum(slice_mask) * pixel_area_mm2)
            sum_area += slice_area
                
            annotated = np.stack([slice_mask * 255] * 3, axis=-1)  # Convert binary mask to RGB
            annotated = annotated.reshape((annotated.shape[0], annotated.shape[1], 3))

                
            sheet1.append([idx, slice_area])
                    
            slice_path = os.path.join(out_dir_slices, f"brain_slice_{idx:03d}.png")
            cv2.imwrite(slice_path, annotated)
            saved_pngs.append(slice_path)



        brain_volume = (sum_area * pixel_size_y)/1000


        sheet1.append(["Volume cm^3",round(brain_volume,2)])

        df = pd.DataFrame(sheet1, columns=["Slice", "Inner_area_mm^2", ])
        xlsx_path = os.path.join(out_dir, "Brain_Volume.xlsx")
        df.to_excel(xlsx_path, index=False)
            

        # Step 3: Apply Mask & Save Extracted Brain NIfTI
        filtered_mask = brain_mask[:, valid_slices, :]
    #    brain_extracted = image_data * filtered_mask
        brain_nii = nib.Nifti1Image(filtered_mask, affine=affine, header=header)
        
        brain_extracted = os.path.join(out_dir, "brain_extracted.nii.gz")
        nib.save(brain_nii, brain_extracted)
        print("[NIfTI Volume] Brain-extracted NIfTI file saved as 'brain_extracted.nii.gz'")
        
        return brain_volume, saved_pngs, valid_slices

    else:
        QMessageBox.information(self, "[NIfTI Volume]", "All slices were filtered out (too small).")
        return




def compute_nifti_arae(file_path: str, out_dir: str, min_contour_area: float=30,):


    nifti_img = nib.load(file_path)
    image_data = nifti_img.get_fdata()  # Get voxel data (3D NumPy array)
    affine = nifti_img.affine        # ✅ Needed for spatial info
    header = nifti_img.header        # ✅ Optional but keeps meta info
    # Get voxel size (in mm)
    voxel_size = nifti_img.header.get_zooms()  # (x, y, z) in mm

    # Extract pixel size
    pixel_size_x, pixel_size_y, pixel_size_z = voxel_size[:3]

    print(f"[NIfTI Area] voxel size: {pixel_size_x:.4f} x {pixel_size_y:.4f} x {pixel_size_z:.4f} mm")


    pixel_area_mm2 = pixel_size_x* pixel_size_z

    unique_labels = np.unique(image_data)
    print("[NIfTI Area] Unique labels in the image:", unique_labels)

    # Step 1: Define the Selected Regions
    selected_regions = {3, 4, 5, 6, 14, 15, 16, 17}  # The regions you want to include

    # Check if selected regions exist in the image
    valid_labels = selected_regions.intersection(set(unique_labels))
    if not valid_labels:
        print(" [NIfTI Area] Warning: None of the selected regions are present in this NIfTI file!")
        threshold = np.percentile(image_data, 50)
        brain_mask = image_data #> threshold
    else:
        print("[NIfTI Area] Extracting regions:", valid_labels)
        # Step 2: Create a Mask Including Only These Regions
        brain_mask = np.isin(image_data, list(valid_labels))


    brain_slices = np.where(np.any(brain_mask, axis=(0, 2)))[0]
    if brain_slices.size == 0:
        QMessageBox.information(self, "[NIfTI Area]", "No slices contain the selected mask.")
        return
        
   
    out_dir_slices = os.path.join(out_dir, "brain_slices")
    os.makedirs(out_dir_slices, exist_ok=True)
    print(f"[NIfTI Area] Temp output dir: {out_dir}")
    
    sheet1=[]

    sum_inner = 0
    valid_slices = []
    saved_pngs = []


    # Loop through all slices and save each as an image
    if len(brain_slices) > 0:
        for idx in brain_slices:  # Iterate over slices (z-dimension)
            slice_mask = brain_mask[:, idx, :].astype(np.uint8)  # Convert to uint8

            nonzero_count = np.count_nonzero(slice_mask)
#            if nonzero_count < 100:
#                print(f"[NIfTI lGI] Slice {idx} ignored: too few non-zero pixels ({nonzero_count})")
#                continue  # Skip this slice
#            else:
            valid_slices.append(idx)

                
            # Inner contour
            inner_contours, _ = cv2.findContours(slice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_contours = [cnt for cnt in inner_contours if cv2.contourArea(cnt) > min_contour_area]
            cont_mm = [cnt.astype(np.float32) * [pixel_size_x, pixel_size_z] for cnt in filtered_contours]
            inner_perimeter = sum(cv2.arcLength(cnt, True) for cnt in cont_mm)
                

                
            # Save perimeters
            sum_inner += inner_perimeter
                
            # Create a grayscale image for visualization
            annotated = np.stack([slice_mask * 255] * 3, axis=-1)  # Convert binary mask to RGB
            annotated = annotated.reshape((annotated.shape[0], annotated.shape[1], 3))
            cv2.drawContours(annotated, filtered_contours, -1, (0, 0, 255), 1)  # Red contours (original)
                            
            sheet1.append([idx, inner_perimeter])
                    
            slice_path = os.path.join(out_dir_slices, f"brain_slice_{idx:03d}.png")
            cv2.imwrite(slice_path, annotated)
            saved_pngs.append(slice_path)


        Area = sum_inner * pixel_size_y/ 100


        sheet1.append(["Surface Area cm^2",round(Area,2)])

        df = pd.DataFrame(sheet1, columns=["Slice", "Inner_Perimeter_mm"])
        xlsx_path = os.path.join(out_dir, "Brain_Surface_Area.xlsx")
        df.to_excel(xlsx_path, index=False)
            

        # Step 3: Apply Mask & Save Extracted Brain NIfTI
        filtered_mask = brain_mask[:, valid_slices, :]
    #    brain_extracted = image_data * filtered_mask
        brain_nii = nib.Nifti1Image(filtered_mask, affine=affine, header=header)
        
        brain_extracted = os.path.join(out_dir, "brain_extracted.nii.gz")
        nib.save(brain_nii, brain_extracted)
        print("[NIfTI Area] Brain-extracted NIfTI file saved as 'brain_extracted.nii.gz'")
        
        return Area, saved_pngs, valid_slices

    else:
        QMessageBox.information(self, "[NIfTI Area]", "All slices were filtered out (too small).")
        return
    
    

def compute_nifti_lGI(file_path: str, out_dir: str, min_contour_area: float=30, kernel_size: int=5):
    nifti_img = nib.load(file_path)
    image_data = nifti_img.get_fdata()  # Get voxel data (3D NumPy array)
    affine = nifti_img.affine        # ✅ Needed for spatial info
    header = nifti_img.header        # ✅ Optional but keeps meta info
    # Get voxel size (in mm)
    voxel_size = nifti_img.header.get_zooms()  # (x, y, z) in mm

    # Extract pixel size
    pixel_size_x, pixel_size_y, pixel_size_z = voxel_size[:3]

    print(f"[NIfTI lGI] voxel size: {pixel_size_x:.4f} x {pixel_size_y:.4f} x {pixel_size_z:.4f} mm")


    pixel_area_mm2 = pixel_size_x* pixel_size_z

    unique_labels = np.unique(image_data)
    print("[NIfTI lGI] Unique labels in the image:", unique_labels)

    # Step 1: Define the Selected Regions
    selected_regions = {2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 17}  # The regions you want to include

    # Check if selected regions exist in the image
    valid_labels = selected_regions.intersection(set(unique_labels))
    if not valid_labels:
        print(" [NIfTI lGI] Warning: None of the selected regions are present in this NIfTI file!")
        threshold = np.percentile(image_data, 50)
        brain_mask = image_data #> threshold
    else:
        print("[NIfTI lGI] Extracting regions:", valid_labels)
        # Step 2: Create a Mask Including Only These Regions
        brain_mask = np.isin(image_data, list(valid_labels))


    brain_slices = np.where(np.any(brain_mask, axis=(0, 2)))[0]
    if brain_slices.size == 0:
        QMessageBox.information(self, "[NIfTI lGI]", "No slices contain the selected mask.")
        return
        
   
    out_dir_slices = os.path.join(out_dir, "brain_slices")
    os.makedirs(out_dir_slices, exist_ok=True)
    print(f"[NIfTI lGI] Temp output dir: {out_dir}")



    sheet1=[]

    sum_inner,sum_outer = 0, 0
    valid_slices = []
    saved_pngs = []
    
    
    # Loop through all slices and save each as an image
    if len(brain_slices) > 0:
        for idx in brain_slices:  # Iterate over slices (z-dimension)
            slice_mask = brain_mask[:, idx, :].astype(np.uint8)  # Convert to uint8

            nonzero_count = np.count_nonzero(slice_mask)
#            if nonzero_count < 100:
#                print(f"[NIfTI lGI] Slice {idx} ignored: too few non-zero pixels ({nonzero_count})")
#                continue  # Skip this slice
#            else:
            valid_slices.append(idx)
                
#                slice_area = float(np.sum(slice_mask) * pixel_area_mm2)
#                sum_area += slice_area
                
            # Inner contour
            inner_contours, _ = cv2.findContours(slice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_contours = [cnt for cnt in inner_contours if cv2.contourArea(cnt) > min_contour_area]
            cnt_mm = [cnt.astype(np.float32) * [pixel_size_x, pixel_size_z] for cnt in filtered_contours]
            inner_perimeter = sum(cv2.arcLength(cnt, True) for cnt in cnt_mm)
                
            # Outer contour using morphological closing
            closed_mask = cv2.morphologyEx(slice_mask, cv2.MORPH_CLOSE, compute_kernel_convex(kernel_size))
            outer_contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_outer_contours = [cnt for cnt in outer_contours if cv2.contourArea(cnt) > min_contour_area]
            outer_cnt_mm = [cnt.astype(np.float32) * [pixel_size_x, pixel_size_z] for cnt in filtered_outer_contours]
            outer_perimeter = sum(cv2.arcLength(cnt, True) for cnt in outer_cnt_mm)
                
            # Save perimeters
            sum_inner += inner_perimeter
                
            sum_outer += outer_perimeter
            GI_slice = inner_perimeter / outer_perimeter if outer_perimeter > 0 else 0
                
            # Create a grayscale image for visualization
            annotated = np.stack([slice_mask * 255] * 3, axis=-1)  # Convert binary mask to RGB
            annotated = annotated.reshape((annotated.shape[0], annotated.shape[1], 3))
            cv2.drawContours(annotated, filtered_contours, -1, (0, 0, 255), 1)  # Red contours (original)
            cv2.drawContours(annotated, filtered_outer_contours, -1, (0, 255, 0), 1)  # green


                
    
#                    all_points = np.vstack(contours_mm)  # Stack all contour points
#                    hull = cv2.convexHull(all_points)  # Compute convex hull
#                    perimeter_convex = cv2.arcLength(hull, True)  # Perimeter in mm
#                    sum_perimeter_conv += perimeter_convex  # Sum total convex perimeters
                    
                    
#                    # Compute Perimeter Rate
#                    perimeter_rate = slice_perimeter / perimeter_convex if perimeter_convex > 0 else 0
#                    perimeters.append(slice_perimeter)  # Save to array
#                    convex_hull_perimeters.append(perimeter_convex)  # Save to array
#    
#                    hull_draw= cv2.convexHull(np.vstack(filtered_contours))
#                    # Draw contours (original in red, convex hull in green)
#                    cv2.drawContours(contour_image, [hull_draw], -1, (0, 255, 0), 1)  # Green convex hull

                
            sheet1.append([idx, inner_perimeter, outer_perimeter])
                    
            slice_path = os.path.join(out_dir_slices, f"brain_slice_{idx:03d}.png")
            cv2.imwrite(slice_path, annotated)
            saved_pngs.append(slice_path)



#        brain_length_cm = ((valid_slices[-1] - valid_slices[0] + 1) * pixel_size_y) /10
#        brain_volume = (sum_area * pixel_size_y)/1000
        GI_total = sum_inner / sum_outer if sum_outer > 0 else 0


        sheet1.append(["GI",round(GI_total,2)])

        df = pd.DataFrame(sheet1, columns=["Slice", "Inner_Perimeter_mm", "Outer_Perimeter_mm"])
        xlsx_path = os.path.join(out_dir, "Brain_lGI.xlsx")
        df.to_excel(xlsx_path, index=False)
            

        # Step 3: Apply Mask & Save Extracted Brain NIfTI
        filtered_mask = brain_mask[:, valid_slices, :]
    #    brain_extracted = image_data * filtered_mask
        brain_nii = nib.Nifti1Image(filtered_mask, affine=affine, header=header)
        
        brain_extracted = os.path.join(out_dir, "brain_extracted.nii.gz")
        nib.save(brain_nii, brain_extracted)
        print("[NIfTI lGI] Brain-extracted NIfTI file saved as 'brain_extracted.nii.gz'")
        
        return GI_total, saved_pngs, valid_slices

    else:
        QMessageBox.information(self, "[NIfTI lGI]", "All slices were filtered out (too small).")
        return


        
def compute_nifti_sulci_depth(file_path: str, out_dir: str, min_contour_area: float=30,):
    nifti_img = nib.load(file_path)
    image_data = nifti_img.get_fdata()  # Get voxel data (3D NumPy array)
    affine = nifti_img.affine        # ✅ Needed for spatial info
    header = nifti_img.header        # ✅ Optional but keeps meta info
    # Get voxel size (in mm)
    voxel_size = nifti_img.header.get_zooms()  # (x, y, z) in mm

    # Extract pixel size
    pixel_size_x, pixel_size_y, pixel_size_z = voxel_size[:3]

    print(f"[NIfTI Sulci depth] voxel size: {pixel_size_x:.4f} x {pixel_size_y:.4f} x {pixel_size_z:.4f} mm")


    pixel_area_mm2 = pixel_size_x* pixel_size_z

    unique_labels = np.unique(image_data)
    print("[NIfTI Sulci depth] Unique labels in the image:", unique_labels)

    # Step 1: Define the Selected Regions
    selected_regions = {2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 17}  # The regions you want to include

    # Check if selected regions exist in the image
    valid_labels = selected_regions.intersection(set(unique_labels))
    if not valid_labels:
        print(" [NIfTI Sulci depth] Warning: None of the selected regions are present in this NIfTI file!")
        threshold = np.percentile(image_data, 50)
        brain_mask = image_data #> threshold
    else:
        print("[NIfTI Sulci depth] Extracting regions:", valid_labels)
        # Step 2: Create a Mask Including Only These Regions
        brain_mask = np.isin(image_data, list(valid_labels))


    brain_slices = np.where(np.any(brain_mask, axis=(0, 2)))[0]
    if brain_slices.size == 0:
        QMessageBox.information(self, "[NIfTI Sulci depth]", "No slices contain the selected mask.")
        return
        
   
    out_dir_slices = os.path.join(out_dir, "brain_slices")
    os.makedirs(out_dir_slices, exist_ok=True)
    print(f"[NIfTI Sulci depth] Temp output dir: {out_dir}")



    sheet1=[]
    valid_slices = []
    saved_pngs = []
    total_depth = []
    
    
    # Loop through all slices and save each as an image
    if len(brain_slices) > 0:
        for idx in brain_slices:  # Iterate over slices (z-dimension)
            slice_mask = brain_mask[:, idx, :].astype(np.uint8)  # Convert to uint8

            nonzero_count = np.count_nonzero(slice_mask)
#            if nonzero_count < 100:
#                print(f"[NIfTI lGI] Slice {idx} ignored: too few non-zero pixels ({nonzero_count})")
#                continue  # Skip this slice
#            else:
            valid_slices.append(idx)

                
            # Inner contour
            inner_contours, _ = cv2.findContours(slice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_contours = [cnt for cnt in inner_contours if cv2.contourArea(cnt) > min_contour_area]
            cnt_mm = [cnt.astype(np.float32) * [pixel_size_x, pixel_size_z] for cnt in filtered_contours]
                
            # Create a grayscale image for visualization
            annotated = np.stack([slice_mask * 255] * 3, axis=-1)  # Convert binary mask to RGB
            annotated = annotated.reshape((annotated.shape[0], annotated.shape[1], 3))
            cv2.drawContours(annotated, filtered_contours, -1, (0, 0, 255), 1)  # Red contours (original)
                
            depth = []
            if filtered_contours:
                for cnt in filtered_contours:
                    hull = cv2.convexHull(cnt, returnPoints=False)  # Compute convex hull
                    if hull is not None and len(hull) >= 3 and len(cnt) >= 3:
                        defects = cv2.convexityDefects(cnt, hull)
                        if defects is not None:
                            for i in range(defects.shape[0]):
                                s, e, f, d = defects[i, 0]
                                start = tuple(cnt[s][0])
                                end = tuple(cnt[e][0])
                                far = tuple(cnt[f][0])
                                annotated = cv2.line(annotated, start, end, [255, 0, 0], 2)
                                if d>256:
                                    annotated = cv2.circle(annotated, far, 2, [0, 255, 0], -1)
                                    if pixel_size_x!=pixel_size_z:
                                        mm_per_px, mm_per_fixed = defect_mm_per_px_and_fixed(start, end, far, pixel_size_x, pixel_size_z)
                                    else:
                                        mm_per_fixed = pixel_size_x/256

                                    depth_mm = d *mm_per_fixed
                                    depth.append(depth_mm)

            total_depth.extend(depth)
            mean_depth = (sum(depth)/len(depth)) if depth else None
            
            sheet1.append([idx,
                len(depth),                         # n_defects
                (min(depth) if depth else None),    # min_depth_mm
                (max(depth) if depth else None),    # max_depth_mm
                mean_depth                          # mean_depth_mm
                ])
                    
            slice_path = os.path.join(out_dir_slices, f"brain_slice_{idx:03d}.png")
            cv2.imwrite(slice_path, annotated)
            saved_pngs.append(slice_path)
        
        mean_total = (sum(total_depth)/ len(total_depth))  if total_depth else None
        sheet1.append(["Total_Number_of_Sluci",len(total_depth), "Mean_value_across_slices",round(mean_total, 2)])
        sheet1.append(["Max_sulci_across_slices",round((max(total_depth) if total_depth else None),2),
        "Min_sulci_across_slices",round((min(total_depth) if total_depth else None),2)])

        df = pd.DataFrame(sheet1, columns=["Slice","Sulci_count", "min_depth_mm", "max_dpeth_mm", "mean_depth_mm"])
        xlsx_path = os.path.join(out_dir, "Brain_Sulci.xlsx")
        df.to_excel(xlsx_path, index=False)
            

        # Step 3: Apply Mask & Save Extracted Brain NIfTI
        filtered_mask = brain_mask[:, valid_slices, :]
        brain_nii = nib.Nifti1Image(filtered_mask, affine=affine, header=header)
        
        brain_extracted = os.path.join(out_dir, "brain_extracted.nii.gz")
        nib.save(brain_nii, brain_extracted)
        print("[NIfTI Slice] Brain-extracted NIfTI file saved as 'brain_extracted.nii.gz'")
        
        return  total_depth, saved_pngs, valid_slices

    else:
        QMessageBox.information(self, "[NIfTI Sulci depth]", "All slices were filtered out (too small).")
        return



