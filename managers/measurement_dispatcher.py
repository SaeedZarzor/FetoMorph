"""Measurement dispatcher -- all measurement/processing operations.

Extracted from MainWindow to consolidate measurement logic.
"""

from __future__ import annotations

import os
import time
import uuid
import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np
from deps import QColor, QFileDialog, QMessageBox, QPixmap, pd
from functions.Nifti2image import draw_new_scale_bar
from functions.curvature import compute_curvature_profile, save_curvature_plot
from functions.hausdorff import calculate_hausdorff_distance
from functions.measurement_Batch import process_on_images_batch
from functions.measurements_Nifti import *
from functions.measurements_image import *
from functions.measurements_stl import *
from functions.measurements_vtk import *
from functions.optimization import optimization
from functions.pial_to_stl import pial_pair_to_combined_stl, pial_to_stl
from helpers.Helpers import compactness_2D, compactness_3D
from helpers.Read_Excel import conver_excel
from managers.metrics_store import MetricsStore
from managers.view_manager import ViewManager
from widgets.optimization_widgets import OptimizationOptionsDialog

if TYPE_CHECKING:
    from FetoMorph import MainWindow

logger = logging.getLogger("fetomorph")


class MeasurementDispatcher:
    """Dispatches measurement operations; reads all state from MainWindow."""

    def __init__(self, mw: MainWindow) -> None:
        self.mw = mw

    def _measure_planar_vtk(self, mode: str = "allmarks"):
        """Measure a planar VTK mesh by capturing a 2D screenshot and running image measurements.

        Args:
            mode: One of "allmarks", "perimeter", "area", "lGI", "sulci_depth".
        """
        import pyvista as pv

        t0 = time.time()
        try:
            # Ensure geometry
            if all(v == 0 for v in self.mw.physical_dim):
                self.mw.settings.load_mesh_and_ask_geometry()

            u = self.mw.units_length

            # Read mesh bounds
            import pyvista as pv
            mesh = pv.read(str(self.mw.current_path))
            xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
            mesh_dim = (xmax - xmin, ymax - ymin, zmax - zmin)

            # Determine camera vertical axis from _flat_axis
            flat = self.mw._flat_axis
            if flat == 0:       # flat in X → camera looks along X, vertical=Y, horizontal=Z
                vert_axis, horiz_axis = 1, 2
            elif flat == 1:     # flat in Y → camera looks along Y, vertical=Z, horizontal=X
                vert_axis, horiz_axis = 2, 0
            else:               # flat in Z → camera looks along Z, vertical=Y, horizontal=X
                vert_axis, horiz_axis = 1, 0

            # Capture screenshot
            bgr, world_per_px = self.mw.vtk_view.capture_polydata2d_screenshot()

            # Compute pixel_size in physical units
            md = mesh_dim[vert_axis]
            if md < 1e-12:
                md = mesh_dim[horiz_axis]
            scale_factor = self.mw.physical_dim[vert_axis] / max(md, 1e-12)
            pixel_size = world_per_px * scale_factor

            # Save clean image to temp for measurement functions
            uid = uuid.uuid4().hex[:8]
            out_dir = os.path.join(self.mw.temp_dir, f"planar_vtk_{mode}_{uid}")
            os.makedirs(out_dir, exist_ok=True)
            self.mw.current_output_dir = out_dir
            name = os.path.splitext(os.path.basename(self.mw.current_path))[0]
            img_path = os.path.join(out_dir, f"{name}.png")
            cv2.imwrite(img_path, bgr)

            # Call measurement function
            if mode == "allmarks":
                area, perimeter, perimeter_convex, lGI, compactness, depth, annotated_bgr = compute_image_allmarks(
                    img_path, pixel_size=pixel_size, kernel_size=self.mw.kernel_size,
                    cnt_threshold=self.mw.cnt_threshold, unit=u, add_scalebar=False,
                    draw_hallmarks=self.mw.draw_hallmarks_on_image)
            elif mode == "perimeter":
                perimeter, annotated_bgr = compute_image_perimeter(
                    img_path, pixel_size=pixel_size, cnt_threshold=self.mw.cnt_threshold, unit=u, add_scalebar=False,
                    draw_hallmarks=self.mw.draw_hallmarks_on_image)
            elif mode == "area":
                area, annotated_bgr = compute_image_area(
                    img_path, pixel_size=pixel_size, cnt_threshold=self.mw.cnt_threshold, unit=u, add_scalebar=False,
                    draw_hallmarks=self.mw.draw_hallmarks_on_image)
            elif mode == "lGI":
                lGI, perimeter, perimeter_convex, annotated_bgr = compute_image_lGI(
                    img_path, pixel_size=pixel_size, kernel_size=self.mw.kernel_size,
                    cnt_threshold=self.mw.cnt_threshold, unit=u, add_scalebar=False,
                    draw_hallmarks=self.mw.draw_hallmarks_on_image)
            elif mode == "sulci_depth":
                depth, annotated_bgr = compute_image_sulci_depth(
                    img_path, pixel_size=pixel_size, cnt_threshold=self.mw.cnt_threshold, unit=u, add_scalebar=False)
            else:
                print(f"[Planar VTK] Unknown mode: {mode}")
                return

            # Add scale bar to annotated image
            image_width_phys = bgr.shape[1] * pixel_size
            target = image_width_phys * 0.2
            magnitude = 10 ** int(np.floor(np.log10(max(target, 1e-9))))
            bar_phys = next((magnitude * n for n in [1, 2, 5, 10] if magnitude * n >= target * 0.7), magnitude * 10)
            bar_px = int(round(bar_phys / pixel_size))
            annotated_bgr = draw_new_scale_bar(annotated_bgr, bar_px, text=f"{bar_phys:g} {u}")

            # Save annotated image to temp
            annotated_path = os.path.join(out_dir, "annotated.png")
            cv2.imwrite(annotated_path, annotated_bgr)

            # Register scale so re-measurement works without prompting
            self.mw.image_scales[img_path] = pixel_size
            self.mw.pixel_size = pixel_size

            # Display
            pm = ViewManager.np_bgr_to_qpixmap(annotated_bgr)
            self.mw.image_label.setImage(pm)
            self.mw.image_label.remove_last_annotation()
            self.mw.view.show_widget(self.mw.image_label)
            self.mw._active_view = "image"
            self.mw._set_current("image", img_path)

            # Record metrics and print
            if mode == "allmarks":
                self.mw.metrics_store.record_metric_for(img_path, unit=u, dimensions=self.mw.physical_dim,
                    kernel_size=self.mw.kernel_size, area=area, perimeter=perimeter,
                    perimeter_convex=perimeter_convex, lgi=lGI, compactness=compactness, sulci_depth=depth)
                print(f"[Planar VTK allmarks] area={area:.2f} {u}^2, perimeter={perimeter:.2f} {u}, GI={lGI:.2f}")
                print(f"[Planar VTK allmarks] Maximum Sulci Depth = {MetricsStore.depth_summary(depth, u)}")

            elif mode == "perimeter":
                self.mw.metrics_store.record_metric_for(img_path, unit=u, dimensions=self.mw.physical_dim, perimeter=perimeter)
                print(f"[Planar VTK perimeter] perimeter={perimeter:.2f} {u}")
            elif mode == "area":
                self.mw.metrics_store.record_metric_for(img_path, unit=u, dimensions=self.mw.physical_dim, area=area)
                print(f"[Planar VTK area] area={area:.2f} {u}^2")
            elif mode == "lGI":
                self.mw.metrics_store.record_metric_for(img_path, unit=u, dimensions=self.mw.physical_dim,
                    kernel_size=self.mw.kernel_size, lgi=lGI)
                print(f"[Planar VTK lGI] GI={lGI:.2f}")
            elif mode == "compactness":
                self.mw.metrics_store.record_metric_for(img_path, unit=u, dimensions=self.mw.physical_dim,
                    kernel_size=self.mw.kernel_size, compactness=compactness)
                print(f"[Planar VTK compactness] Compactness={compactness:.2f}")
            elif mode == "sulci_depth":
                self.mw.metrics_store.record_metric_for(img_path, unit=u, dimensions=self.mw.physical_dim, sulci_depth=depth)
                if isinstance(depth, (list, tuple)) and len(depth) > 0:
                    summary = ", ".join(f"{float(v):.2f}" for v in depth[:3])
                    print(f"[Planar VTK sulci depth] Maximum depths = {MetricsStore.depth_summary(depth, u)}")

            dt = time.time() - t0
            print(f"[Planar VTK {mode}] Done in {dt:.2f}s.")

        except Exception as ex:
            logger.error("Planar VTK {mode} failed: %s", ex)
            QMessageBox.critical(self, f"Planar VTK {mode} Failed", f"{type(ex).__name__}: {ex}")

    # ---------- Process menu (stubs) ----------
    def on_measure_allmarks(self):
        """Process → Measures → All hallmarks: compute and show annotated result WITHOUT saving."""
        if not self.mw.current_path or not os.path.isfile(self.mw.current_path):
            print("[All hallmarks] No image file is loaded."); return
        if self.mw.current_kind == "image":
            try:
                result = self.mw.settings.ensure_calibrated()
                if result is None:
                    return
                u, px_size = result

                print(f"[All hallmarks] Measuring: {self.mw.current_path}")
                print(f"[All hallmarks] Measuring with pixel size = {px_size} {u}/pixel")

                image_path = self.mw.current_path
                if self.mw.last_annotated_path is not None:
                    image_path = self.mw.last_annotated_path
                    
                area, perimeter, perimeter_convex, lGI, compactness, depth, annotated_bgr = compute_image_allmarks(
                    image_path,
                    pixel_size=px_size,
                    kernel_size= self.mw.kernel_size,
                    cnt_threshold=self.mw.cnt_threshold,
                    unit = u,
                    add_scalebar=not bool(self.mw.image_scale_from_scalebar.get(self.mw.current_path, False)),
                    draw_hallmarks=self.mw.draw_hallmarks_on_image,
                )
                
                print(f"[All hallmarks] Results:")
                print(f"Annotated area = {area:.2f} {u}^2.")
                print(f"Annotated Perimeter = {perimeter:.2f} {u}.")
                print(f"Convex Perimeter = {perimeter_convex:.2f} {u}.")
                print(f"LGI (Convex Perimeter/ Perimeter) = {lGI:.2f} .")
                print(f"Compactness = {compactness:.2f} .")
                print(f"Maximum Sulci Depth = {MetricsStore.depth_summary(depth, u)}")
                
                # Convert BGR ndarray → QPixmap and show (no disk write)
                label_text = self.mw.get_label_for_cropped_path(image_path)
                if label_text:
                    annotated_bgr = put_label_on_bgr(annotated_bgr, label_text, pos="topleft")


                pm = ViewManager.np_bgr_to_qpixmap(annotated_bgr)
                self.mw.image_label.setImage(pm)
                
                self.mw.image_label.remove_last_annotation()
                self.mw.view.show_widget(self.mw.image_label)
                # Keep kind/path as the ORIGINAL file; Save View As… will ask user where to save what they see.
                self.mw._active_view = "image"
                # Ensure File/Process actions stay enabled
                self.mw._set_current("image", self.mw.current_path)
                self.mw.metrics_store.record_metric_for(
                    self.mw.current_path,
                    annotation = label_text,
                    pixel_size_units = f"{self.mw.units_length}/pixel",
                    unit = self.mw.units_length,
                    pixel_size = self.mw.pixel_size,
                    kernel_size = self.mw.kernel_size,
                    area=area,
                    perimeter=perimeter,
                    perimeter_convex = perimeter_convex,
                    lgi=lGI,
                    compactness=compactness,
                    sulci_depth = depth)
                    
            except Exception as ex:
                logger.error("All hallmarks failed: %s", ex)
                QMessageBox.critical(self, "All hallmarks Failed", f"{type(ex).__name__}: {ex}")
        elif self.mw.current_kind == "nifti":
            t0 = time.time()
            try:
                nif_path = self.mw.current_path
                print(f"[NIfTI] Computing area/perimeter from: {nif_path}")

                uid = uuid.uuid4().hex[:8]
                out_dir = os.path.join(self.mw.temp_dir, f"nifti_allmarks_{uid}")
                os.makedirs(out_dir, exist_ok=True)
                
                self.mw.current_output_dir = out_dir
                
                labels = self.mw.nifti_selected_regions if self.mw.nifti_selected_regions else self.mw.labels_available
                dims, area, volume, gi, depth, saved_pngs, valid_slices = compute_nifti_allmarks(self, file_path=nif_path,
                out_dir=out_dir,valid_labels = labels, min_contour_area=self.mw.cnt_threshold, kernel_size = self.mw.kernel_size)
            
                if area is None:
                    return

                # record metrics (consistent with your global export; units in mm unless noted)
                self.mw.metrics_store.record_metric_for(
                    self.mw.current_path,
                    kernel_size = self.mw.kernel_size,
                    dimensions = dims,
                    unit = "cm",
                    volume=volume,
                    area=area,
                    lgi=gi,
                    sulci_depth = depth)
                self.mw.view.enable_png_navigation(saved_pngs, slice_indices=valid_slices)

                mid = len(saved_pngs) // 2
                self.mw.view.on_slice_slider_changed(mid)
                
                print(f"[NIfTI hallmarks] Results:")
                print("The Brain Volume Result = {volume:.2f} cm^3.")
                print(f"The Brain Outer Surface Area Result = {area:.2f} cm^2.")
                print(f"The Brain GI (Convex surface area/ surfacearea) = {gi:.2f} .")
                print(f"Maximum Sulci Depth = {MetricsStore.depth_summary(depth, 'cm')}")

                    
                dt = time.time() - t0
                print(f"[NIfTI hallmarks] Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")

            except Exception as ex:
                logger.error("NIfTI hallmarks failed: %s", ex)
                QMessageBox.critical(self, "NIfTI All hallmarks Failed", f"{type(ex).__name__}: {ex}")
            return
            
        elif self.mw.current_kind == "stl":
            t0 = time.time()
            try:
                uid = uuid.uuid4().hex[:8]
                out_dir = os.path.join(self.mw.temp_dir, f"STL_allmarks_{uid}")
                os.makedirs(out_dir, exist_ok=True)
                
                self.mw.current_output_dir = out_dir
                source_label, dims, area, volume, gi, compactness ,depth, saved_pngs, valid_slices = compute_stl_allmarks(self, file_path=self.mw.current_path,     out_dir=out_dir, min_contour_area=self.mw.cnt_threshold,
                kernel_size = self.mw.kernel_size, slice_thickness=self.mw.slice_thickness)
            
                if source_label == "not_brain":
                    QMessageBox.warning(self, "Mesh ignored", "The computation has been canceled")
                    return
                elif area is None:
                    return

                # record metrics (consistent with your global export; units in mm unless noted)
                self.mw.metrics_store.record_metric_for(
                    self.mw.current_path,
                    source = source_label,
                    kernel_size = self.mw.kernel_size,
                    dimensions = dims,
                    unit = "cm",
                    slice_thickness= self.mw.slice_thickness,
                    volume=volume,
                    area=area,
                    compactness=compactness,
                    sulci_depth = depth,
                    lgi=gi)

                self.mw.view.two_mode_view(out_dir, saved_pngs, valid_slices)

                
                print(f"[STL hallmarks] Results:")
                print(f"STL mesh Volume Result = {volume:.2f} cm^3.")
                print(f"STL mesh Outer Surface Area Result = {area:.2f} cm^2.")
                print(f"STL mesh GI (Convex surface area/ surfacearea) = {gi:.2f} .")
                print(f"STL mesh Compactness = {compactness:.2f} .")
                print(f"The Maximum Grooves Depth = {MetricsStore.depth_summary(depth, 'cm')}")

                dt = time.time() - t0
                print(f"[STL hallmarks] Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")
            
            except Exception as ex:
                logger.error("STL hallmarks failed: %s", ex)
                QMessageBox.critical(self, "STL hallmarks Failed", f"{type(ex).__name__}: {ex}")
                return
        
        elif self.mw.is_vtk:
            if self.mw._flat_axis is not None:
                self._measure_planar_vtk(mode="allmarks")
                return
            t0 = time.time()
            try:
                uid = uuid.uuid4().hex[:8]
                out_dir = os.path.join(self.mw.temp_dir, f"VTL_allmarks_{uid}")
                os.makedirs(out_dir, exist_ok=True)
                
                self.mw.current_output_dir = out_dir
                
                if all(v == 0 for v in self.mw.physical_dim):
                    self.mw.settings.load_mesh_and_ask_geometry()

                u = self.mw.units_length
                area, volume, gi, compactness ,depth, saved_pngs, valid_slices = compute_vtk_allmarks(self, file_path=self.mw.current_path, out_dir=out_dir, min_contour_area=self.mw.cnt_threshold,
                    kernel_size = self.mw.kernel_size, Slice_direction = self.mw.slice_direction, Physical_dim= self.mw.physical_dim, unit = u, slice_thickness=self.mw.slice_thickness)
            
                if area is None:
                    return
       
                # record metrics (consistent with your global export; units in mm unless noted)
                self.mw.metrics_store.record_metric_for(
                    self.mw.current_path,
                    direction = self.mw.slice_direction,
                    kernel_size = self.mw.kernel_size,
                    unit = u,
                    dimensions = self.mw.physical_dim,
                    slice_thickness= self.mw.slice_thickness,
                    volume=volume,
                    area=area,
                    compactness=compactness,
                    sulci_depth = depth,
                    lgi=gi)
                    
                
                self.mw.view.two_mode_view(out_dir, saved_pngs, valid_slices)
                
                print(f"[VTK hallmarks] Results:")
                print(f"VTK mesh Volume Result = {volume:.2f} {u}^3.")
                print(f"VTK mesh Outer Surface Area Result = {area:.2f} {u}^2.")
                print(f"VTK mesh GI (Convex surface area/ surfacearea) = {gi:.2f} .")
                print(f"VTK mesh Compactness = {compactness:.2f} .")
                print(f"VTK mesh Maximum Sulci Depth = {MetricsStore.depth_summary(depth, u)}")

                dt = time.time() - t0
                print(f"[VTK hallmarks] Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")
                      
            except Exception as ex:
                logger.error("VTK hallmarks failed: %s", ex)
                QMessageBox.critical(self, "VTK hallmarks Failed", f"{type(ex).__name__}: {ex}")
                return
            
        else:
            print("[All hallmarks] Unsupported current kind.")


    def on_measure_volumes(self):
        """Compute volume for the currently loaded 3-D object (NIfTI, STL, or VTK)."""
        if not self.mw.current_path or not os.path.isfile(self.mw.current_path):
            print("[Volume] No file is loaded."); return
        if self.mw.current_kind == "image":
            print("[Volume] Implemented for 3D objects only."); return
        
        elif self.mw.current_kind == "nifti":
            t0 = time.time()
            try:
                nif_path = self.mw.current_path
                print(f"[NIfTI] Computing Volume from: {nif_path}")


                uid = uuid.uuid4().hex[:8]
                out_dir = os.path.join(self.mw.temp_dir, f"nifti_volume_{uid}")
                os.makedirs(out_dir, exist_ok=True)
                
                self.mw.current_output_dir = out_dir
                labels = self.mw.nifti_selected_regions if self.mw.nifti_selected_regions else self.mw.labels_available

                dims, volume,saved_pngs, valid_slices = compute_nifti_volume(self, file_path=nif_path, out_dir=out_dir, valid_labels = labels)
            
                if volume is None:
                    return

                # record metrics (consistent with your global export; units in mm unless noted)
                self.mw.metrics_store.record_metric_for(self.mw.current_path, unit="cm", dimensions = dims, volume = volume,)

                self.mw.view.enable_png_navigation(saved_pngs, slice_indices=valid_slices)
                
                mid = len(saved_pngs) // 2
                self.mw.view.on_slice_slider_changed(mid)
                
                print(f"[NIfTI Volume] The Brain Volume Result = {volume:.2f} cm^3. ")
                dt = time.time() - t0
                print(f"[NIfTI Volume] Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")

            except Exception as ex:
                logger.error("NIfTI volume failed: %s", ex)
                QMessageBox.critical(self, "NIfTI Volume Failed", f"{type(ex).__name__}: {ex}")
            return
        elif self.mw.current_kind == "stl":
            t0 = time.time()
            try:
                uid = uuid.uuid4().hex[:8]
                out_dir = os.path.join(self.mw.temp_dir, f"STL_volume_{uid}")
                os.makedirs(out_dir, exist_ok=True)
                
                self.mw.current_output_dir = out_dir
                source_label, dims,volume, saved_pngs, valid_slices = compute_stl_volume(self, file_path=self.mw.current_path,     out_dir=out_dir, min_contour_area=self.mw.cnt_threshold, slice_thickness=self.mw.slice_thickness)
            
                if source_label == "not_brain":
                    QMessageBox.warning(self, "Mesh ignored", "The computation has been canceled")
                    return
                elif volume is None:
                    return

                # record metrics (consistent with your global export; units in mm unless noted)
                self.mw.metrics_store.record_metric_for(
                    self.mw.current_path,
                    source = source_label,
                    slice_thickness= self.mw.slice_thickness,
                    dimensions = dims,
                    unit = "cm",
                    volume=volume)

                self.mw.view.two_mode_view(out_dir, saved_pngs, valid_slices)

                
                print(f"STL mesh Volume Result = {volume:.2f} cm^3.")


                dt = time.time() - t0
                print(f"[STL Volume] Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")

            except Exception as ex:
                logger.error("STL Volume failed: %s", ex)
                QMessageBox.critical(self, "STL Volume Failed", f"{type(ex).__name__}: {ex}")
            return
            
        elif self.mw.is_vtk:
            if self.mw._flat_axis is not None:
                print("[Volume] Not applicable for planar 2D meshes.")
                QMessageBox.information(self, "Volume", "Volume measurement is not applicable for planar 2D meshes.")
                return
            t0 = time.time()
            try:
                uid = uuid.uuid4().hex[:8]
                out_dir = os.path.join(self.mw.temp_dir, f"VTL_volume_{uid}")
                os.makedirs(out_dir, exist_ok=True)
                
                self.mw.current_output_dir = out_dir
                
                if all(v == 0 for v in self.mw.physical_dim):
                    self.mw.settings.load_mesh_and_ask_geometry()

                u = self.mw.units_length
                volume, saved_pngs, valid_slices = compute_vtk_volume(self, file_path=self.mw.current_path, out_dir=out_dir, min_contour_area=self.mw.cnt_threshold,
                    Slice_direction = self.mw.slice_direction, Physical_dim= self.mw.physical_dim, unit = u, slice_thickness=self.mw.slice_thickness)
            
                if volume is None:
                    return
       
                # record metrics (consistent with your global export; units in mm unless noted)
                self.mw.metrics_store.record_metric_for(
                    self.mw.current_path,
                    direction = self.mw.slice_direction,
                    unit = u,
                    dimensions = self.mw.physical_dim,
                    slice_thickness= self.mw.slice_thickness,
                    volume=volume)
                    
                self.mw.view.two_mode_view(out_dir, saved_pngs, valid_slices)
                
                print(f"VTK mesh Volume Result = {volume:.2f} {u}^3.")

                dt = time.time() - t0
                print(f"[VTK hallmarks] Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")
                      
            except Exception as ex:
                logger.error("VTK Volume failed: %s", ex)
                QMessageBox.critical(self, "VTK hallmarks Failed", f"{type(ex).__name__}: {ex}")
                return
                
        else:
            print("[Volume] Unsupported current kind. Open an image, NIfTI or STL file.")

    def on_measure_perimeter(self):
        """Process → Measures → Perimeter: compute and show annotated result WITHOUT saving."""
        if not self.mw.current_path or not os.path.isfile(self.mw.current_path):
            print("[Perimeter] No file is loaded."); return
        
        if self.mw.current_kind == "image":
            try:
                result = self.mw.settings.ensure_calibrated()
                if result is None:
                    return
                u, px_size = result

                print(f"[Perimeter] Measuring: {self.mw.current_path}")
                print(f"[Perimeter] Measuring with pixel size = {px_size} {u}/pixel")

                image_path = self.mw.current_path
                if self.mw.last_annotated_path is not None:
                    image_path = self.mw.last_annotated_path
                
                perimeter, annotated_bgr = compute_image_perimeter(
                    image_path,
                    pixel_size = px_size,
                    cnt_threshold = self.mw.cnt_threshold,
                    unit = u,
                    add_scalebar=not bool(self.mw.image_scale_from_scalebar.get(self.mw.current_path, False)),
                    draw_hallmarks=self.mw.draw_hallmarks_on_image,
                )
                print(f"Annotated perimeter = {perimeter:.2f} {u}.")
                
                label_text = self.mw.get_label_for_cropped_path(image_path)
                if label_text:
                    annotated_bgr = put_label_on_bgr(annotated_bgr, label_text, pos="topleft")
                # Convert BGR ndarray → QPixmap and show (no disk write)
                pm = ViewManager.np_bgr_to_qpixmap(annotated_bgr)
                self.mw.image_label.setImage(pm)
                self.mw.image_label.remove_last_annotation()
                self.mw.view.show_widget(self.mw.image_label)
                # Keep kind/path as the ORIGINAL file; Save View As… will ask user where to save what they see.
                self.mw._active_view = "image"
                # Ensure File/Process actions stay enabled
                self.mw._set_current("image", self.mw.current_path)
                self.mw.metrics_store.record_metric_for(self.mw.current_path,label=label_text,
                    pixel_size_units = f"{self.mw.units_length}/pixel",
                    unit= self.mw.units_length,
                    pixel_size = self.mw.pixel_size,
                    perimeter=perimeter)

            except Exception as ex:
                logger.error("Perimeter failed: %s", ex)
                QMessageBox.critical(self, "Perimeter Failed", f"{type(ex).__name__}: {ex}")
            
        elif self.mw.is_vtk:
            if self.mw._flat_axis is not None:
                self._measure_planar_vtk(mode="perimeter")
                return
            print("[Perimeter] Not supported for 3D VTK meshes.")
            return

        else:
            return

    def on_measure_compactness(self):
        """Compute compactness for current image or 3D mesh, reusing saved metrics when available."""
        if not self.mw.current_path or not os.path.isfile(self.mw.current_path):
            print("[Compactness] No file is loaded."); return

        # ── 3D mesh  ──────────────────────────────────────────────
        if self.mw.current_kind == "stl" or (self.mw.is_vtk and self.mw._flat_axis is None):
            try:
                t0 = time.time()
                rows = self.mw.metrics.get(self.mw.current_path, []) if isinstance(getattr(self, "metrics", None), dict) else []
                if isinstance(rows, dict):
                    rows = [rows]
                last_row = rows[-1] if rows else None

                volume = float(last_row["Volume"]) if last_row and last_row.get("Volume") is not None else None
                area = float(last_row["Area"]) if last_row and last_row.get("Area") is not None else None

                if volume is not None and area is not None:
                    comp = compactness_3D(volume, area)
                else:
                    uid = uuid.uuid4().hex[:8]
                    out_dir = os.path.join(self.mw.temp_dir, f"3D_compactness_{uid}")
                    os.makedirs(out_dir, exist_ok=True)
                    self.mw.current_output_dir = out_dir

                    if self.mw.current_kind == "stl":
                        source_label, dims, comp, saved_pngs, valid_slices = compute_compactness_stl(
                            self, file_path=self.mw.current_path, out_dir=out_dir,
                            min_contour_area=self.mw.cnt_threshold, slice_thickness=self.mw.slice_thickness)
                        if source_label == "not_brain":
                            return

                    elif self.mw.is_vtk:
                        if all(v == 0 for v in self.mw.physical_dim):
                            self.mw.settings.load_mesh_and_ask_geometry()
                        comp, saved_pngs, valid_slices = compute_compactness_vtk(self, file_path=self.mw.current_path,
                        out_dir=out_dir, min_contour_area=self.mw.cnt_threshold,
                        Slice_direction=self.mw.slice_direction, Physical_dim=self.mw.physical_dim,
                        unit=self.mw.units_length, slice_thickness=self.mw.slice_thickness)

                    if comp is None:
                        return

                    self.mw.view.two_mode_view(out_dir, saved_pngs, valid_slices)

                self.mw.metrics_store.record_metric_for(
                    self.mw.current_path,
                    slice_thickness=self.mw.slice_thickness, 
                    compactness=comp)

                base_name = os.path.basename(self.mw.current_path)
                print(f"[Compactness] for {base_name}: Compactness(3D)={comp:.4f}")
                if comp > 1.0:
                    QMessageBox.warning(self, "Compactness Warning",
                        f"Compactness = {comp:.4f} exceeds 1.0.\n"
                        "The expected range is [0, 1]. This may indicate incorrect "
                        "physical dimensions or unit settings.")
                dt = time.time() - t0
                print(f"[Compactness] Done in {dt:.2f}s.")

            except Exception as ex:
                logger.error("Compactness failed: %s", ex)
                QMessageBox.critical(self, "Compactness Failed", f"{type(ex).__name__}: {ex}")
            return

        # ── 2D image ───────────────────────────────────────────────────
        if self.mw.current_kind == "image":
            try:
                image_path = self.mw.current_path
                if self.mw.last_annotated_path is not None:
                    image_path = self.mw.last_annotated_path
                label_text = self.mw.get_label_for_cropped_path(image_path)

                rows = self.mw.metrics.get(self.mw.current_path, []) if isinstance(getattr(self, "metrics", None), dict) else []
                if isinstance(rows, dict):
                    rows = [rows]
                last_row = next((r for r in reversed(rows) if r.get("Annotation") == label_text), None)

                area = last_row.get("Area") if last_row else None
                perimeter = last_row.get("Perimeter") if last_row else None

                if area is not None and perimeter is not None:
                    area = float(area)
                    perimeter = float(perimeter)
                    compactness_2D_value = compactness_2D(area, perimeter)
                else:
                    compactness_2D_value, annotated_bgr = compute_compactness_2D(image_path, cnt_threshold=self.mw.cnt_threshold)
                    pm = ViewManager.np_bgr_to_qpixmap(annotated_bgr)
                    self.mw.image_label.setImage(pm)
                    self.mw.image_label.remove_last_annotation()
                    self.mw.view.show_widget(self.mw.image_label)
                    self.mw._active_view = "image"

                base_name = os.path.basename(image_path)
                print(f"[Compactness] for {base_name}: Compactness={compactness_2D_value:.4f}")
                if compactness_2D_value > 1.0:
                    QMessageBox.warning(self, "Compactness Warning",
                        f"Compactness = {compactness_2D_value:.4f} exceeds 1.0.\n"
                        "The expected range is [0, 1]. This may indicate an issue "
                        "with contour detection or image quality.")
                self.mw._set_current("image", self.mw.current_path)

            except Exception as ex:
                logger.error("Compactness failed: %s", ex)
                QMessageBox.critical(self, "Compactness Failed", f"{type(ex).__name__}: {ex}")
        else:
            QMessageBox.information(self, "Compactness", "Compactness measurement is currently only supported for 2D images and 3D meshes. Please open an image or 3D mesh file.")      
            print("[Compactness] Unsupported current kind. Open an image or 3D mesh file.")
            return

    def on_measure_straight(self):
        """Process → Measures → Straight Line: interactive two-click distance measurement."""
        if not self.mw.current_path or not os.path.isfile(self.mw.current_path):
            print("[Straight line] No file is loaded."); return

        if self.mw.current_kind != "image":
            print("[Straight line] Only supported for images."); return

        # ensure calibration
        result = self.mw.settings.ensure_calibrated()
        if result is None:
            return
        u, px_size = result

        print(f"[Straight line] Click two points on the image to measure distance.")

        def _finish(pixel_length, p1, p2):
            distance = pixel_length * px_size
            self.mw.image_label.add_line_annotation(
                p1, p2, label=f"{distance:.2f} {u}", color=QColor(0, 200, 255))
            self.mw.metrics_store.record_metric_for(
                self.mw.current_path,
                unit=u,
                pixel_size=px_size,
                straight_line_distance=distance)
            print(f"[Straight line] Distance = {distance:.2f} {u}")

        self.mw.image_label.start_line_measure(_finish)
    
    def on_measure_lgi(self):
        """Process → Measures → lGI: compute and show annotated result WITHOUT saving."""
        
        if not self.mw.current_path or not os.path.isfile(self.mw.current_path):
            print("[lGI] No file is loaded."); return
        if self.mw.current_kind == "image":
            try:
                result = self.mw.settings.ensure_calibrated()
                if result is None:
                    return
                u, px_size = result

                print(f"[lGI] Measuring: {self.mw.current_path}")

                image_path = self.mw.current_path
                if self.mw.last_annotated_path is not None:
                    image_path = self.mw.last_annotated_path
                lGI,perimeter, perimeter_convex, annotated_bgr = compute_image_lGI(
                    image_path,
                    pixel_size = px_size,
                    kernel_size= self.mw.kernel_size,
                    cnt_threshold=self.mw.cnt_threshold,
                    unit = u,
                    add_scalebar=not bool(self.mw.image_scale_from_scalebar.get(self.mw.current_path, False)),
                    draw_hallmarks=self.mw.draw_hallmarks_on_image,
                )
                print(f"lGI = {lGI:.2f}.")

                label_text = self.mw.get_label_for_cropped_path(image_path)
                if label_text:
                    annotated_bgr = put_label_on_bgr(annotated_bgr, label_text, pos="topleft")
                # Convert BGR ndarray → QPixmap and show (no disk write)
                pm = ViewManager.np_bgr_to_qpixmap(annotated_bgr)
                self.mw.image_label.setImage(pm)
                self.mw.image_label.remove_last_annotation()
                self.mw.view.show_widget(self.mw.image_label)
                # Keep kind/path as the ORIGINAL file; Save View As… will ask user where to save what they see.
                self.mw._active_view = "image"
                # Ensure File/Process actions stay enabled
                self.mw._set_current("image", self.mw.current_path)
                self.mw.metrics_store.record_metric_for(self.mw.current_path, annotation=label_text,
                    pixel_size_units = f"{self.mw.units_length}/pixel",
                    unit = self.mw.units_length,
                    pixel_size = self.mw.pixel_size,
                    kernel_size = self.mw.kernel_size,
                    perimeter=perimeter, perimeter_convex=perimeter_convex, lgi=lGI)

            except Exception as ex:
                logger.error("lGI failed: %s", ex)
                QMessageBox.critical(self, "lGI Failed", f"{type(ex).__name__}: {ex}")
                
        elif self.mw.current_kind == "nifti":
            t0 = time.time()
            reply = QMessageBox.question(self,"Enhance measurement",
            "For accurate LGI computation, please provide the FreeSurfer pial surfaces for both hemispheres (lh.pial and rh.pial). Do you have these files?",   # message
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            
            if reply == QMessageBox.No:
                QMessageBox.warning(self, "LGI Input Missing",
                    "The LGI can be computed based on the NIfTI file alone, but the accuracy of the results is not guaranteed.")
                
                try:
                    nif_path = self.mw.current_path
                    print(f"[NIfTI] Computing lGI from: {nif_path}")


                    uid = uuid.uuid4().hex[:8]
                    out_dir = os.path.join(self.mw.temp_dir, f"nifti_lGI_{uid}")
                    os.makedirs(out_dir, exist_ok=True)
                    
                    self.mw.current_output_dir = out_dir
                    labels = self.mw.nifti_selected_regions if self.mw.nifti_selected_regions else self.mw.labels_available

                    lGI,saved_pngs, valid_slices = compute_nifti_lGI(self, file_path=nif_path, out_dir=out_dir, valid_labels = labels, min_contour_area=self.mw.cnt_threshold, kernel_size= self.mw.kernel_size,)
                
                    if lGI is None:
                        return

                    # record metrics (consistent with your global export; units in mm unless noted)
                    self.mw.metrics_store.record_metric_for(self.mw.current_path, kernel_size= self.mw.kernel_size ,lgi = lGI,)

                    self.mw.view.enable_png_navigation(saved_pngs, slice_indices=valid_slices)
                    
                    mid = len(saved_pngs) // 2
                    self.mw.view.on_slice_slider_changed(mid)
                    
                    print(f"[NIfTI lGI] The Brain GI (Convex surface area/ surfacearea) = {lGI:.2f}. ")
                    dt = time.time() - t0
                    print(f"[NIfTI lGI] Done in {dt:.2f}s. Results live in TEMP.\n"
                          f"Use File → Save Data As… to copy outputs you want to keep.")

                except Exception as ex:
                    logger.error("NIfTI lGI failed: %s", ex)
                    QMessageBox.critical(self, "NIfTI lGI Failed", f"{type(ex).__name__}: {ex}")
                return
        
            elif reply == QMessageBox.Yes:
                nif_path = self.mw.current_path
#                QTimer.singleShot(0, self.mw.on_combined_stl)
                self.mw.on_combined_stl()
                stl_path = self.mw.current_path if (self.mw.current_path and os.path.isfile(self.mw.current_path)) else None
                
                try:
                    print(f"[NIfTI] Computing lGI from: {nif_path} based on rh & lh .pial")

                    uid = uuid.uuid4().hex[:8]
                    out_dir = os.path.join(self.mw.temp_dir, f"STL_lGI_{uid}")
                    os.makedirs(out_dir, exist_ok=True)
                    
                    self.mw.current_output_dir = out_dir
                    source_label, dims, gi, saved_pngs, valid_slices =compute_stl_lGI(
                        self,
                        file_path=stl_path,
                        out_dir=out_dir,
                        min_contour_area=self.mw.cnt_threshold,
                        kernel_size=self.mw.kernel_size,
                        slice_thickness=self.mw.slice_thickness,
                        build_solid=False,   # keep False for stability
                    )
                                        
                
                    if gi is None:
                        return

                    # record metrics (consistent with your global export; units in mm unless noted)
                    self.mw.metrics_store.record_metric_for(
                        self.mw.current_path,
                        source = source_label,
                        kernel_size = self.mw.kernel_size,
                        dimensions = dims,
                        unit = "cm",
                        slice_thickness= self.mw.slice_thickness,
                        lgi=gi)
                        
                    self.mw.view.enable_png_navigation(saved_pngs, slice_indices=valid_slices)
                    
                    mid = len(saved_pngs) // 2
                    self.mw.view.on_slice_slider_changed(mid)
                    
                    print(f"[STL lGI] The Brain GI (Convex surface area/ surfacearea) = {gi:.2f}. ")
                    dt = time.time() - t0
                    print(f"[STL lGI] Done in {dt:.2f}s. Results live in TEMP.\n"
                          f"Use File → Save Data As… to copy outputs you want to keep.")

                except Exception as ex:
                    logger.error("STL lGI failed: %s", ex)
                    QMessageBox.critical(self, "STL lGI Failed", f"{type(ex).__name__}: {ex}")
                return
                
        elif self.mw.current_kind == "stl":
        
            t0 = time.time()
            try:
                uid = uuid.uuid4().hex[:8]
                out_dir = os.path.join(self.mw.temp_dir, f"STL_lgi_{uid}")
                os.makedirs(out_dir, exist_ok=True)
                
                self.mw.current_output_dir = out_dir
                source_label, dims, gi, saved_pngs, valid_slices = compute_stl_lGI(self, file_path=self.mw.current_path,     out_dir=out_dir, min_contour_area=self.mw.cnt_threshold,
                kernel_size = self.mw.kernel_size, slice_thickness=self.mw.slice_thickness)
            
                if source_label == "not_brain":
                    QMessageBox.warning(self, "Mesh ignored", "The computation has been canceled")
                    return
                elif gi is None:
                    return

                # record metrics (consistent with your global export; units in mm unless noted)
                self.mw.metrics_store.record_metric_for(
                    self.mw.current_path,
                    source = source_label,
                    kernel_size = self.mw.kernel_size,
                    dimensions = dims,
                    unit = "cm",
                    slice_thickness= self.mw.slice_thickness,
                    lgi=gi)

                self.mw.view.two_mode_view(out_dir, saved_pngs, valid_slices)

                
                print(f"STL mesh GI (Convex surface area/ surfacearea) = {gi:.2f} .")

                dt = time.time() - t0
                print(f"[STL lGI] Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")

            except Exception as ex:
                logger.error("STL lGI failed: %s", ex)
                QMessageBox.critical(self, "STL lGI Failed", f"{type(ex).__name__}: {ex}")
            return
            
        elif self.mw.is_vtk:
            if self.mw._flat_axis is not None:
                self._measure_planar_vtk(mode="lGI")
                return
            t0 = time.time()
            try:
                uid = uuid.uuid4().hex[:8]
                out_dir = os.path.join(self.mw.temp_dir, f"VTL_lGI_{uid}")
                os.makedirs(out_dir, exist_ok=True)
                
                self.mw.current_output_dir = out_dir
                
                if all(v == 0 for v in self.mw.physical_dim):
                    self.mw.settings.load_mesh_and_ask_geometry()

                u = self.mw.units_length
                gi, saved_pngs, valid_slices = compute_vtk_lGI(self, file_path=self.mw.current_path, out_dir=out_dir, min_contour_area=self.mw.cnt_threshold,
                    kernel_size = self.mw.kernel_size, Slice_direction = self.mw.slice_direction, Physical_dim= self.mw.physical_dim, unit = u, slice_thickness=self.mw.slice_thickness)
            
                if gi is None:
                    return
       
                # record metrics (consistent with your global export; units in mm unless noted)
                self.mw.metrics_store.record_metric_for(
                    self.mw.current_path,
                    direction = self.mw.slice_direction,
                    kernel_size = self.mw.kernel_size,
                    unit = u,
                    dimensions = self.mw.physical_dim,
                    slice_thickness= self.mw.slice_thickness,
                    lgi=gi)
                    
                
                self.mw.view.two_mode_view(out_dir, saved_pngs, valid_slices)
                
                print(f"VTK mesh GI (Convex surface area/ surfacearea) = {gi:.2f} .")

                dt = time.time() - t0
                print(f"[VTK lGI] Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")
                      
            except Exception as ex:
                logger.error("VTK lGI failed: %s", ex)
                QMessageBox.critical(self, "VTK lGI Failed", f"{type(ex).__name__}: {ex}")
                return
            
        else:
            print("[lGI] Unsupported current kind.")

            
        
    
    def on_measure_sulci_depth(self):
        """Process → Measures → All hallmarks for 2D images: compute and show annotated result WITHOUT saving."""
        if not self.mw.current_path or not os.path.isfile(self.mw.current_path):
            print("[Sulci depth] No file is loaded."); return
            
        if self.mw.current_kind == "image":
            try:
                result = self.mw.settings.ensure_calibrated()
                if result is None:
                    return
                u, px_size = result

                print(f"[Sulci depth] Measuring: {self.mw.current_path}")
                print(f"[Sulci depth] Measuring with pixel size = {px_size} {u}/pixel")

                image_path = self.mw.current_path
                if self.mw.last_annotated_path is not None:
                    image_path = self.mw.last_annotated_path
                depth, annotated_bgr = compute_image_sulci_depth(
                    image_path,
                    pixel_size = px_size,
                    cnt_threshold=self.mw.cnt_threshold,
                    unit = u,
                    add_scalebar=not bool(self.mw.image_scale_from_scalebar.get(self.mw.current_path, False))
                )
                print(f"[Sulci depth] Maximum Sulci Depth = {MetricsStore.depth_summary(depth, u)}")
                
                label_text = self.mw.get_label_for_cropped_path(image_path)
                if label_text:
                    annotated_bgr = put_label_on_bgr(annotated_bgr, label_text, pos="topleft")


                # Convert BGR ndarray → QPixmap and show (no disk write)
                pm = ViewManager.np_bgr_to_qpixmap(annotated_bgr)
                self.mw.image_label.setImage(pm)
                self.mw.image_label.remove_last_annotation()
                self.mw.view.show_widget(self.mw.image_label)
                # Keep kind/path as the ORIGINAL file; Save View As… will ask user where to save what they see.
                self.mw._active_view = "image"
                # Ensure File/Process actions stay enabled
                self.mw._set_current("image", self.mw.current_path)
                self.mw.metrics_store.record_metric_for(self.mw.current_path, annotation=label_text,
                    pixel_size_units = f"{self.mw.units_length}/pixel",
                    unit = self.mw.units_length,
                    pixel_size = self.mw.pixel_size,
                    sulci_depth = depth)

            except Exception as ex:
                logger.error("Sulci depth failed: %s", ex)
                QMessageBox.critical(self, "Sulci depth Failed", f"{type(ex).__name__}: {ex}")
        
        elif self.mw.current_kind == "nifti":
            t0 = time.time()
            try:
                nif_path = self.mw.current_path
                print(f"[NIfTI] Computing Volume from: {nif_path}")


                uid = uuid.uuid4().hex[:8]
                out_dir = os.path.join(self.mw.temp_dir, f"nifti_volume_{uid}")
                os.makedirs(out_dir, exist_ok=True)
                
                self.mw.current_output_dir = out_dir
                labels = self.mw.nifti_selected_regions if self.mw.nifti_selected_regions else self.mw.labels_available

                dims, depth,saved_pngs, valid_slices = compute_nifti_sulci_depth(self, file_path=nif_path, out_dir=out_dir, valid_labels = labels, min_contour_area=self.mw.cnt_threshold)
            
                if depth is None:
                    return

                # record metrics (consistent with your global export; units in mm unless noted)
                self.mw.metrics_store.record_metric_for(self.mw.current_path, unit ="mm", dimensions = dims, sulci_depth = depth,)

                self.mw.view.enable_png_navigation(saved_pngs, slice_indices=valid_slices)
                
                mid = len(saved_pngs) // 2
                self.mw.view.on_slice_slider_changed(mid)
                
                print(f"[NIfTI Sulci depth] The max Brain Sulci depth across slices = {MetricsStore.depth_summary(depth, 'mm')}")
                dt = time.time() - t0
                print(f"[NIfTI Sulci depth] Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")

            except Exception as ex:
                logger.error("NIfTI Sulci depth failed: %s", ex)
                QMessageBox.critical(self, "NIfTI Sulci depth Failed", f"{type(ex).__name__}: {ex}")
            return
            
        elif self.mw.current_kind == "stl":
            t0 = time.time()
            try:
                uid = uuid.uuid4().hex[:8]
                out_dir = os.path.join(self.mw.temp_dir, f"STL_sulic_depth_{uid}")
                os.makedirs(out_dir, exist_ok=True)
                
                self.mw.current_output_dir = out_dir
                source_label, dims, depth, saved_pngs, valid_slices = compute_stl_sulci_depth (self, file_path=self.mw.current_path, out_dir=out_dir, min_contour_area=self.mw.cnt_threshold, slice_thickness=self.mw.slice_thickness)
            
                if source_label == "not_brain":
                    QMessageBox.warning(self, "Mesh ignored", "The computation has been canceled")
                    return
                elif depth is None:
                    return

                # record metrics (consistent with your global export; units in mm unless noted)
                self.mw.metrics_store.record_metric_for(self.mw.current_path, source = source_label, slice_thickness= self.mw.slice_thickness,
                    dimensions = dims,unit ="mm", sulci_depth = depth)

                self.mw.view.two_mode_view(out_dir, saved_pngs, valid_slices)
                
                print(f"[STL Sulci depth] The max Brain Sulci depth across slices = {MetricsStore.depth_summary(depth, 'mm')}")
                dt = time.time() - t0
                print(f"[STL Sulci depth] Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")

            except Exception as ex:
                logger.error("STL Sulci depth failed: %s", ex)
                QMessageBox.critical(self, "STL Sulci depth Failed", f"{type(ex).__name__}: {ex}")
            return
            
        elif self.mw.is_vtk:
            if self.mw._flat_axis is not None:
                self._measure_planar_vtk(mode="sulci_depth")
                return
            t0 = time.time()
            try:
                uid = uuid.uuid4().hex[:8]
                out_dir = os.path.join(self.mw.temp_dir, f"VTL_sulic_depth_{uid}")
                os.makedirs(out_dir, exist_ok=True)
                
                self.mw.current_output_dir = out_dir
                
                if all(v == 0 for v in self.mw.physical_dim):
                    self.mw.settings.load_mesh_and_ask_geometry()

                u = self.mw.units_length
                depth, saved_pngs, valid_slices = compute_vtk_sulci_depth(self, file_path=self.mw.current_path, out_dir=out_dir, min_contour_area=self.mw.cnt_threshold,
                Slice_direction = self.mw.slice_direction, Physical_dim= self.mw.physical_dim, unit = u, slice_thickness=self.mw.slice_thickness)
            
                if depth is None:
                    return
       
                # record metrics (consistent with your global export; units in mm unless noted)
                self.mw.metrics_store.record_metric_for(
                    self.mw.current_path,
                    direction = self.mw.slice_direction,
                    unit = u,
                    dimensions = self.mw.physical_dim,
                    slice_thickness= self.mw.slice_thickness,
                    sulci_depth = depth)
                                    
                self.mw.view.two_mode_view(out_dir, saved_pngs, valid_slices)
                
                if isinstance(depth, (list, tuple)) and len(depth) > 0:
                    print("[VTK Sulci depth]")
                    print(f"The Maximum Grooves Depth = {MetricsStore.depth_summary(depth, u)}")

                dt = time.time() - t0
                print(f"[VTK Sulci depth] Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")
                      
            except Exception as ex:
                logger.error("VTK Sulci depth failed: %s", ex)
                QMessageBox.critical(self, "VTK Sulci depth Failed", f"{type(ex).__name__}: {ex}")
                return
        else:
            print("[Sulci depth] Unsupported current kind.")

            
            
    def on_measure_area(self):
        """Compute surface area for the current file and display the annotated result.

        Dispatches to the correct back-end depending on ``current_kind``
        (image, NIfTI, STL, or VTK).  Results are stored in the metrics
        dict but not automatically saved to disk.
        """
        if self.mw.current_kind == "image":
            if not self.mw.current_path or not os.path.isfile(self.mw.current_path):
                print("[Area] No image file is loaded."); return
            try:
                result = self.mw.settings.ensure_calibrated()
                if result is None:
                    return
                u, px_size = result

                print(f"[Area] Measuring: {self.mw.current_path}")
                print(f"[Area] Measuring with pixel size = {px_size} {u}/pixel")
                
                image_path = self.mw.current_path
                if self.mw.last_annotated_path is not None:
                    image_path = self.mw.last_annotated_path
                    
                area, annotated_bgr = compute_image_area(
                    image_path,
                    pixel_size=px_size,
                    cnt_threshold=self.mw.cnt_threshold,
                    unit = u,
                    add_scalebar=not bool(self.mw.image_scale_from_scalebar.get(self.mw.current_path, False)),
                    draw_hallmarks=self.mw.draw_hallmarks_on_image,
                )
                
                label_text = self.mw.get_label_for_cropped_path(image_path)
                if label_text:
                    annotated_bgr = put_label_on_bgr(annotated_bgr, label_text, pos="topleft")
                    
                print(f"[Area] Result = {area:.2f} {u}^2.")
                # Convert BGR ndarray → QPixmap and show (no disk write)
                pm = ViewManager.np_bgr_to_qpixmap(annotated_bgr)
                self.mw.image_label.setImage(pm)
                self.mw.image_label.remove_last_annotation()
                self.mw.view.show_widget(self.mw.image_label)
                # Keep kind/path as the ORIGINAL file; Save View As… will ask user where to save what they see.
                self.mw._active_view = "image"
                # Ensure File/Process actions stay enabled
                self.mw._set_current("image", self.mw.current_path)
                self.mw.metrics_store.record_metric_for(self.mw.current_path, annotation=label_text ,
                pixel_size_units = f"{self.mw.units_length}/pixel",
                pixel_size = self.mw.pixel_size,
                unit = self.mw.units_length,
                area=area)
                
            except Exception as ex:
                print(f"[Area] ERROR : {ex}")
                QMessageBox.critical(self, "[Area] Failed", f"{type(ex).__name__}: {ex}")
        elif self.mw.current_kind == "nifti":
            t0 = time.time()
            try:
                nif_path = self.mw.current_path
                print(f"[NIfTI] Computing area/perimeter from: {nif_path}")


                uid = uuid.uuid4().hex[:8]
                out_dir = os.path.join(self.mw.temp_dir, f"nifti_area_{uid}")
                os.makedirs(out_dir, exist_ok=True)
                
                self.mw.current_output_dir = out_dir
                labels = self.mw.nifti_selected_regions if self.mw.nifti_selected_regions else self.mw.labels_available

                dims, area,saved_pngs, valid_slices = compute_nifti_area(self, file_path=nif_path, out_dir=out_dir, valid_labels = labels, min_contour_area=self.mw.cnt_threshold,)
            
                if area == 0:
                    QMessageBox.information(self, "NIfTI Area", "All slices were filtered out (too small).")
                    return

                # record metrics (consistent with your global export; units in mm unless noted)
                self.mw.metrics_store.record_metric_for(self.mw.current_path, unit="cm", dimensions = dims, area = area,)

                self.mw.view.enable_png_navigation(saved_pngs, slice_indices=valid_slices)
                
                mid = len(saved_pngs) // 2
                self.mw.view.on_slice_slider_changed(mid)
                
                print(f"[NIfTI Area] The Brain Outer Surface Area Result = {area:.2f} cm^2. ")
                dt = time.time() - t0
                print(f"[NIfTI Area] Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")

            except Exception as ex:
                logger.error("NIfTI Area failed: %s", ex)
                QMessageBox.critical(self, "[NIfTI Area] Failed", f"{type(ex).__name__}: {ex}")
            return
            
        elif self.mw.current_kind == "stl":
            t0 = time.time()
            try:
                uid = uuid.uuid4().hex[:8]
                out_dir = os.path.join(self.mw.temp_dir, f"STL_area_{uid}")
                os.makedirs(out_dir, exist_ok=True)
                
                self.mw.current_output_dir = out_dir
                source_label, dims,area, saved_pngs, valid_slices = compute_stl_area(self, file_path=self.mw.current_path,     out_dir=out_dir, min_contour_area=self.mw.cnt_threshold, slice_thickness=self.mw.slice_thickness)
            
                if source_label == "not_brain":
                    QMessageBox.warning(self, "Mesh ignored", "The computation has been canceled")
                    return
                elif area is None:
                    return

                # record metrics (consistent with your global export; units in mm unless noted)
                self.mw.metrics_store.record_metric_for(
                    self.mw.current_path,
                    source = source_label,
                    slice_thickness= self.mw.slice_thickness,
                    dimensions = dims,
                    unit = "cm",
                    area=area)
  
                self.mw.view.two_mode_view(out_dir, saved_pngs, valid_slices)

                
                print(f"STL mesh Area Result = {area:.2f} cm^2.")


                dt = time.time() - t0
                print(f"[STL Area] Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")

            except Exception as ex:
                logger.error("STL Area failed: %s", ex)
                QMessageBox.critical(self, "STL Area Failed", f"{type(ex).__name__}: {ex}")
            return
        
        elif self.mw.is_vtk:
            if self.mw._flat_axis is not None:
                self._measure_planar_vtk(mode="area")
                return
            t0 = time.time()
            try:
                uid = uuid.uuid4().hex[:8]
                out_dir = os.path.join(self.mw.temp_dir, f"VTL_area_{uid}")
                os.makedirs(out_dir, exist_ok=True)
                
                self.mw.current_output_dir = out_dir
                
                if all(v == 0 for v in self.mw.physical_dim):
                    self.mw.settings.load_mesh_and_ask_geometry()

                u = self.mw.units_length
                area, saved_pngs, valid_slices = compute_vtk_area(self, file_path=self.mw.current_path, out_dir=out_dir, min_contour_area=self.mw.cnt_threshold, Slice_direction = self.mw.slice_direction, Physical_dim= self.mw.physical_dim, unit = u, slice_thickness=self.mw.slice_thickness)
            
                if area is None:
                    return
       
                # record metrics (consistent with your global export; units in mm unless noted)
                self.mw.metrics_store.record_metric_for(
                    self.mw.current_path,
                    direction = self.mw.slice_direction,
                    unit = u,
                    dimensions = self.mw.physical_dim,
                    slice_thickness= self.mw.slice_thickness,
                    area=area)
                    
                
                self.mw.view.two_mode_view(out_dir, saved_pngs, valid_slices)
                
                print(f"VTK mesh Outer Surface Area Result = {area:.2f} {u}^2.")


                dt = time.time() - t0
                print(f" Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")
                      
            except Exception as ex:
                logger.error("VTK Area failed: %s", ex)
                QMessageBox.critical(self, "VTK area Failed", f"{type(ex).__name__}: {ex}")
                return
    
        else:
            print("[Area] Unsupported current kind. Open an image, NIfTI, or STL file.")

    def on_process_batch(self):
        """Run all-hallmarks measurement on every image in a user-selected folder.

        The user is prompted to adjust the first image (annotation, scale,
        etc.) and press Shift+Alt+E to continue.  All images in the batch
        must share the same resolution and unit.
        """
        start = self.mw.last_dir if os.path.isdir(self.mw.last_dir) else ""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Choose a folder", start,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if not dir_path:
            return  # user canceled
            
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        imgs = sorted(
        (e.path for e in os.scandir(dir_path) if e.is_file()
         and os.path.splitext(e.name.lower())[1] in exts),
        key=lambda p: os.path.basename(p).lower())
        if not imgs:
            QMessageBox.warning(self, "No images", "The selected folder contains no image files.")
            return

        self.mw.last_dir = dir_path
        first_pm = QPixmap(imgs[0])
        if first_pm.isNull():
            QMessageBox.critical(
                self,
                "Process Batch Failed",
                f"Could not open image file:\n{imgs[0]}",
            )
            return
        self.mw.file_mgr.load_image(imgs[0])  # show first image

        self.mw._enter_adjustment_mode()
        self.mw.statusBar().showMessage("Adjust the image now and then press Shift+Alt+E to continue.")
        print("[Process Batch] Adjust the image now and then press Shift+Alt+E to continue.")
        self.mw.wait_for_resume()   # blocks here; resumes after key press
        self.mw.statusBar().clearMessage()
        self.mw._exit_adjustment_mode()

        btn = QMessageBox.warning(self,
                    "Processing Images Batch",
                    "All images must share the same resolution (pixel spacing) and measurement unit.",
                    QMessageBox.Ok | QMessageBox.Cancel)
        if btn == QMessageBox.Cancel:
            return

        result = self.mw.settings.ensure_calibrated()
        if result is None:
            return
        u, px_size = result

        uid = uuid.uuid4().hex[:8]
        out_dir = os.path.join(self.mw.temp_dir, f"Process_images_{uid}")
        os.makedirs(out_dir, exist_ok=True)
        self.mw.current_output_dir = out_dir
        print(f"[Process Batch] TEMP output: {out_dir}")

        self.mw.reset_view()


        try:
            # Fail fast: stop the whole batch if any image cannot be opened.
            for img_path in imgs:
                if cv2.imread(img_path) is None:
                    raise ValueError(f"Could not read image: {img_path}")

            valid_slices, saved_pngs = process_on_images_batch(dir_path, out_dir, pixel_size=px_size, kernel_size= self.mw.kernel_size,
                cnt_threshold = self.mw.cnt_threshold, unit = u)
                
            self.mw.view.enable_png_navigation(saved_pngs, slice_indices=valid_slices)

            mid = len(saved_pngs) // 2
            self.mw.view.on_slice_slider_changed(mid)
            
        except Exception as ex:
            logger.error("Process Batch failed: %s", ex)
            QMessageBox.critical(self, "Process Batch Failed", f"{type(ex).__name__}: {ex}")
            return

    def on_measure_curvature(self):
        """Compute and display curvature profiles for the current 2-D image.

        Generates two plot variants accessible via Ctrl+M / Ctrl+Shift+M.
        """
        if self.mw.current_kind == "image":
            try:
                uid = uuid.uuid4().hex[:8]
                out_dir = os.path.join(self.mw.temp_dir, f"Curvature_{uid}")
                os.makedirs(out_dir, exist_ok=True)
                self.mw.current_output_dir = out_dir

                mask, edge_pixels, curvature_values,curvature_values_s  = compute_curvature_profile(path =self.mw.current_path, min_area = self.mw.cnt_threshold)
                
                print(f"[Curvature] Analysis completed for image {self.mw.current_path}")
                # Convert BGR ndarray → QPixmap and show (no disk write)
                
                img = save_curvature_plot(out_dir,  mask, edge_pixels, curvature_values)
                img2 = save_curvature_plot(out_dir,  mask, edge_pixels, curvature_values_s, filename="curvature_plot_2.png")
                pm = ViewManager.np_bgr_to_qpixmap(img)
                pm2 = ViewManager.np_bgr_to_qpixmap(img2)
                
                self.mw._pms = [pm, pm2]
                self.mw._pm_index = 0
                self.mw.image_label.setImage(self.mw._pms[self.mw._pm_index])
                self.mw.view.show_widget(self.mw.image_label)
                # Keep kind/path as the ORIGINAL file; Save View As… will ask user where to save what they see.
                self.mw.statusBar().showMessage("Use Ctrl+M to toggle between the two modes.")

                self.mw._active_view = "image"
                # Ensure File/Process actions stay enabled
                self.mw._set_current("image", self.mw.current_path)
            except Exception as ex:
                logger.error("Curvature failed: %s", ex)
                QMessageBox.critical(self, "[Curvature] Failed", f"{type(ex).__name__}: {ex}")
            return
            
        else:
            print("[Curvature] Unsupported current kind. Open an image first.")

    def on_optimization(self):
        """Launch multi-objective optimisation from one or more Excel metric files.

        Opens a dialog for objective/constraint configuration, runs the
        selected algorithm (default NSGA-III), and displays Pareto-optimal
        results.
        """
        # TEMP output
        uid = uuid.uuid4().hex[:8]
        out_dir = os.path.join(self.mw.temp_dir, f"Optimization_{uid}")
        os.makedirs(out_dir, exist_ok=True)
        self.mw.current_output_dir = out_dir
        print(f"[Optimization] TEMP output: {out_dir}")
        
        start = self.mw.last_dir if os.path.isdir(self.mw.last_dir) else ""
        while True:
            excel_files, _ = QFileDialog.getOpenFileNames(self, "Select one or multiple Excel files",
                    start, "Excel Files (*.xlsx *.xls)")
            if not excel_files:
                reply = QMessageBox.question(self, "No files selected",
                            "No Excel files were selected. Would you like to try again?",
                            QMessageBox.Retry | QMessageBox.Cancel)
                if reply == QMessageBox.Cancel:
                    return
                continue
            break

        try:
            df1, max_sulci, max_cell_density = conver_excel(excel_files)
            if df1 is None or df1.empty:
                QMessageBox.warning(self, "Optimization Failed", "No valid rows were found in the selected Excel files.")
                return

            self.mw.last_dir = os.path.dirname(excel_files[0]) or self.mw.last_dir
            opt_dialog = OptimizationOptionsDialog(
                self,
                max_sulci_count=max_sulci,
                max_cell_density=max_cell_density,
            )
            if not opt_dialog.exec():
                return

            self.mw.optimization_objectives = opt_dialog.get_selected_objectives()
            self.mw.optimization_objective_directions = opt_dialog.get_objective_directions()
            self.mw.optimization_constraints = opt_dialog.get_constraints()
            self.mw.optimization_algorithms = opt_dialog.get_selected_algorithms()
            self.mw.optimization_n_gen = opt_dialog.get_termination_criterion()

            if max_sulci is not None:
                print(f"[Optimization] Max SulciCount in selected files: {max_sulci}")
            if max_cell_density is not None:
                print(f"[Optimization] Max CellDensity in selected files: {max_cell_density}")

            results, saved_pngs, n_optimal_results = optimization(
                self,
                df1,
                out_dir,
                objectives=self.mw.optimization_objectives,
                objective_directions=self.mw.optimization_objective_directions,
                constraints=self.mw.optimization_constraints,
                algorithms=self.mw.optimization_algorithms,
                n_gen=self.mw.optimization_n_gen,
            )
            if results is not None:
                print(f"[Optimization] Optimization completed. Results saved in {out_dir}.")
                print(f"[Optimization] Number of optimal results: {n_optimal_results}")
                print("Use File → Save Data As… to copy outputs you want to keep.")

                if isinstance(results, pd.DataFrame) and not results.empty:
                    obj_to_column = {
                        "perimeter_rate": "LGI",
                        "cell_density": "CellDensity",
                        "min_d_value": "MinDepth",
                        "max_min_d_value": "MinDepth",
                        "mean_d_value": "MeanDepth",
                        "max_d_value": "MaxDepth",
                        "area": "area", 
                    }
                    objective_cols = []
                    for obj in self.mw.optimization_objectives:
                        col = obj_to_column.get(obj, obj)
                        if col in results.columns and col not in objective_cols:
                            objective_cols.append(col)
                    cols_to_print = [c for c in ["File"] + objective_cols if c in results.columns]
                    if cols_to_print:
                        print("[Optimization] Pareto results:")
                        print(results[cols_to_print].to_string(index=False))

                    source_paths_seen = []
                    for idx, r in results.reset_index(drop=True).iterrows():
                        metric_path = r.get("__source_excel_path")
                        if not metric_path or not isinstance(metric_path, str):
                            metric_path = excel_files[0]
                        source_paths_seen.append(metric_path)
                        self.mw.metrics_store.record_metric_for(
                            path=metric_path,
                            annotation=f"pareto_optimal_{idx + 1}",
                            source=metric_path,
                            area=r.get("area"),
                            volume=r.get("Volume"),
                            perimeter=r.get("Perimeter"),
                            perimeter_convex=r.get("Perimeter_convex"),
                            lgi=r.get("LGI"),
                            File=r.get("File", f"index_{idx}"),
                            SulciCount=r.get("SulciCount"),
                            MinDepth=r.get("MinDepth"),
                            MaxDepth=r.get("MaxDepth"),
                            MeanDepth=r.get("MeanDepth"),
                        )
                    if source_paths_seen:
                        self.mw._set_current("Optimization", source_paths_seen[0])
                    self.mw.metrics_store.rebuild_for_current()
            else:
                print(f"[Optimization] Optimization failed or was canceled.")
                QMessageBox.warning(self, "Optimization Failed", "Optimization failed or was canceled.")

            if len(saved_pngs) == 1:
                img_array = cv2.imread(saved_pngs[0])
                pm = ViewManager.np_bgr_to_qpixmap(img_array)
                self.mw.image_label.setImage(pm)
                self.mw.view.show_widget(self.mw.image_label)
                self.mw._active_view = "image"

            elif saved_pngs and len(saved_pngs) > 1:
                # Provide a default list of indices if valid_slices is not available
                indices = list(range(len(saved_pngs)))
                self.mw.view.enable_png_navigation(saved_pngs, slice_indices=indices)
                mid = len(saved_pngs) // 2
                self.mw.view.on_slice_slider_changed(mid)


    
        except Exception as ex:
            logger.error("Optimization failed: %s", ex)
            QMessageBox.critical(self, "Optimization Failed", f"{type(ex).__name__}: {ex}")
            return  
        
    
    def on_measure_hausdorff(self):
        """Pick first & second images, convert and save in TEMP, compute hausdorff distance and show the plot."""

        # TEMP output
        uid = uuid.uuid4().hex[:8]
        out_dir = os.path.join(self.mw.temp_dir, f"Huasdorff_{uid}")
        os.makedirs(out_dir, exist_ok=True)
        self.mw.current_output_dir = out_dir
        print(f"[Hausdorff] TEMP output: {out_dir}")

        if  self.mw.current_kind !="image" and self.mw.current_path is None:
            start = self.mw.last_dir if os.path.isdir(self.mw.last_dir) else ""
            First, _ = QFileDialog.getOpenFileName(self, "Select the first image",
                                            start, "Images (*.png *.jpg *.jpeg )")
            if not First:
                return
            
            self.mw.last_dir = os.path.dirname(First)
            self.mw.file_mgr.load_image(First)
        else:
            First = self.mw.current_path
        
        self.mw._enter_adjustment_mode()
        self.mw.statusBar().showMessage("Adjust now. Press Shift+Alt+E to continue.")
        print("[Hausdorff] Adjust now. Press Shift+Alt+E to continue.")
        self.mw.wait_for_resume()   # blocks here; resumes after key press
        self.mw.statusBar().clearMessage()

        result = self.mw.settings.ensure_calibrated()
        if result is None:
            return
        u1, px_size_1 = result

                        
        annotated1, basename1, First_array, label1 =self.mw.annotation_con(out_dir)
        
        self.mw.reset_view()
        
        self.mw.statusBar().showMessage("Select two images to measure the Hausdorff distance")
        print("[Hausdorff] Select two images to measure the Hausdorff distance.")
        
        start = self.mw.last_dir if os.path.isdir(self.mw.last_dir) else ""
        Second, _ =  QFileDialog.getOpenFileName(self, "Select the second image",
                                            start, "Images (*.png *.jpg *.jpeg )")
        if not Second:
            return
        self.mw.file_mgr.load_image(Second)
        self.mw.last_dir = os.path.dirname(Second)

        
        self.mw.statusBar().clearMessage()
        
        self.mw._enter_adjustment_mode()
        self.mw.statusBar().showMessage("Adjust now. Press Shift+Alt+E to continue.")
        print("[Hausdorff] Adjust now. Press Shift+Alt+E to continue.")
        self.mw.wait_for_resume()   # blocks here; resumes after key press
        self.mw.statusBar().clearMessage()
        
        result = self.mw.settings.ensure_calibrated()
        if result is None:
            return
        u2, px_size_2 = result
                
        while True:
            if u1 == u2:
                break
            else:
                btn = QMessageBox.warning(
                    self,
                    "Hausdorff distance",
                    "Both images must use the same measuring units to compute the Hausdorff distance.",
                    QMessageBox.Ok | QMessageBox.Cancel)
                if btn == QMessageBox.Cancel:
                    return
                ok = self.mw.settings.set_image_scale()
                if ok:
                    u2 = self.mw.settings.ensure_units()                 # return unit string or None on cancel
                    px_size_2 = self.mw.image_scales.get(self.mw.current_path, self.mw.pixel_size)
                else:
                    return
            # Units mismatch: ask to retry or cancel


        annotated2,basename2, Second_array, label2= self.mw.annotation_con(out_dir)
        self.mw.reset_view()
        self.mw._exit_adjustment_mode()

        mode = self.mw.ask_align_direction()
    
        try:
            img, hd, d12, d21 = calculate_hausdorff_distance(First_array, Second_array, First_label= label1 or "First", Second_label = label2 or "Second", align_mode= mode,  out_dir=out_dir )
            
            pm = ViewManager.np_bgr_to_qpixmap(img)
            pm2 = ViewManager.np_bgr_to_qpixmap(annotated1)
            pm3 = ViewManager.np_bgr_to_qpixmap(annotated2)
                
            self.mw._pms = [pm, pm2, pm3]
            self.mw._pm_index = 0
            self.mw.image_label.setImage(self.mw._pms[self.mw._pm_index])
            self.mw.view.show_widget(self.mw.image_label)
            # Keep kind/path as the ORIGINAL file; Save View As… will ask user where to save what they see.
                
            self.mw._active_view = "image"
            # Ensure File/Process actions stay enabled
            self.mw._set_current("image", str(First))

            print("[Hausdorff] The Hausdorff distance results:")
            print(f"Between {basename1} and {basename2}: {d12} {u1}")
            print(f"Between {basename2} and {basename1}: {d21} {u1}")
            print(f"Maximum distance: {hd} {u1}")
            
            self.mw.statusBar().showMessage("Use Ctrl+M and Ctrl+Shift+M to switch between images.")

        except Exception as ex:
            logger.error("Hausdorff failed: %s", ex)
            QMessageBox.critical(self, "Hausdorff distance", f"{type(ex).__name__}: {ex}")
    
    def on_pial_to_stl(self):
        """Pick one .pial, convert to STL in TEMP, show it, and keep source in metrics."""
        pial = None
        if not self.mw.current_kind == "Freesurfer":
            start = self.mw.last_dir if os.path.isdir(self.mw.last_dir) else ""
            pial, _ = QFileDialog.getOpenFileName(self, "Select FreeSurfer Pial Surface",
                                                  start, "FreeSurfer Surface (*.pial);;All Files (*)")
            if not pial:
                return
        
            self.mw.last_dir = os.path.dirname(pial)
            
            
        elif len(self.mw.Freesurfer_record) == 2:
            self.mw.on_combined_stl()
            return
        
        else:
            pial = self.mw.Freesurfer_record[0]['path']

        # Save to TEMP (don’t pester user yet)
        uid = uuid.uuid4().hex[:8]
        base = os.path.splitext(os.path.basename(pial))[0]
        temp_out = os.path.join(self.mw.temp_dir, f"{base}_{uid}.stl")

        try:
            print(f"[Pial → STL] TEMP output: {temp_out}")
            saved = pial_to_stl(pial, temp_out)

            # Show it immediately
            self.mw.file_mgr.load_stl(saved)                   # shows in VTK window
            print("[Pial → STL] Hint: use File → Save Data As… to keep a permanent copy.")
        except Exception as ex:
            logger.error("Pial → STL failed: %s", ex)
            QMessageBox.critical(self, "Pial → STL", f"{type(ex).__name__}: {ex}")
        
    def on_combined_stl(self):
        """Pick rh & lh .pial, convert + merge in TEMP, show combined STL, record provenance."""
        rh, lh = None, None
        if self.mw.current_kind != "Freesurfer" or  len(self.mw.Freesurfer_record) == 1:
            start = self.mw.last_dir if os.path.isdir(self.mw.last_dir) else ""
            while True:
                files, _ = QFileDialog.getOpenFileNames(self, "Select Both hemisphere (e.g. rh.pial, lh.pial)",
                                                    start, "FreeSurfer Surface (*.pial *.white *.inflated);;All Files (*)")
                                                    
                if not files:
                    return
                
                self.mw.last_dir = os.path.dirname(files[0])
                if len(files) != 2:
                    QMessageBox.warning(self, "Invalid selection", "You must select exactly two files.")
                    continue
                
                names=set()
                exts=set()
                for f in files:
                    base = os.path.basename(f)
                    name, ext = os.path.splitext(base)
                    names.add(name)
                    exts.add(ext)
                if not {"lh", "rh"}.issubset(names):
                    QMessageBox.warning(
                        self,
                        "Invalid selection",
                        "You must select both 'lh' and 'rh' files (e.g., lh.pial and rh.pial)."
                    )
                    continue
                
                if len(exts) != 1:
                    reply = QMessageBox.question(
                    self,
                    "Confirm",
                    "You have selected two different file types. Would you like to proceed?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                   )
                    if reply == QMessageBox.Yes:
                        break
                    else:
                        continue
                
                rh = files[0]; lh = files[1]
                break
        else:
            rh = self.mw.Freesurfer_record[0]['path']
            lh = self.mw.Freesurfer_record[1]['path']

        # TEMP output
        uid = uuid.uuid4().hex[:8]
        temp_out = os.path.join(self.mw.temp_dir, f"brain_both_{uid}.stl")

        try:
            print(f"[Combined STL] TEMP output (combined): {temp_out}")
            saved = pial_pair_to_combined_stl(rh, lh, temp_out)

            # Show combined STL
            self.mw.file_mgr.load_stl(saved)
            print("[Combined STL] Combined STL loaded. Use File → Save Data As… to export.")
        except Exception as ex:
            logger.error("Combined STL failed: %s", ex)
            QMessageBox.critical(self, "Pial (rh & lh) → Combined STL", f"{type(ex).__name__}: {ex}")
        
