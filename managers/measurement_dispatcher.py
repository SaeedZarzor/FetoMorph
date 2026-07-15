"""Measurement dispatcher -- all measurement/processing operations.

Extracted from MainWindow to consolidate measurement logic.
"""

from __future__ import annotations

import math

from deps import *
from typing import TYPE_CHECKING
from functions.nifti_to_image import draw_new_scale_bar
from functions.curvature import compute_curvature_profile, save_curvature_plot
from functions.hausdorff import calculate_hausdorff_distance
from functions.measurement_batch import process_on_images_batch
from functions.measurements_nifti import *
from functions.measurements_image import *
from functions.measurements_stl import *
from functions.measurements_vtk import *
from functions.optimization import OBJ_TO_COLUMN, optimization
from functions.pial_to_stl import pial_pair_to_combined_stl, pial_to_stl
from helpers.helpers import compactness_2D, compactness_3D
from helpers.gestational_week_profile import (
    GestationalWeekProfile, GASPResult, GASPSummary, METRIC_MAP, NORMALIZED_METRIC_MAP,
    MetricStats, WeekProfile, compute_similarity_scores, _augment_normalized_metrics,
)
from helpers.read_excel import conver_excel
from managers.metrics_store import MetricsStore
from managers.view_manager import ViewManager
from widgets.optimization_widgets import OptimizationOptionsDialog

if TYPE_CHECKING:
    from FetoMorph import MainWindow

logger = logging.getLogger("fetomorph")


def _fmt_optional(value, precision: int = 2) -> str:
    if value is None:
        return "None"
    try:
        if math.isnan(float(value)):
            return "NaN"
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


class MeasurementDispatcher:
    """Dispatches measurement operations; reads all state from MainWindow."""

    def __init__(self, mw: MainWindow) -> None:
        self.mw = mw

    def _ensure_stl_slice_direction(self) -> bool:
        """Prompt the user to choose a slice direction for STL processing.

        Sets ``self.mw.slice_direction`` and returns True if the user
        confirmed, False if they cancelled.
        """
        items = ["X (Sagittal)", "Y (Coronal)", "Z (Axial)"]
        current_idx = {"X": 0, "Y": 1, "Z": 2}.get(self.mw.slice_direction, 1)
        choice, ok = QInputDialog.getItem(
            self.mw, "Slice Direction",
            "Choose the slicing axis:",
            items, current_idx, False,
        )
        if not ok:
            return False
        self.mw.slice_direction = choice[0]  # first char: "X", "Y", or "Z"
        return True

    def _measure_planar_vtk(self, mode: str = "allmarks"):
        """Measure a planar VTK mesh by capturing a 2D screenshot and running image measurements.

        Args:
            mode: One of "allmarks", "perimeter", "area", "lGI", "sulci_depth".
        """
        t0 = time.time()
        try:
            # Ensure geometry
            if all(v == 0 for v in self.mw.physical_dim):
                self.mw.settings.load_mesh_and_ask_geometry()

            u = self.mw.units_length

            # Read mesh bounds
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
            depth_sets = None
            if mode == "allmarks":
                area, perimeter, perimeter_internal, perimeter_outer_envelope, lGI, compactness, depth, depth_sets, annotated_bgr, slice_kind = compute_image_allmarks(
                    img_path, pixel_size=pixel_size, kernel_size_mm=self.mw.settings.kernel_size_mm,
                    cnt_threshold=self.mw.cnt_threshold, unit=u, add_scalebar=False,
                    draw_hallmarks=self.mw.draw_hallmarks_on_image, contour_mode=self.mw.contour_mode)
            elif mode == "perimeter":
                perimeter, perimeter_internal, annotated_bgr, slice_kind = compute_image_perimeter(
                    img_path, pixel_size=pixel_size, cnt_threshold=self.mw.cnt_threshold, unit=u, add_scalebar=False,
                    draw_hallmarks=self.mw.draw_hallmarks_on_image, contour_mode=self.mw.contour_mode)
            elif mode == "area":
                area, annotated_bgr, slice_kind = compute_image_area(
                    img_path, pixel_size=pixel_size, cnt_threshold=self.mw.cnt_threshold, unit=u, add_scalebar=False,
                    draw_hallmarks=self.mw.draw_hallmarks_on_image, contour_mode=self.mw.contour_mode)
            elif mode == "lGI":
                lGI, perimeter, perimeter_outer_envelope, annotated_bgr, slice_kind = compute_image_lGI(
                    img_path, pixel_size=pixel_size, kernel_size_mm=self.mw.settings.kernel_size_mm,
                    cnt_threshold=self.mw.cnt_threshold, unit=u, add_scalebar=False,
                    draw_hallmarks=self.mw.draw_hallmarks_on_image)
            elif mode == "sulci_depth":
                depth, depth_sets, annotated_bgr, slice_kind = compute_image_sulci_depth(
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
                    kernel_size_mm=self.mw.settings.kernel_size_mm, kernel_size_px=self.mw.settings.kernel_size_px(pixel_size), area=area, perimeter=perimeter,
                    perimeter_internal=perimeter_internal, contour_mode=self.mw.contour_mode,
                    perimeter_outer_envelope=perimeter_outer_envelope, lgi=lGI, compactness=compactness,
                    sulci_depth=depth, sulci_depth_sets=depth_sets, slice_kind=slice_kind)
                print(f"[Planar VTK | All hallmarks] area={area:.2f} {u}^2, perimeter={perimeter:.2f} {u}, GI={_fmt_optional(lGI)}")
                print(f"[Planar VTK | All hallmarks] Maximum Sulci Depth = {MetricsStore.depth_summary(depth, u)}")

            elif mode == "perimeter":
                self.mw.metrics_store.record_metric_for(img_path, unit=u, dimensions=self.mw.physical_dim,
                    perimeter=perimeter, perimeter_internal=perimeter_internal,
                    contour_mode=self.mw.contour_mode, slice_kind=slice_kind)
                print(f"[Planar VTK | Perimeter] perimeter={perimeter:.2f} {u}"
                      + (f", interior={perimeter_internal:.2f} {u}" if perimeter_internal is not None else ""))
            elif mode == "area":
                self.mw.metrics_store.record_metric_for(img_path, unit=u, dimensions=self.mw.physical_dim,
                    area=area, contour_mode=self.mw.contour_mode, slice_kind=slice_kind)
                print(f"[Planar VTK | Area] area={area:.2f} {u}^2")
            elif mode == "lGI":
                self.mw.metrics_store.record_metric_for(img_path, unit=u, dimensions=self.mw.physical_dim,
                    kernel_size_mm=self.mw.settings.kernel_size_mm, kernel_size_px=self.mw.settings.kernel_size_px(pixel_size), lgi=lGI, slice_kind=slice_kind)
                print(f"[Planar VTK | LGI] GI={_fmt_optional(lGI)}")
            elif mode == "compactness":
                self.mw.metrics_store.record_metric_for(img_path, unit=u, dimensions=self.mw.physical_dim,
                    kernel_size_mm=self.mw.settings.kernel_size_mm, kernel_size_px=self.mw.settings.kernel_size_px(pixel_size), compactness=compactness, slice_kind=slice_kind)
                print(f"[Planar VTK | Compactness] Compactness={_fmt_optional(compactness)}")
            elif mode == "sulci_depth":
                self.mw.metrics_store.record_metric_for(img_path, unit=u, dimensions=self.mw.physical_dim,
                    sulci_depth=depth, sulci_depth_sets=depth_sets, slice_kind=slice_kind)
                if isinstance(depth, (list, tuple)) and len(depth) > 0:
                    summary = ", ".join(f"{float(v):.2f}" for v in depth[:3])
                    print(f"[Planar VTK | Sulci Depth] Maximum depths = {MetricsStore.depth_summary(depth, u)}")

            dt = time.time() - t0
            print(f"[Planar VTK | {mode}] Done in {dt:.2f}s.")

        except Exception as ex:
            logger.error("Planar VTK {mode} failed: %s", ex)
            QMessageBox.critical(self.mw, f"Planar VTK {mode} Failed", f"{type(ex).__name__}: {ex}")

    # ---------- Process menu (stubs) ----------
    def on_measure_allmarks(self):
        """Process → Measures → All hallmarks: compute and show annotated result WITHOUT saving."""
        if not self.mw.current_path or not os.path.isfile(self.mw.current_path):
            print("[Image All hallmarks] No image file is loaded."); return
        if self.mw.current_kind == "image":
            try:
                result = self.mw.settings.ensure_calibrated()
                if result is None:
                    return
                u, px_size = result

                print(f"[Image | All hallmarks] Measuring: {self.mw.current_path}")
                print(f"[Image | All hallmarks] Measuring with pixel size = {px_size} {u}/pixel")

                image_path = self.mw.current_path
                if self.mw.last_annotated_path is not None:
                    image_path = self.mw.last_annotated_path
                    
                area, perimeter, perimeter_internal, perimeter_outer_envelope, lGI, compactness, depth, depth_sets, annotated_bgr, slice_kind = compute_image_allmarks(
                    image_path,
                    pixel_size=px_size,
                    kernel_size_mm=self.mw.settings.kernel_size_mm,
                    cnt_threshold=self.mw.cnt_threshold,
                    unit = u,
                    add_scalebar=not bool(self.mw.image_scale_from_scalebar.get(self.mw.current_path, False)),
                    draw_hallmarks=self.mw.draw_hallmarks_on_image,
                    perimeter_method=self.mw.settings.perimeter_method,
                    simplify_contours_for_perimeter=self.mw.settings.simplify_contours_for_perimeter,
                    contour_simplify_epsilon=self.mw.settings.contour_simplify_epsilon,
                    contour_mode=self.mw.contour_mode,
                )
                
                print(f"[Image | All hallmarks] Results:")
                print(f"\tAnnotated area = {area:.2f} {u}^2.")
                print(f"\tAnnotated Perimeter = {perimeter:.2f} {u}.")
                print(f"\tClosed-envelope perimeter = {perimeter_outer_envelope:.2f} {u}.")
                print(f"\tLGI (Closed-envelope perimeter / Perimeter) = {_fmt_optional(lGI)} .")
                print(f"\tCompactness = {_fmt_optional(compactness)} .")
                print(f"\tMaximum Sulci Depth = {MetricsStore.depth_summary(depth, u)}")
                
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
                    kernel_size_mm=self.mw.settings.kernel_size_mm,
                    kernel_size_px=self.mw.settings.kernel_size_px(px_size),
                    area=area,
                    perimeter=perimeter,
                    perimeter_internal=perimeter_internal,
                    contour_mode=self.mw.contour_mode,
                    perimeter_outer_envelope = perimeter_outer_envelope,
                    lgi=lGI,
                    compactness=compactness,
                    sulci_depth = depth,
                    sulci_depth_sets = depth_sets,
                    slice_kind = slice_kind)
                    
            except Exception as ex:
                logger.error("Image All hallmarks failed: %s", ex)
                QMessageBox.critical(self.mw, "[Image | All hallmarks] Failed", f"{type(ex).__name__}: {ex}")
        elif self.mw.current_kind == "nifti":
            t0 = time.time()
            try:
                nif_path = self.mw.current_path
                print(f"[NIfTI | All hallmarks] Computing hallmarks from: {nif_path}")

                uid = uuid.uuid4().hex[:8]
                out_dir = os.path.join(self.mw.temp_dir, f"nifti_allmarks_{uid}")
                os.makedirs(out_dir, exist_ok=True)
                
                self.mw.current_output_dir = out_dir
                
                labels = self.mw.nifti_selected_regions if self.mw.nifti_selected_regions else self.mw.labels_available
                dims, area, volume, gi, depth, saved_pngs, valid_slices = compute_nifti_allmarks(self.mw, file_path=nif_path,
                out_dir=out_dir, valid_labels=labels, min_contour_area=self.mw.cnt_threshold, kernel_size_mm=self.mw.settings.kernel_size_mm,
                cavity_correction_enabled=self.mw.settings.cavity_correction_enabled, cavity_area_threshold_mm2=self.mw.settings.cavity_area_threshold_mm2,
                perimeter_method=self.mw.settings.perimeter_method,
                simplify_contours_for_perimeter=self.mw.settings.simplify_contours_for_perimeter,
                contour_simplify_epsilon=self.mw.settings.contour_simplify_epsilon)
            
                if area is None:
                    return

                # record metrics (consistent with your global export; units in mm unless noted)
                self.mw.metrics_store.record_metric_for(
                    self.mw.current_path,
                    kernel_size_mm=self.mw.settings.kernel_size_mm,
                    dimensions = dims,
                    unit = "cm",
                    volume=volume,
                    area=area,
                    lgi=gi,
                    sulci_depth = depth)
                self.mw.view.enable_png_navigation(saved_pngs, slice_indices=valid_slices)

                mid = len(saved_pngs) // 2
                self.mw.view.on_slice_slider_changed(mid)
                
                print(f"[NIfTI | All hallmarks] Results:")
                print(f"\tVolume Result = {volume:.2f} cm^3.")
                print(f"\tOuter Surface Area Result = {area:.2f} cm^2.")
                print(f"\tGI (Closed-envelope surface area/ surfacearea) = {gi:.2f} .")
                print(f"\tMaximum Sulci Depth = {MetricsStore.depth_summary(depth, 'cm')}")

                    
                dt = time.time() - t0
                print(f"[NIfTI | All hallmarks] Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")

            except Exception as ex:
                logger.error("NIfTI All hallmarks failed: %s", ex)
                QMessageBox.critical(self.mw, "[NIfTI | All hallmarks] Failed", f"{type(ex).__name__}: {ex}")
            return
            
        elif self.mw.current_kind == "stl":
            if not self._ensure_stl_slice_direction():
                return
            t0 = time.time()
            try:
                uid = uuid.uuid4().hex[:8]
                out_dir = os.path.join(self.mw.temp_dir, f"STL_allmarks_{uid}")
                os.makedirs(out_dir, exist_ok=True)

                self.mw.current_output_dir = out_dir
                source_label, dims, area, volume, gi, compactness ,depth, saved_pngs, valid_slices = compute_stl_allmarks(self.mw, file_path=self.mw.current_path,     out_dir=out_dir, min_contour_area=self.mw.cnt_threshold,
                kernel_size_mm=self.mw.settings.kernel_size_mm, slice_thickness=self.mw.slice_thickness, Slice_direction=self.mw.slice_direction,
                cavity_correction_enabled=self.mw.settings.cavity_correction_enabled, cavity_area_threshold_mm2=self.mw.settings.cavity_area_threshold_mm2,
                perimeter_method=self.mw.settings.perimeter_method,
                simplify_contours_for_perimeter=self.mw.settings.simplify_contours_for_perimeter,
                contour_simplify_epsilon=self.mw.settings.contour_simplify_epsilon,
                fill_cross_section=self.mw.settings.fill_cross_section)

                if source_label == "not_brain":
                    QMessageBox.warning(self.mw, "Mesh ignored", "The computation has been canceled")
                    return
                elif area is None:
                    return

                # record metrics (consistent with your global export; units in mm unless noted)
                self.mw.metrics_store.record_metric_for(
                    self.mw.current_path,
                    source = source_label,
                    kernel_size_mm=self.mw.settings.kernel_size_mm,
                    dimensions = dims,
                    unit = "cm",
                    slice_thickness= self.mw.slice_thickness,
                    volume=volume,
                    area=area,
                    compactness=compactness,
                    sulci_depth = depth,
                    lgi=gi)

                self.mw.view.two_mode_view(out_dir, saved_pngs, valid_slices)

                
                print(f"[STL | All hallmarks] Results:")
                print(f"\tVolume Result = {volume:.2f} cm^3.")
                print(f"\tOuter Surface Area Result = {area:.2f} cm^2.")
                print(f"\tGI (Closed-envelope surface area/ surfacearea) = {gi:.2f} .")
                print(f"\tCompactness = {compactness:.2f} .")
                print(f"\tThe Maximum Grooves Depth = {MetricsStore.depth_summary(depth, 'cm')}")

                if compactness > 1.0:
                    QMessageBox.warning(self.mw, "Compactness Warning",
                        f"Compactness = {compactness:.2f} exceeds 1.0.\n"
                        "The expected range is [0, 1]. This may indicate incorrect "
                        "physical dimensions or unit settings.")

                dt = time.time() - t0
                print(f"[STL | All hallmarks] Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")
            
            except Exception as ex:
                logger.error("STL All hallmarks failed: %s", ex)
                QMessageBox.critical(self.mw, "[STL | All hallmarks] Failed", f"{type(ex).__name__}: {ex}")
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
                area, volume, gi, compactness ,depth, saved_pngs, valid_slices = compute_vtk_allmarks(self.mw, file_path=self.mw.current_path, out_dir=out_dir, min_contour_area=self.mw.cnt_threshold,
                    kernel_size_mm=self.mw.settings.kernel_size_mm, Slice_direction=self.mw.slice_direction, Physical_dim=self.mw.physical_dim, unit=u, slice_thickness=self.mw.slice_thickness,
                    cavity_correction_enabled=self.mw.settings.cavity_correction_enabled, cavity_area_threshold_mm2=self.mw.settings.cavity_area_threshold_mm2,
                    perimeter_method=self.mw.settings.perimeter_method,
                    simplify_contours_for_perimeter=self.mw.settings.simplify_contours_for_perimeter,
                    contour_simplify_epsilon=self.mw.settings.contour_simplify_epsilon,
                    fill_cross_section=self.mw.settings.fill_cross_section)

                if area is None:
                    return
       
                # record metrics (consistent with your global export; units in mm unless noted)
                self.mw.metrics_store.record_metric_for(
                    self.mw.current_path,
                    direction = self.mw.slice_direction,
                    kernel_size_mm=self.mw.settings.kernel_size_mm,
                    unit = u,
                    dimensions = self.mw.physical_dim,
                    slice_thickness= self.mw.slice_thickness,
                    contour_mode = self.mw.contour_mode,
                    volume=volume,
                    area=area,
                    compactness=compactness,
                    sulci_depth = depth,
                    lgi=gi)
                    
                
                self.mw.view.two_mode_view(out_dir, saved_pngs, valid_slices)
                
                print(f"[VTK | All hallmarks] Results:")
                print(f"\tVolume Result = {volume:.2f} {u}^3.")
                print(f"\tEnclosing Surface Area Result = {area:.2f} {u}^2.")
                print(f"\tGI (Closed-envelope surface area/ surfacearea) = {gi:.2f} .")
                print(f"\tCompactness = {compactness:.2f} .")
                print(f"\tMaximum Sulci Depth = {MetricsStore.depth_summary(depth, u)}")

                if compactness > 1.0:
                    QMessageBox.warning(self.mw, "Compactness Warning",
                        f"Compactness = {compactness:.2f} exceeds 1.0.\n"
                        "The expected range is [0, 1]. This may indicate incorrect "
                        "physical dimensions or unit settings.")

                dt = time.time() - t0
                print(f"[VTK | All hallmarks] Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")
                      
            except Exception as ex:
                logger.error("VTK All hallmarks failed: %s", ex)
                QMessageBox.critical(self.mw, "[VTK | All hallmarks] Failed", f"{type(ex).__name__}: {ex}")
                return
            
        else:
            print("[VTK | All hallmarks] Unsupported current kind.")


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
                print(f"[NIfTI | Volume] Computing Volume from: {nif_path}")


                uid = uuid.uuid4().hex[:8]
                out_dir = os.path.join(self.mw.temp_dir, f"nifti_volume_{uid}")
                os.makedirs(out_dir, exist_ok=True)
                
                self.mw.current_output_dir = out_dir
                labels = self.mw.nifti_selected_regions if self.mw.nifti_selected_regions else self.mw.labels_available

                dims, volume,saved_pngs, valid_slices = compute_nifti_volume(self.mw, file_path=nif_path, out_dir=out_dir, valid_labels = labels,
                    cavity_correction_enabled=self.mw.settings.cavity_correction_enabled, cavity_area_threshold_mm2=self.mw.settings.cavity_area_threshold_mm2)
            
                if volume is None:
                    return

                # record metrics (consistent with your global export; units in mm unless noted)
                self.mw.metrics_store.record_metric_for(self.mw.current_path, unit="cm", dimensions = dims, volume = volume,)

                self.mw.view.enable_png_navigation(saved_pngs, slice_indices=valid_slices)
                
                mid = len(saved_pngs) // 2
                self.mw.view.on_slice_slider_changed(mid)
                
                print(f"[NIfTI | Volume] The Brain Volume Result = {volume:.2f} cm^3. ")
                dt = time.time() - t0
                print(f"[NIfTI | Volume] Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")

            except Exception as ex:
                logger.error("NIfTI volume failed: %s", ex)
                QMessageBox.critical(self.mw, "[NIfTI | Volume] Failed", f"{type(ex).__name__}: {ex}")
            return
        elif self.mw.current_kind == "stl":
            if not self._ensure_stl_slice_direction():
                return
            t0 = time.time()
            try:
                uid = uuid.uuid4().hex[:8]
                out_dir = os.path.join(self.mw.temp_dir, f"STL_volume_{uid}")
                os.makedirs(out_dir, exist_ok=True)
                
                self.mw.current_output_dir = out_dir
                source_label, dims,volume, saved_pngs, valid_slices = compute_stl_volume(self.mw, file_path=self.mw.current_path,     out_dir=out_dir, min_contour_area=self.mw.cnt_threshold, slice_thickness=self.mw.slice_thickness, Slice_direction=self.mw.slice_direction,
                fill_cross_section=self.mw.settings.fill_cross_section,
                cavity_correction_enabled=self.mw.settings.cavity_correction_enabled, cavity_area_threshold_mm2=self.mw.settings.cavity_area_threshold_mm2)
            
                if source_label == "not_brain":
                    QMessageBox.warning(self.mw, "[STL | Volume] Mesh ignored", "The computation has been canceled")
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

                
                print(f"[STL | Volume] STL mesh Volume Result = {volume:.2f} cm^3.")


                dt = time.time() - t0
                print(f"[STL | Volume] Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")

            except Exception as ex:
                logger.error("STL Volume failed: %s", ex)
                QMessageBox.critical(self.mw, "[STL | Volume] Failed", f"{type(ex).__name__}: {ex}")
            return
            
        elif self.mw.is_vtk:
            if self.mw._flat_axis is not None:
                print("[Volume] Not applicable for planar 2D meshes.")
                QMessageBox.information(self.mw, "Volume", "Volume measurement is not applicable for planar 2D meshes.")
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
                volume, saved_pngs, valid_slices = compute_vtk_volume(self.mw, file_path=self.mw.current_path, out_dir=out_dir, min_contour_area=self.mw.cnt_threshold,
                    Slice_direction = self.mw.slice_direction, Physical_dim= self.mw.physical_dim, unit = u, slice_thickness=self.mw.slice_thickness,
                    fill_cross_section=self.mw.settings.fill_cross_section,
                    cavity_correction_enabled=self.mw.settings.cavity_correction_enabled, cavity_area_threshold_mm2=self.mw.settings.cavity_area_threshold_mm2)

                if volume is None:
                    return
       
                # record metrics (consistent with your global export; units in mm unless noted)
                self.mw.metrics_store.record_metric_for(
                    self.mw.current_path,
                    direction = self.mw.slice_direction,
                    unit = u,
                    dimensions = self.mw.physical_dim,
                    slice_thickness= self.mw.slice_thickness,
                    contour_mode = self.mw.contour_mode,
                    volume=volume)
                    
                self.mw.view.two_mode_view(out_dir, saved_pngs, valid_slices)
                
                print(f"[VTK | Volume] VTK mesh Volume Result = {volume:.2f} {u}^3.")

                dt = time.time() - t0
                print(f"[VTK | Volume] Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")
                      
            except Exception as ex:
                logger.error("VTK Volume failed: %s", ex)
                QMessageBox.critical(self.mw, "[VTK | Volume] Failed", f"{type(ex).__name__}: {ex}")
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

                print(f"[Image | Perimeter] Measuring: {self.mw.current_path}")
                print(f"[Image | Perimeter] Measuring with pixel size = {px_size} {u}/pixel")

                image_path = self.mw.current_path
                if self.mw.last_annotated_path is not None:
                    image_path = self.mw.last_annotated_path
                
                perimeter, perimeter_internal, annotated_bgr, slice_kind = compute_image_perimeter(
                    image_path,
                    pixel_size = px_size,
                    cnt_threshold = self.mw.cnt_threshold,
                    unit = u,
                    add_scalebar=not bool(self.mw.image_scale_from_scalebar.get(self.mw.current_path, False)),
                    draw_hallmarks=self.mw.draw_hallmarks_on_image,
                    perimeter_method=self.mw.settings.perimeter_method,
                    simplify_contours_for_perimeter=self.mw.settings.simplify_contours_for_perimeter,
                    contour_simplify_epsilon=self.mw.settings.contour_simplify_epsilon,
                    contour_mode=self.mw.contour_mode,
                )
                print(f"[Image | Perimeter] Results:")
                print(f"\tAnnotated Perimeter = {perimeter:.2f} {u}."
                      + (f" \tInterior Perimeter = {perimeter_internal:.2f} {u}." if perimeter_internal is not None else ""))
                
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
                    perimeter=perimeter,
                    perimeter_internal=perimeter_internal,
                    contour_mode=self.mw.contour_mode,
                    slice_kind=slice_kind)

            except Exception as ex:
                logger.error("Image Perimeter failed: %s", ex)
                QMessageBox.critical(self.mw, "[Image | Perimeter] Failed", f"{type(ex).__name__}: {ex}")
            
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

                    contour_mode_used = None  # STL path does not subtract internal contours
                    if self.mw.current_kind == "stl":
                        if not self._ensure_stl_slice_direction():
                            return
                        source_label, dims, comp, saved_pngs, valid_slices = compute_compactness_stl(
                            self, file_path=self.mw.current_path, out_dir=out_dir,
                            min_contour_area=self.mw.cnt_threshold, slice_thickness=self.mw.slice_thickness,
                            Slice_direction=self.mw.slice_direction,
                            cavity_correction_enabled=self.mw.settings.cavity_correction_enabled,
                            cavity_area_threshold_mm2=self.mw.settings.cavity_area_threshold_mm2)
                        if source_label == "not_brain":
                            return

                    elif self.mw.is_vtk:
                        if all(v == 0 for v in self.mw.physical_dim):
                            self.mw.settings.load_mesh_and_ask_geometry()
                        comp, saved_pngs, valid_slices = compute_compactness_vtk(self, file_path=self.mw.current_path,
                        out_dir=out_dir, min_contour_area=self.mw.cnt_threshold,
                        Slice_direction=self.mw.slice_direction, Physical_dim=self.mw.physical_dim,
                        unit=self.mw.units_length, slice_thickness=self.mw.slice_thickness,
                        cavity_correction_enabled=self.mw.settings.cavity_correction_enabled,
                        cavity_area_threshold_mm2=self.mw.settings.cavity_area_threshold_mm2)
                        contour_mode_used = None  # VTK uses the cavity correction, not a manual mode

                    if comp is None:
                        return

                    self.mw.view.two_mode_view(out_dir, saved_pngs, valid_slices)

                self.mw.metrics_store.record_metric_for(
                    self.mw.current_path,
                    slice_thickness=self.mw.slice_thickness,
                    contour_mode=contour_mode_used,
                    compactness=comp)

                base_name = os.path.basename(self.mw.current_path)
                print(f"[3D | Compactness] for {base_name}: Compactness(3D)={comp:.4f}")
                if comp > 1.0:
                    QMessageBox.warning(self.mw, "Compactness Warning",
                        f"Compactness = {comp:.4f} exceeds 1.0.\n"
                        "The expected range is [0, 1]. This may indicate incorrect "
                        "physical dimensions or unit settings.")
                dt = time.time() - t0
                print(f"[3D | Compactness] Done in {dt:.2f}s.")

            except Exception as ex:
                logger.error("Compactness failed: %s", ex)
                QMessageBox.critical(self.mw, "[3D | Compactness] Failed", f"{type(ex).__name__}: {ex}")
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

                slice_kind = None
                if area is not None and perimeter is not None:
                    area = float(area)
                    perimeter = float(perimeter)
                    compactness_2D_value = compactness_2D(area, perimeter)
                else:
                    compactness_2D_value, annotated_bgr, slice_kind = compute_compactness_2D(image_path, cnt_threshold=self.mw.cnt_threshold, pixel_size=self.mw.pixel_size)
                    pm = ViewManager.np_bgr_to_qpixmap(annotated_bgr)
                    self.mw.image_label.setImage(pm)
                    self.mw.image_label.remove_last_annotation()
                    self.mw.view.show_widget(self.mw.image_label)
                    self.mw._active_view = "image"

                print(f"[Image | Compactness] Results:")
                print(f"\tCompactness = {_fmt_optional(compactness_2D_value)} .")
                if compactness_2D_value > 1.0:
                    QMessageBox.warning(self.mw, "Image | Compactness Warning",
                        f"Compactness = {compactness_2D_value:.4f} exceeds 1.0.\n"
                        "The expected range is [0, 1]. This may indicate an issue "
                        "with contour detection or image quality.")
                self.mw._set_current("image", self.mw.current_path)

            except Exception as ex:
                logger.error("Image Compactness failed: %s", ex)
                QMessageBox.critical(self.mw, "[Image | Compactness] Failed", f"{type(ex).__name__}: {ex}")
        else:
            QMessageBox.information(self.mw, "Compactness", "Compactness measurement is currently only supported for 2D images and 3D meshes. Please open an image or 3D mesh file.")      
            print("[Compactness] Unsupported current kind. Open an image or 3D mesh file.")
            return

    def on_measure_curve_length(self):
        """Process → Measures → Curve Length: measure only the longest curved segment."""
        if not self.mw.current_path or not os.path.isfile(self.mw.current_path):
            print("[Curve Length] No file is loaded."); return

        if self.mw.current_kind == "image":
            try:
                result = self.mw.settings.ensure_calibrated()
                if result is None:
                    return
                u, px_size = result

                print(f"[Image | Curve Length] Measuring: {self.mw.current_path}")
                print(f"[Image | Curve Length] Measuring with pixel size = {px_size} {u}/pixel")

                image_path = self.mw.current_path
                if self.mw.last_annotated_path is not None:
                    image_path = self.mw.last_annotated_path

                curved_length, annotated_bgr, slice_kind = compute_image_curved_length(
                    image_path,
                    pixel_size=px_size,
                    cnt_threshold=self.mw.cnt_threshold,
                    unit=u,
                    add_scalebar=not bool(self.mw.image_scale_from_scalebar.get(self.mw.current_path, False)),
                    draw_hallmarks=self.mw.draw_hallmarks_on_image,
                )
                print(f"[Image | Curve Length] Curved length = {curved_length:.2f} {u}.")

                label_text = self.mw.get_label_for_cropped_path(image_path)
                if label_text:
                    annotated_bgr = put_label_on_bgr(annotated_bgr, label_text, pos="topleft")
                pm = ViewManager.np_bgr_to_qpixmap(annotated_bgr)
                self.mw.image_label.setImage(pm)
                self.mw.image_label.remove_last_annotation()
                self.mw.view.show_widget(self.mw.image_label)
                self.mw._active_view = "image"
                self.mw._set_current("image", self.mw.current_path)
                self.mw.metrics_store.record_metric_for(
                    self.mw.current_path,
                    label=label_text,
                    pixel_size_units=f"{self.mw.units_length}/pixel",
                    unit=self.mw.units_length,
                    pixel_size=self.mw.pixel_size,
                    curved_length=curved_length,
                    slice_kind=slice_kind,
                )

            except Exception as ex:
                logger.error("Curve Length failed: %s", ex)
                QMessageBox.critical(self.mw, "[Image | Curve Length] Failed", f"{type(ex).__name__}: {ex}")

        else:
            print("[Image | Curve Length] Only supported for images.")
            return

    def on_measure_straight(self):
        """Process → Measures → Straight Line: interactive two-click distance measurement."""
        if not self.mw.current_path or not os.path.isfile(self.mw.current_path):
            print("[Image | Straight Line] No file is loaded."); return

        if self.mw.current_kind != "image":
            print("[Image | Straight Line] Only supported for images."); return

        # ensure calibration
        result = self.mw.settings.ensure_calibrated()
        if result is None:
            return
        u, px_size = result

        print(f"[Image | Straight Line] Click two points on the image to measure distance.")

        def _finish(pixel_length, p1, p2):
            distance = pixel_length * px_size
            self.mw.image_label.add_line_annotation(
                p1, p2, label=f"{distance:.2f} {u}", color=QColor(0, 200, 255))
            self.mw.metrics_store.record_metric_for(
                self.mw.current_path,
                unit=u,
                pixel_size=px_size,
                straight_line_distance=distance)
            print(f"[Image | Straight Line] Distance = {distance:.2f} {u}")

        self.mw.image_label.start_line_measure(_finish)
    
    def on_measure_lgi(self):
        """Process → Measures → lGI: compute and show annotated result WITHOUT saving."""
        
        if not self.mw.current_path or not os.path.isfile(self.mw.current_path):
            print("[Image | LGI] No file is loaded."); return
        if self.mw.current_kind == "image":
            try:
                result = self.mw.settings.ensure_calibrated()
                if result is None:
                    return
                u, px_size = result

                print(f"[Image | LGI] Measuring: {self.mw.current_path}")

                image_path = self.mw.current_path
                if self.mw.last_annotated_path is not None:
                    image_path = self.mw.last_annotated_path
                def _measure_lgi(k_mm):
                    return compute_image_lGI(
                        image_path,
                        pixel_size = px_size,
                        kernel_size_mm=k_mm,
                        cnt_threshold=self.mw.cnt_threshold,
                        unit = u,
                        add_scalebar=not bool(self.mw.image_scale_from_scalebar.get(self.mw.current_path, False)),
                        draw_hallmarks=self.mw.draw_hallmarks_on_image,
                        perimeter_method=self.mw.settings.perimeter_method,
                        simplify_contours_for_perimeter=self.mw.settings.simplify_contours_for_perimeter,
                        contour_simplify_epsilon=self.mw.settings.contour_simplify_epsilon,
                    )

                # Kernel diameter in px (morphology clamps to a 3 px minimum).
                def _kernel_px(k_mm):
                    return max(3, int(round(float(k_mm) / max(float(px_size), 1e-9))))

                kernel_mm = float(self.mw.settings.kernel_size_mm)
                kernel_px = _kernel_px(kernel_mm)
                lGI, perimeter, perimeter_outer_envelope, annotated_bgr, slice_kind = _measure_lgi(kernel_mm)

                # An LGI < 1 means the closed-envelope perimeter came out longer
                # than the exterior perimeter (usually a too-large close kernel
                # over-smoothing into artefacts). Retry with progressively
                # smaller kernels until LGI > 1 or the 3 px floor is reached.
                MAX_LGI_RETRIES = 25
                retries = 0
                while lGI is not None and lGI < 1.0 and kernel_px > 3 and retries < MAX_LGI_RETRIES:
                    kernel_px = max(3, min(kernel_px - 1, int(round(kernel_px * 0.8))))
                    kernel_mm = kernel_px * float(px_size)
                    retries += 1
                    print(f"[Image | LGI] LGI = {lGI:.3f} < 1 → retrying with a smaller kernel size: "
                          f"{kernel_mm:.3f} {u} ({kernel_px} px).")
                    lGI, perimeter, perimeter_outer_envelope, annotated_bgr, slice_kind = _measure_lgi(kernel_mm)

                if lGI is not None and lGI < 1.0:
                    print(f"[Image | LGI] Note: LGI = {_fmt_optional(lGI)} is still < 1 at the minimum "
                          f"kernel size ({kernel_px} px); cannot reduce further.")

                print(f"[Image | LGI] Results:")
                print(f"\tPerimeter = {perimeter:.2f} {u}.")
                print(f"\tClosed-envelope perimeter = {perimeter_outer_envelope:.2f} {u}.")
                print(f"\tLGI (Closed-envelope perimeter / Perimeter) = {_fmt_optional(lGI)} .")
                if retries:
                    print(f"\t(kernel size auto-reduced to {kernel_mm:.3f} {u} = {kernel_px} px "
                          f"for {retries} retr{'y' if retries == 1 else 'ies'}; kernel size reset to "
                          f"the last set value {float(self.mw.settings.kernel_size_mm):.3f} {u})")

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
                    kernel_size_mm=self.mw.settings.kernel_size_mm,
                    kernel_size_px=self.mw.settings.kernel_size_px(px_size),
                    perimeter=perimeter, perimeter_outer_envelope=perimeter_outer_envelope, lgi=lGI,
                    slice_kind=slice_kind)

            except Exception as ex:
                logger.error("Image LGI failed: %s", ex)
                QMessageBox.critical(self.mw, "[Image | LGI] Failed", f"{type(ex).__name__}: {ex}")
                
        elif self.mw.current_kind == "nifti":
            t0 = time.time()
            reply = QMessageBox.question(self.mw,"Enhance measurement",
            "For accurate LGI computation, please provide the FreeSurfer pial surfaces for both hemispheres (lh.pial and rh.pial). Do you have these files?",   # message
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            
            if reply == QMessageBox.No:
                QMessageBox.warning(self.mw, "NIFTI LGI Input Missing",
                    "The LGI can be computed based on the NIfTI file alone, but the accuracy of the results is not guaranteed.")
                
                try:
                    nif_path = self.mw.current_path
                    print(f"[NIfTI | LGI] Computing lGI from: {nif_path}")


                    uid = uuid.uuid4().hex[:8]
                    out_dir = os.path.join(self.mw.temp_dir, f"nifti_lGI_{uid}")
                    os.makedirs(out_dir, exist_ok=True)
                    
                    self.mw.current_output_dir = out_dir
                    labels = self.mw.nifti_selected_regions if self.mw.nifti_selected_regions else self.mw.labels_available

                    lGI, saved_pngs, valid_slices = compute_nifti_lGI(
                        self.mw, file_path=nif_path, out_dir=out_dir,
                        valid_labels=labels, min_contour_area=self.mw.cnt_threshold,
                        kernel_size_mm=self.mw.settings.kernel_size_mm,
                        perimeter_method=self.mw.settings.perimeter_method,
                        simplify_contours_for_perimeter=self.mw.settings.simplify_contours_for_perimeter,
                        contour_simplify_epsilon=self.mw.settings.contour_simplify_epsilon)
                
                    if lGI is None:
                        return

                    # record metrics (consistent with your global export; units in mm unless noted)
                    self.mw.metrics_store.record_metric_for(self.mw.current_path, kernel_size_mm=self.mw.settings.kernel_size_mm, lgi=lGI)

                    self.mw.view.enable_png_navigation(saved_pngs, slice_indices=valid_slices)
                    
                    mid = len(saved_pngs) // 2
                    self.mw.view.on_slice_slider_changed(mid)
                    
                    print(f"[NIfTI | LGI] The Brain GI (Closed-envelope surface area/ surfacearea) = {lGI:.2f}. ")
                    dt = time.time() - t0
                    print(f"[NIfTI | LGI] Done in {dt:.2f}s. Results live in TEMP.\n"
                          f"Use File → Save Data As… to copy outputs you want to keep.")

                except Exception as ex:
                    logger.error("NIfTI LGI failed: %s", ex)
                    QMessageBox.critical(self.mw, "[NIfTI | LGI] Failed", f"{type(ex).__name__}: {ex}")
                return
        
            elif reply == QMessageBox.Yes:
                nif_path = self.mw.current_path
#                QTimer.singleShot(0, self.mw.on_combined_stl)
                self.mw.on_combined_stl()
                stl_path = self.mw.current_path if (self.mw.current_path and os.path.isfile(self.mw.current_path)) else None
                
                try:
                    print(f"[NIfTI | LGI] Computing lGI from: {nif_path} based on rh & lh .pial")

                    uid = uuid.uuid4().hex[:8]
                    out_dir = os.path.join(self.mw.temp_dir, f"STL_lGI_{uid}")
                    os.makedirs(out_dir, exist_ok=True)
                    
                    self.mw.current_output_dir = out_dir
                    source_label, dims, gi, saved_pngs, valid_slices =compute_stl_lGI(
                        self.mw,
                        file_path=stl_path,
                        out_dir=out_dir,
                        min_contour_area=self.mw.cnt_threshold,
                        kernel_size_mm=self.mw.settings.kernel_size_mm,
                        slice_thickness=self.mw.slice_thickness,
                        build_solid=False,   # keep False for stability
                        Slice_direction=self.mw.slice_direction,
                        perimeter_method=self.mw.settings.perimeter_method,
                        simplify_contours_for_perimeter=self.mw.settings.simplify_contours_for_perimeter,
                        contour_simplify_epsilon=self.mw.settings.contour_simplify_epsilon,
                    )
                                        
                
                    if gi is None:
                        return

                    # record metrics (consistent with your global export; units in mm unless noted)
                    self.mw.metrics_store.record_metric_for(
                        self.mw.current_path,
                        source = source_label,
                        kernel_size_mm=self.mw.settings.kernel_size_mm,
                        dimensions = dims,
                        unit = "cm",
                        slice_thickness= self.mw.slice_thickness,
                        lgi=gi)
                        
                    self.mw.view.enable_png_navigation(saved_pngs, slice_indices=valid_slices)
                    
                    mid = len(saved_pngs) // 2
                    self.mw.view.on_slice_slider_changed(mid)
                    
                    print(f"[STL | LGI] The Brain GI (Closed-envelope surface area/ surfacearea) = {gi:.2f}. ")
                    dt = time.time() - t0
                    print(f"[STL | LGI] Done in {dt:.2f}s. Results live in TEMP.\n"
                          f"Use File → Save Data As… to copy outputs you want to keep.")

                except Exception as ex:
                    logger.error("STL LGI failed: %s", ex)
                    QMessageBox.critical(self.mw, "[STL | LGI] Failed", f"{type(ex).__name__}: {ex}")
                return
                
        elif self.mw.current_kind == "stl":
            if not self._ensure_stl_slice_direction():
                return
            t0 = time.time()
            try:
                uid = uuid.uuid4().hex[:8]
                out_dir = os.path.join(self.mw.temp_dir, f"STL_lgi_{uid}")
                os.makedirs(out_dir, exist_ok=True)
                
                self.mw.current_output_dir = out_dir
                source_label, dims, gi, saved_pngs, valid_slices = compute_stl_lGI(self.mw, file_path=self.mw.current_path,     out_dir=out_dir, min_contour_area=self.mw.cnt_threshold,
                kernel_size_mm=self.mw.settings.kernel_size_mm, slice_thickness=self.mw.slice_thickness, Slice_direction=self.mw.slice_direction,
                perimeter_method=self.mw.settings.perimeter_method,
                simplify_contours_for_perimeter=self.mw.settings.simplify_contours_for_perimeter,
                contour_simplify_epsilon=self.mw.settings.contour_simplify_epsilon)

                if source_label == "not_brain":
                    QMessageBox.warning(self.mw, "Mesh ignored", "The computation has been canceled")
                    return
                elif gi is None:
                    return

                # record metrics (consistent with your global export; units in mm unless noted)
                self.mw.metrics_store.record_metric_for(
                    self.mw.current_path,
                    source = source_label,
                    kernel_size_mm=self.mw.settings.kernel_size_mm,
                    dimensions = dims,
                    unit = "cm",
                    slice_thickness= self.mw.slice_thickness,
                    lgi=gi)

                self.mw.view.two_mode_view(out_dir, saved_pngs, valid_slices)

                
                print(f"[STL | LGI] STL mesh GI (Closed-envelope surface area/ surfacearea) = {gi:.2f} .")

                dt = time.time() - t0
                print(f"[STL | LGI] Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")

            except Exception as ex:
                logger.error("STL LGI failed: %s", ex)
                QMessageBox.critical(self.mw, "[STL | LGI] Failed", f"{type(ex).__name__}: {ex}")
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
                gi, saved_pngs, valid_slices = compute_vtk_lGI(self.mw, file_path=self.mw.current_path, out_dir=out_dir, min_contour_area=self.mw.cnt_threshold,
                    kernel_size_mm=self.mw.settings.kernel_size_mm, Slice_direction=self.mw.slice_direction, Physical_dim=self.mw.physical_dim, unit=u, slice_thickness=self.mw.slice_thickness,
                    perimeter_method=self.mw.settings.perimeter_method,
                    simplify_contours_for_perimeter=self.mw.settings.simplify_contours_for_perimeter,
                    contour_simplify_epsilon=self.mw.settings.contour_simplify_epsilon)

                if gi is None:
                    return
       
                # record metrics (consistent with your global export; units in mm unless noted)
                self.mw.metrics_store.record_metric_for(
                    self.mw.current_path,
                    direction = self.mw.slice_direction,
                    kernel_size_mm=self.mw.settings.kernel_size_mm,
                    unit = u,
                    dimensions = self.mw.physical_dim,
                    slice_thickness= self.mw.slice_thickness,
                    lgi=gi)
                    
                
                self.mw.view.two_mode_view(out_dir, saved_pngs, valid_slices)
                
                print(f"[VTK | LGI] VTK mesh GI (Closed-envelope surface area/ surfacearea) = {gi:.2f} .")

                dt = time.time() - t0
                print(f"[VTK | LGI] Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")
                      
            except Exception as ex:
                logger.error("VTK LGI failed: %s", ex)
                QMessageBox.critical(self.mw, "[VTK | LGI] Failed", f"{type(ex).__name__}: {ex}")
                return
            
        else:
            print("[LGI] Unsupported current kind.")

            
        
    
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

                print(f"[Image Sulci depth] Measuring: {self.mw.current_path}")
                print(f"[Image Sulci depth] Measuring with pixel size = {px_size} {u}/pixel")

                image_path = self.mw.current_path
                if self.mw.last_annotated_path is not None:
                    image_path = self.mw.last_annotated_path
                depth, depth_sets, annotated_bgr, slice_kind = compute_image_sulci_depth(
                    image_path,
                    pixel_size = px_size,
                    cnt_threshold=self.mw.cnt_threshold,
                    unit = u,
                    add_scalebar=not bool(self.mw.image_scale_from_scalebar.get(self.mw.current_path, False))
                )
                print(f"[Image | Sulci depth] Results:")
                print(f"\tMaximum Sulci Depth = {MetricsStore.depth_summary(depth, u)}")
                
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
                    sulci_depth = depth,
                    sulci_depth_sets = depth_sets,
                    slice_kind = slice_kind)

            except Exception as ex:
                logger.error("Image Sulci depth failed: %s", ex)
                QMessageBox.critical(self.mw, "[Image | Sulci depth] Failed", f"{type(ex).__name__}: {ex}")
        
        elif self.mw.current_kind == "nifti":
            t0 = time.time()
            try:
                nif_path = self.mw.current_path
                print(f"[NIfTI] Computing Sulci depth from: {nif_path}")


                uid = uuid.uuid4().hex[:8]
                out_dir = os.path.join(self.mw.temp_dir, f"nifti_Sulci_depth_{uid}")
                os.makedirs(out_dir, exist_ok=True)
                
                self.mw.current_output_dir = out_dir
                labels = self.mw.nifti_selected_regions if self.mw.nifti_selected_regions else self.mw.labels_available

                dims, depth,saved_pngs, valid_slices = compute_nifti_sulci_depth(self.mw, file_path=nif_path, out_dir=out_dir, valid_labels = labels, min_contour_area=self.mw.cnt_threshold)
            
                if depth is None:
                    return

                # record metrics (consistent with your global export; units in mm unless noted)
                self.mw.metrics_store.record_metric_for(self.mw.current_path, unit ="mm", dimensions = dims, sulci_depth = depth,)

                self.mw.view.enable_png_navigation(saved_pngs, slice_indices=valid_slices)
                
                mid = len(saved_pngs) // 2
                self.mw.view.on_slice_slider_changed(mid)
                
                print(f"[NIfTI | Sulci depth] The max Brain Sulci depth across slices = {MetricsStore.depth_summary(depth, 'mm')}")
                dt = time.time() - t0
                print(f"[NIfTI | Sulci depth] Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")

            except Exception as ex:
                logger.error("NIfTI Sulci depth failed: %s", ex)
                QMessageBox.critical(self.mw, "[NIfTI | Sulci depth] Failed", f"{type(ex).__name__}: {ex}")
            return
            
        elif self.mw.current_kind == "stl":
            if not self._ensure_stl_slice_direction():
                return
            t0 = time.time()
            try:
                uid = uuid.uuid4().hex[:8]
                out_dir = os.path.join(self.mw.temp_dir, f"STL_sulic_depth_{uid}")
                os.makedirs(out_dir, exist_ok=True)
                
                self.mw.current_output_dir = out_dir
                source_label, dims, depth, saved_pngs, valid_slices = compute_stl_sulci_depth (self, file_path=self.mw.current_path, out_dir=out_dir, min_contour_area=self.mw.cnt_threshold, slice_thickness=self.mw.slice_thickness, Slice_direction=self.mw.slice_direction)
            
                if source_label == "not_brain":
                    QMessageBox.warning(self.mw, "Mesh ignored", "The computation has been canceled")
                    return
                elif depth is None:
                    return

                # record metrics (consistent with your global export; units in mm unless noted)
                self.mw.metrics_store.record_metric_for(self.mw.current_path, source = source_label, slice_thickness= self.mw.slice_thickness,
                    dimensions = dims,unit ="mm", sulci_depth = depth)

                self.mw.view.two_mode_view(out_dir, saved_pngs, valid_slices)
                
                print(f"[STL | Sulci depth] The max Brain Sulci depth across slices = {MetricsStore.depth_summary(depth, 'mm')}")
                dt = time.time() - t0
                print(f"[STL | Sulci depth] Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")

            except Exception as ex:
                logger.error("STL Sulci depth failed: %s", ex)
                QMessageBox.critical(self.mw, "[STL | Sulci depth] Failed", f"{type(ex).__name__}: {ex}")
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
                depth, saved_pngs, valid_slices = compute_vtk_sulci_depth(self.mw, file_path=self.mw.current_path, out_dir=out_dir, min_contour_area=self.mw.cnt_threshold,
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
                    print("[VTK | Sulci depth]")
                    print(f"The Maximum Grooves Depth = {MetricsStore.depth_summary(depth, u)}")

                dt = time.time() - t0
                print(f"[VTK | Sulci depth] Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")
                      
            except Exception as ex:
                logger.error("VTK Sulci depth failed: %s", ex)
                QMessageBox.critical(self.mw, "[VTK | Sulci depth] Failed", f"{type(ex).__name__}: {ex}")
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
                print("[Image | Area] No image file is loaded."); return
            try:
                result = self.mw.settings.ensure_calibrated()
                if result is None:
                    return
                u, px_size = result

                print(f"[Image | Area] Measuring: {self.mw.current_path}")
                print(f"[Image | Area] Measuring with pixel size = {px_size} {u}/pixel")
                
                image_path = self.mw.current_path
                if self.mw.last_annotated_path is not None:
                    image_path = self.mw.last_annotated_path
                    
                area, annotated_bgr, slice_kind = compute_image_area(
                    image_path,
                    pixel_size=px_size,
                    cnt_threshold=self.mw.cnt_threshold,
                    unit = u,
                    add_scalebar=not bool(self.mw.image_scale_from_scalebar.get(self.mw.current_path, False)),
                    draw_hallmarks=self.mw.draw_hallmarks_on_image,
                    contour_mode=self.mw.contour_mode,
                )
                
                label_text = self.mw.get_label_for_cropped_path(image_path)
                if label_text:
                    annotated_bgr = put_label_on_bgr(annotated_bgr, label_text, pos="topleft")
                    
                print(f"[Image | Area] Results:")
                print(f"\tAnnotated area = {area:.2f} {u}^2.")
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
                area=area,
                contour_mode=self.mw.contour_mode,
                slice_kind=slice_kind)

            except Exception as ex:
                print(f"[Image | Area] ERROR : {ex}")
                QMessageBox.critical(self.mw, "[Image | Area] Failed", f"{type(ex).__name__}: {ex}")
        elif self.mw.current_kind == "nifti":
            t0 = time.time()
            try:
                nif_path = self.mw.current_path
                print(f"[NIfTI] Computing area from: {nif_path}")


                uid = uuid.uuid4().hex[:8]
                out_dir = os.path.join(self.mw.temp_dir, f"nifti_area_{uid}")
                os.makedirs(out_dir, exist_ok=True)
                
                self.mw.current_output_dir = out_dir
                labels = self.mw.nifti_selected_regions if self.mw.nifti_selected_regions else self.mw.labels_available

                dims, area,saved_pngs, valid_slices = compute_nifti_area(
                    self.mw, file_path=nif_path, out_dir=out_dir,
                    valid_labels=labels, min_contour_area=self.mw.cnt_threshold,
                    perimeter_method=self.mw.settings.perimeter_method,
                    simplify_contours_for_perimeter=self.mw.settings.simplify_contours_for_perimeter,
                    contour_simplify_epsilon=self.mw.settings.contour_simplify_epsilon,
                    cavity_correction_enabled=self.mw.settings.cavity_correction_enabled,
                    cavity_area_threshold_mm2=self.mw.settings.cavity_area_threshold_mm2)
            
                if area == 0:
                    QMessageBox.information(self.mw, "NIfTI Area", "All slices were filtered out (too small).")
                    return

                # record metrics (consistent with your global export; units in mm unless noted)
                self.mw.metrics_store.record_metric_for(self.mw.current_path, unit="cm", dimensions = dims, area = area,)

                self.mw.view.enable_png_navigation(saved_pngs, slice_indices=valid_slices)
                
                mid = len(saved_pngs) // 2
                self.mw.view.on_slice_slider_changed(mid)
                
                print(f"[NIfTI | Area] The Brain Outer Surface Area Result = {area:.2f} cm^2. ")
                dt = time.time() - t0
                print(f"[NIfTI | Area] Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")

            except Exception as ex:
                logger.error("NIfTI Area failed: %s", ex)
                QMessageBox.critical(self.mw, "[NIfTI | Area] Failed", f"{type(ex).__name__}: {ex}")
            return
            
        elif self.mw.current_kind == "stl":
            if not self._ensure_stl_slice_direction():
                return
            t0 = time.time()
            try:
                uid = uuid.uuid4().hex[:8]
                out_dir = os.path.join(self.mw.temp_dir, f"STL_area_{uid}")
                os.makedirs(out_dir, exist_ok=True)
                
                self.mw.current_output_dir = out_dir
                source_label, dims,area, saved_pngs, valid_slices = compute_stl_area(self.mw, file_path=self.mw.current_path,     out_dir=out_dir, min_contour_area=self.mw.cnt_threshold, slice_thickness=self.mw.slice_thickness, Slice_direction=self.mw.slice_direction,
                perimeter_method=self.mw.settings.perimeter_method, simplify_contours_for_perimeter=self.mw.settings.simplify_contours_for_perimeter, contour_simplify_epsilon=self.mw.settings.contour_simplify_epsilon,
                fill_cross_section=self.mw.settings.fill_cross_section,
                cavity_correction_enabled=self.mw.settings.cavity_correction_enabled, cavity_area_threshold_mm2=self.mw.settings.cavity_area_threshold_mm2)
            
                if source_label == "not_brain":
                    QMessageBox.warning(self.mw, "Mesh ignored", "The computation has been canceled")
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

                
                print(f"[STL | Area] STL mesh Area Result = {area:.2f} cm^2.")


                dt = time.time() - t0
                print(f"[STL | Area] Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")

            except Exception as ex:
                logger.error("STL Area failed: %s", ex)
                QMessageBox.critical(self.mw, "[STL | Area] Failed", f"{type(ex).__name__}: {ex}")
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
                area, saved_pngs, valid_slices = compute_vtk_area(self.mw, file_path=self.mw.current_path, out_dir=out_dir, min_contour_area=self.mw.cnt_threshold, Slice_direction = self.mw.slice_direction, Physical_dim= self.mw.physical_dim, unit = u, slice_thickness=self.mw.slice_thickness,
                perimeter_method=self.mw.settings.perimeter_method, simplify_contours_for_perimeter=self.mw.settings.simplify_contours_for_perimeter, contour_simplify_epsilon=self.mw.settings.contour_simplify_epsilon,
                fill_cross_section=self.mw.settings.fill_cross_section,
                cavity_correction_enabled=self.mw.settings.cavity_correction_enabled, cavity_area_threshold_mm2=self.mw.settings.cavity_area_threshold_mm2)
            
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
                
                print(f"[VTK | Area] VTK mesh Outer Surface Area Result = {area:.2f} {u}^2.")


                dt = time.time() - t0
                print(f"[VTK | Area] Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")
                      
            except Exception as ex:
                logger.error("VTK Area failed: %s", ex)
                QMessageBox.critical(self.mw, "[VTK | Area] Failed", f"{type(ex).__name__}: {ex}")
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
            self.mw, "Choose a folder", start,
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
            QMessageBox.warning(self.mw, "No images", "The selected folder contains no image files.")
            return

        self.mw.last_dir = dir_path
        first_pm = QPixmap(imgs[0])
        if first_pm.isNull():
            QMessageBox.critical(
                self.mw,
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

        btn = QMessageBox.warning(self.mw,
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

            valid_slices, saved_pngs = process_on_images_batch(dir_path, out_dir, pixel_size=px_size, kernel_size_mm=self.mw.settings.kernel_size_mm,
                cnt_threshold=self.mw.cnt_threshold, unit=u,
                perimeter_method=self.mw.settings.perimeter_method,
                simplify_contours_for_perimeter=self.mw.settings.simplify_contours_for_perimeter,
                contour_simplify_epsilon=self.mw.settings.contour_simplify_epsilon,
                contour_mode=self.mw.contour_mode)

            # Append an Analysis sheet (summary tables + boxplots) to the batch
            # workbook so every GUI batch run is analyzed like the master reports.
            # Best-effort: a missing optional dependency must never fail the batch.
            batch_xlsx = os.path.join(out_dir, "Batch_Allmarks.xlsx")
            if os.path.isfile(batch_xlsx):
                try:
                    from helpers.analyze_master_measurement_reports import analyze_workbook
                    analyze_workbook(batch_xlsx)
                    print(f"[Process Batch] Analysis sheet added to {batch_xlsx}")
                except Exception as ex:
                    logger.warning("Batch analysis sheet skipped: %s", ex)

            self.mw.view.enable_png_navigation(saved_pngs, slice_indices=valid_slices)

            mid = len(saved_pngs) // 2
            self.mw.view.on_slice_slider_changed(mid)
            
        except Exception as ex:
            logger.error("Process Batch failed: %s", ex)
            QMessageBox.critical(self.mw, "Process Batch Failed", f"{type(ex).__name__}: {ex}")
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

                mask, edge_pixels, curvature_values,curvature_values_s  = compute_curvature_profile(path =self.mw.current_path, min_area = self.mw.cnt_threshold, pixel_size = self.mw.pixel_size)
                
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
                QMessageBox.critical(self.mw, "[Curvature] Failed", f"{type(ex).__name__}: {ex}")
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
            excel_files, _ = QFileDialog.getOpenFileNames(self.mw, "Select one or multiple Excel files",
                    start, "Excel Files (*.xlsx *.xls)")
            if not excel_files:
                reply = QMessageBox.question(self.mw, "No files selected",
                            "No Excel files were selected. Would you like to try again?",
                            QMessageBox.Retry | QMessageBox.Cancel)
                if reply == QMessageBox.Cancel:
                    return
                continue
            break

        try:
            df1, max_sulci, max_cell_density = conver_excel(excel_files)
            if df1 is None or df1.empty:
                QMessageBox.warning(self.mw, "Optimization Failed", "No valid rows were found in the selected Excel files.")
                return

            self.mw.last_dir = os.path.dirname(excel_files[0]) or self.mw.last_dir
            opt_dialog = OptimizationOptionsDialog(
                self.mw,
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
                self.mw,
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
                    objective_cols = []
                    for obj in self.mw.optimization_objectives:
                        col = OBJ_TO_COLUMN.get(obj, obj)
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
                            perimeter_outer_envelope=r.get("Closed-envelopePerimeter"),
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
                QMessageBox.warning(self.mw, "Optimization Failed", "Optimization failed or was canceled.")

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
            QMessageBox.critical(self.mw, "Optimization Failed", f"{type(ex).__name__}: {ex}")
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
            First, _ = QFileDialog.getOpenFileName(self.mw, "Select the first image",
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
        Second, _ =  QFileDialog.getOpenFileName(self.mw, "Select the second image",
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
                    self.mw,
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
            print(f"\tBetween {basename1} and {basename2}: {d12} {u1}")
            print(f"\tBetween {basename2} and {basename1}: {d21} {u1}")
            print(f"\tMaximum distance: {hd} {u1}")
            
            self.mw.statusBar().showMessage("Use Ctrl+M and Ctrl+Shift+M to switch between images.")

        except Exception as ex:
            logger.error("Hausdorff failed: %s", ex)
            QMessageBox.critical(self.mw, "Hausdorff distance", f"{type(ex).__name__}: {ex}")
    
    def on_pial_to_stl(self):
        """Pick one .pial, convert to STL in TEMP, show it, and keep source in metrics."""
        pial = None
        if not self.mw.current_kind == "Freesurfer":
            start = self.mw.last_dir if os.path.isdir(self.mw.last_dir) else ""
            pial, _ = QFileDialog.getOpenFileName(self.mw, "Select FreeSurfer Pial Surface",
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
            QMessageBox.critical(self.mw, "Pial → STL", f"{type(ex).__name__}: {ex}")
        
    def on_combined_stl(self):
        """Pick rh & lh .pial, convert + merge in TEMP, show combined STL, record provenance."""
        rh, lh = None, None
        if self.mw.current_kind != "Freesurfer" or  len(self.mw.Freesurfer_record) == 1:
            start = self.mw.last_dir if os.path.isdir(self.mw.last_dir) else ""
            while True:
                files, _ = QFileDialog.getOpenFileNames(self.mw, "Select Both hemisphere (e.g. rh.pial, lh.pial)",
                                                    start, "FreeSurfer Surface (*.pial *.white *.inflated);;All Files (*)")
                                                    
                if not files:
                    return
                
                self.mw.last_dir = os.path.dirname(files[0])
                if len(files) != 2:
                    QMessageBox.warning(self.mw, "Invalid selection", "You must select exactly two files.")
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
                        self.mw,
                        "Invalid selection",
                        "You must select both 'lh' and 'rh' files (e.g., lh.pial and rh.pial)."
                    )
                    continue
                
                if len(exts) != 1:
                    reply = QMessageBox.question(
                    self.mw,
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
            QMessageBox.critical(self.mw, "Pial (rh & lh) → Combined STL", f"{type(ex).__name__}: {ex}")

    def _detect_axis(self, measured: dict) -> str | None:
        """Auto-detect the anatomical axis from stored metrics, CNN, or slice direction."""
        valid = ("sagittal", "coronal", "axial")

        stored = measured.get("SliceKind")
        if stored in valid:
            return stored

        if self.mw.current_kind == "image" and os.path.isfile(self.mw.current_path):
            from helpers.slice_kind_classifier import classify_slice_kind
            img_bgr = cv2.imread(self.mw.current_path)
            if img_bgr is not None:
                label, conf = classify_slice_kind(img_bgr)
                if label in valid and conf >= 0.6:
                    return label

        dir_map = {"X": "sagittal", "Y": "coronal", "Z": "axial"}
        sd = getattr(self.mw, "slice_direction", None)
        if sd in dir_map:
            return dir_map[sd]

        # Cropped sub-slices classify as "not_full_slice" (no axis), so fall back
        # to the axis named in the image's folder path (…/{axis}/image.png).
        if self.mw.current_kind == "image":
            parts = os.path.normpath(
                (getattr(self.mw, "current_path", "") or "").lower()).split(os.sep)
            for a in valid:
                if a in parts:
                    return a

        return None

    def _is_cropped_slice(self, measured: dict) -> bool:
        """True when the current slice is a cropped sub-slice (``not_full_slice``).

        Prefers a stored ``SliceKind``; otherwise classifies the loaded image.
        Non-image contexts (mesh / manual entry) are treated as full slices.
        """
        stored = measured.get("SliceKind")
        if isinstance(stored, str):
            s = stored.strip().lower()
            if s == "not_full_slice":
                return True
            if s in ("axial", "coronal", "sagittal"):
                return False

        if self.mw.current_kind == "image" and os.path.isfile(self.mw.current_path):
            from helpers.slice_kind_classifier import classify_slice_kind
            img_bgr = cv2.imread(self.mw.current_path)
            if img_bgr is not None:
                label, _conf = classify_slice_kind(img_bgr)
                return label == "not_full_slice"
        return False

    def _show_gasp_week_detail(
        self,
        result: GASPResult,
        ref: WeekProfile | None,
        measured: dict,
        metric_labels: dict[str, str],
        result_alt: GASPResult | None = None,
    ) -> None:
        """Open a dialog with a full metric table for a single gestational week."""
        dlg = QDialog(self.mw)
        dlg.setWindowTitle(f"Week {result.week} — GASP Detail")
        dlg.resize(820, 460)
        layout = QVBoxLayout(dlg)

        if result_alt is not None:
            header_text = (
                f"<b>Gestational Week {result.week}</b> &nbsp;|&nbsp; "
                f"Gaussian = {result.gasp:.2%} &nbsp;&nbsp; "
                f"Global Distance = {result_alt.gasp:.2%}")
        else:
            header_text = (
                f"<b>Gestational Week {result.week}</b> &nbsp;|&nbsp; "
                f"GASP = {result.gasp:.2%}")
        header = QLabel(header_text)
        header.setTextFormat(Qt.RichText)
        layout.addWidget(header)

        cols = ["Metric", "Measured", "Ref Mean", "Ref Std",
                "Ref Min", "Ref Max", "z-score", "Similarity", "OOR"]
        table = QTableView()
        model = QStandardItemModel(0, len(cols), dlg)
        model.setHorizontalHeaderLabels(cols)

        comparison_maps = (NORMALIZED_METRIC_MAP, METRIC_MAP)
        measured_key_by_field = {}
        for comparison_map in comparison_maps:
            for meas_key, ref_field in comparison_map.items():
                measured_key_by_field.setdefault(ref_field, meas_key)

        # Show the same derived measured values the scorer used (depth ÷ the
        # slice's own max sulcus depth, per-class count ÷ total count).
        measured_display = _augment_normalized_metrics(measured)

        ordered_fields = []
        for comparison_map in comparison_maps:
            for ref_field in comparison_map.values():
                if ref_field not in ordered_fields:
                    ordered_fields.append(ref_field)
        for ref_field in result.per_metric:
            if ref_field not in ordered_fields:
                ordered_fields.append(ref_field)

        for ref_field in ordered_fields:
            if ref_field not in result.per_metric and (
                result_alt is None or ref_field not in result_alt.per_metric
            ):
                continue
            meas_key = measured_key_by_field.get(ref_field)
            val = measured_display.get(meas_key)
            label = metric_labels.get(ref_field, ref_field)

            stats: MetricStats | None = getattr(ref, ref_field, None) if ref else None
            z = result.z_scores.get(ref_field)
            sim = result.per_metric.get(ref_field)
            oor = result.out_of_range.get(ref_field, False)

            def _fmt(v, decimals=4):
                if v is None:
                    return ""
                try:
                    return f"{float(v):.{decimals}f}"
                except (TypeError, ValueError):
                    return str(v)

            row_items = [
                QStandardItem(label),
                QStandardItem(_fmt(val)),
                QStandardItem(_fmt(stats.mean if stats else None)),
                QStandardItem(_fmt(stats.std if stats else None)),
                QStandardItem(_fmt(stats.min if stats else None)),
                QStandardItem(_fmt(stats.max if stats else None)),
                QStandardItem(_fmt(z)),
                QStandardItem(f"{sim:.2%}" if sim is not None else ""),
                QStandardItem("Yes" if oor else ""),
            ]
            for it in row_items[1:]:
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            if oor:
                for it in row_items:
                    it.setBackground(QColor(255, 235, 235))
            model.appendRow(row_items)

        table.setModel(model)
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.setEditTriggers(QTableView.NoEditTriggers)
        table.setSelectionBehavior(QTableView.SelectRows)
        layout.addWidget(table)

        btn = QPushButton("Close")
        btn.clicked.connect(dlg.close)
        layout.addWidget(btn, alignment=Qt.AlignRight)

        dlg.exec()

    def on_measure_similarity_profile(self):
        """Ask the user how to provide hallmark data, then dispatch.

        The mesh-section option appears only when an STL/VTK mesh has
        already been measured (allmarks computed) AND the 2D slices have
        been generated and are currently being browsed via the slider.
        """
        # If an image is already loaded, skip the picker and score it directly.
        if self.mw.current_kind == "image" and self.mw.current_path:
            self._similarity_profile_from_image()
            return

        mesh_ready = self._mesh_section_workflow_ready()

        mb = QMessageBox(self.mw)
        mb.setWindowTitle("Similarity Profile")
        mb.setIcon(QMessageBox.Question)
        mb.setText("How do you want to provide the brain hallmark data?")
        info_lines = [
            "Import Image — load a brain image, run the measurement "
            "pipeline, then score it.",
            "Enter Data Manually — type hallmark values (Area, "
            "Perimeter, LGI, …) into a form.",
        ]
        if mesh_ready:
            info_lines.append(
                "Use Current Mesh Section — score the slice currently "
                "shown by the slider."
            )
        mb.setInformativeText("\n".join(info_lines))
        btn_import = mb.addButton("Import Image…", QMessageBox.AcceptRole)
        btn_manual = mb.addButton("Enter Data Manually…", QMessageBox.ActionRole)
        btn_mesh = (mb.addButton("Use Current Mesh Section", QMessageBox.ActionRole)
                    if mesh_ready else None)
        mb.addButton(QMessageBox.Cancel)
        mb.exec()
        clicked = mb.clickedButton()
        if clicked is btn_import:
            # No image is loaded: import one first, then score it.
            self.mw.file_mgr.import_image()
            if self.mw.current_kind == "image" and self.mw.current_path:
                self._similarity_profile_from_image()
        elif clicked is btn_manual:
            self._similarity_profile_manual()
        elif btn_mesh is not None and clicked is btn_mesh:
            self._similarity_profile_from_mesh_section()

    def _mesh_section_workflow_ready(self) -> bool:
        """Return True if the per-slice similarity workflow is available."""
        mw = self.mw
        is_mesh = bool(getattr(mw, "is_vtk", False)) or (
            mw.current_kind == "stl")
        if not is_mesh:
            return False
        out_dir = getattr(mw, "current_output_dir", None)
        if not out_dir or not os.path.isdir(out_dir):
            return False
        view = getattr(mw, "view", None)
        if (getattr(view, "slice_nav_mode", None) != "png"
                or not getattr(view, "slice_nav_items", None)):
            return False
        store = getattr(mw, "metrics_store", None)
        rows = store.metrics.get(mw.current_path) if store and mw.current_path else None
        if not rows:
            return False
        return self._mesh_per_slice_xlsx_path() is not None

    def _mesh_per_slice_xlsx_path(self) -> str | None:
        """Locate the per-slice Excel produced by the mesh allmarks pass."""
        mw = self.mw
        out_dir = getattr(mw, "current_output_dir", None)
        if not out_dir or not os.path.isdir(out_dir):
            return None
        candidates = []
        if mw.current_kind == "stl":
            candidates.append(os.path.join(out_dir, "Mesh_Allmarks.xlsx"))
        if getattr(mw, "is_vtk", False):
            sd = getattr(mw, "slice_direction", None) or "Y"
            candidates.append(os.path.join(
                out_dir, f"Mesh_Allmarks_{sd}.xlsx"))
        for p in candidates:
            if os.path.isfile(p):
                return p
        return None

    def _similarity_profile_from_image(self):
        """Image-based similarity profile.

        Requires that at least one measurement (allmarks, area, perimeter, etc.)
        has already been recorded for the current file.  The axis is detected
        automatically from the slice-kind classifier or the slice direction.
        """
        if not self.mw.current_path:
            QMessageBox.information(self.mw, "Similarity Profile",
                "No file is loaded. Open a file and run a measurement first.")
            return

        reply = QMessageBox.information(
            self.mw, "Similarity Profile",
            "To obtain an accurate comparison, make sure to adjust the "
            "image scale properly and set the kernel size to 5 mm and the minimum sulcus depth set to 1 mm.",
            QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Ok)
        if reply != QMessageBox.Ok:
            return

        rows = self.mw.metrics_store.metrics.get(self.mw.current_path)
        if not rows:
            mb = QMessageBox(self.mw)
            mb.setWindowTitle("Similarity Profile")
            mb.setIcon(QMessageBox.Information)
            mb.setText("No morphometric measurements found for the current file.")
            mb.setInformativeText(
                "Run 'All hallmarks' first, or adjust the image (scale, kernel "
                "size, ROI) before measuring.")
            btn_allmarks = mb.addButton("Compute All Hallmarks", QMessageBox.AcceptRole)
            btn_adjust = mb.addButton("Adjust Image…", QMessageBox.ActionRole)
            mb.addButton(QMessageBox.Cancel)
            mb.exec()
            clicked = mb.clickedButton()
            if clicked is btn_allmarks:
                self.on_measure_allmarks()
                self._similarity_profile_from_image()
            elif clicked is btn_adjust:
                self.mw._enter_adjustment_mode()
                self.mw.statusBar().showMessage(
                    "Adjust the image (scale, kernel, ROI), then press "
                    "Shift+Alt+E to continue.")
                print("[Similarity Profile] Adjust the image now and press "
                      "Shift+Alt+E to continue.")
                self.mw.wait_for_resume()
                self.mw.statusBar().clearMessage()
                self.mw._exit_adjustment_mode()
                self.on_measure_allmarks()
                self._similarity_profile_from_image()
            return

        if isinstance(rows, dict):
            rows = [rows]
        measured = rows[-1]

        has_data = any(measured.get(k) is not None for k in
                       ("Area", "Perimeter", "LGI", "Compactness", "PrimarySulciCount","SecondarySulciCount","TertiarySulciCount","UnclassifiedSulciCount", "PrimaryMeanDepth","SecondaryMeanDepth","TertiaryMeanDepth","UnclassifiedMeanDepth"))
        if not has_data:
            QMessageBox.information(self.mw, "Similarity Profile",
                "The latest measurement row has no comparable metrics.\n"
                "Run 'All hallmarks' first to populate area, perimeter, LGI, etc.")
            return

        axis = self._detect_axis(measured)
        if axis is None:
            QMessageBox.warning(self.mw, "Similarity Profile",
                "Could not determine the anatomical axis automatically.\n"
                "Please run 'All hallmarks' first so the slice classifier can identify the axis.")
            return

        self._run_similarity_profile(measured, axis)

    def _similarity_profile_manual(self):
        """Open the manual hallmark-entry dialog and score against the registry."""
        from widgets.manual_gasp_dialog import ManualGASPDialog

        # Ask which comparison to use BEFORE entering data, so the form can be
        # tailored (normalized hides the unit-based Area/Perimeter fields).
        cmp_box = QMessageBox(self.mw)
        cmp_box.setWindowTitle("Similarity Profile")
        cmp_box.setIcon(QMessageBox.Question)
        cmp_box.setText("Which comparison do you want to use?")
        cmp_box.setInformativeText(
            "Normalized — unit-free metrics only (LGI, compactness, sulcal "
            "counts, normalized sulcal depth); use when scale/units differ.\n"
            "Full — also includes absolute metrics (area, perimeter, depth).")
        btn_norm = cmp_box.addButton("Normalized", QMessageBox.AcceptRole)
        btn_full = cmp_box.addButton("Full", QMessageBox.ActionRole)
        cmp_box.addButton(QMessageBox.Cancel)
        cmp_box.exec()
        clicked = cmp_box.clickedButton()
        if clicked is btn_norm:
            force_normalized = True
        elif clicked is btn_full:
            force_normalized = False
        else:
            return

        settings = getattr(self.mw, "settings", None)
        default_kernel = getattr(settings, "kernel_size_mm", 5.0)
        default_unit = getattr(settings, "units_length", None) or "mm"
        default_pixel = None
        if settings is not None and self.mw.current_path:
            default_pixel = settings.image_scales.get(
                self.mw.current_path, getattr(settings, "pixel_size", None))
        default_name = getattr(self.mw, "custom_label", None) or ""

        dlg = ManualGASPDialog(
            self.mw,
            default_axis="coronal",
            default_unit=default_unit,
            default_kernel_size=default_kernel,
            default_pixel_size=default_pixel,
            default_project_name=default_name,
            normalized=force_normalized,
        )
        if dlg.exec() != QDialog.Accepted:
            return
        vals = dlg.values()
        measured = vals["measured"]
        if not measured:
            QMessageBox.information(self.mw, "Similarity Profile",
                "No hallmark values were entered.")
            return
        axis = vals["axis"]

        overrides = {
            "project_name": vals["project_name"] or None,
            "kernel_size_mm": vals["kernel_size"],
            "pixel_size": vals["pixel_size"],
            "length_unit": vals["length_unit"],
            "pixel_size_units": vals["length_unit"],
            "source_path": None,
            "source_label": "(manually entered)",
        }
        self._run_similarity_profile(measured, axis, manual_overrides=overrides,
                                     force_normalized=force_normalized)

    def _similarity_profile_from_mesh_section(self):
        """Score the slice currently selected on the slider for an STL/VTK mesh.

        Requires that hallmarks have been computed (per-slice metrics Excel
        present in *current_output_dir*) and the slider is in PNG-browse mode.
        Disables the slider after a successful run so the selection cannot
        drift away from the results.
        """
        mw = self.mw
        if not self._mesh_section_workflow_ready():
            QMessageBox.information(
                mw, "Similarity Profile",
                "This option needs a 3D mesh with hallmarks computed and 2D "
                "slices generated. Run 'All hallmarks' on the mesh first.")
            return

        # The png slider value is the PNG-list position; the real slice index
        # (the Excel "Section") comes from the index map, and the PNG is that same
        # position. This stays correct when slices were filtered (gaps / offset).
        view = getattr(mw, "view", None)
        items = list(getattr(view, "slice_nav_items", []) or [])
        idx_map = list(getattr(view, "slice_nav_index_map", []) or [])
        list_idx = int(mw.slice_slider.value())
        if 0 <= list_idx < len(idx_map) and idx_map[list_idx] is not None:
            slice_idx = int(idx_map[list_idx])
        else:
            slice_idx = list_idx
        png_path = items[list_idx] if 0 <= list_idx < len(items) else None

        measured = self._load_mesh_per_slice_measured(slice_idx)
        if not measured:
            QMessageBox.warning(
                mw, "Similarity Profile",
                f"Could not find per-slice metrics for slice {slice_idx}.")
            return

        axis_map = {"X": "sagittal", "Y": "coronal", "Z": "axial"}
        direction = (getattr(mw, "slice_direction", None) or "Y").upper()
        axis = axis_map.get(direction, "coronal")

        base_name = (
            os.path.splitext(os.path.basename(mw.current_path))[0]
            if mw.current_path else "Mesh"
        )
        project_name = (
            getattr(mw, "custom_label", None) or base_name
        ) + f"_slice_{slice_idx}"

        overrides = {
            "project_name": project_name,
            "source_path": png_path,
            "source_label": (png_path or f"{base_name} — slice {slice_idx}"),
        }
        self._run_similarity_profile(measured, axis, manual_overrides=overrides)

        try:
            mw.nav_tb.hide()
            mw.slice_slider.setEnabled(False)
            print(f"[GASP] Slider locked to slice {slice_idx} "
                  "(results match the selected section).")
        except Exception as ex:
            logger.debug("Could not disable slice slider: %s", ex)

    def _load_mesh_per_slice_measured(self, slice_idx: int) -> dict | None:
        """Read the per-slice hallmark values for *slice_idx* from the
        mesh-allmarks Excel sitting in *current_output_dir*.

        Returns a measured dict keyed by ``Area``/``Perimeter``/``LGI``/…
        ready to be fed into ``_run_similarity_profile``.
        """
        xlsx_path = self._mesh_per_slice_xlsx_path()
        if xlsx_path is None:
            return None

        # The allmarks Excel is written in the shared spec layout
        # (helpers.results_excel_format), so it must be parsed with
        # read_results_sheet — a raw pd.read_excel sees the title/merged cells,
        # not a flat table, and the per-slice column is "Section", not "Slice".
        try:
            from helpers.results_excel_format import read_results_sheet
            data = read_results_sheet(xlsx_path)
        except Exception as ex:
            logger.error("Could not read %s: %s", xlsx_path, ex)
            return None

        def _as_int(v):
            try:
                return int(float(v))
            except (TypeError, ValueError):
                return None

        target = None
        for r in data.get("rows", []):
            if _as_int(r.get("Section")) == int(slice_idx):
                target = r
                break
        if target is None:
            return None

        def _f(col: str):
            v = target.get(col)
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return None
            if math.isnan(fv):
                return None
            return fv

        def _i(col: str):
            v = _f(col)
            return int(v) if v is not None else None

        from helpers.helpers import compactness_2D
        # The spec sheet already carries resolved per-slice LGI / Compactness /
        # mean-depth columns (keys match the GASP measured-dict names), so they
        # map straight across. Compactness falls back to a recompute if absent.
        area = _f("Area")
        perim = _f("Perimeter")
        lgi = _f("LGI")
        compact = _f("Compactness")
        if compact is None and area is not None and perim:
            compact = compactness_2D(area, perim)

        measured: dict = {}
        if area is not None:
            measured["Area"] = area
        if perim is not None:
            measured["Perimeter"] = perim
        if lgi is not None:
            measured["LGI"] = lgi
        if compact is not None:
            measured["Compactness"] = compact

        for col in ("PrimarySulciCount", "SecondarySulciCount",
                    "TertiarySulciCount", "UnclassifiedSulciCount"):
            v = _i(col)
            if v is not None:
                measured[col] = v

        for col in ("PrimaryMeanDepth", "SecondaryMeanDepth",
                    "TertiaryMeanDepth", "UnclassifiedMeanDepth"):
            v = _f(col)
            if v is not None:
                measured[col] = v

        return measured

    def _run_similarity_profile(self, measured, axis, *, manual_overrides=None,
                                force_normalized=None):
        """Compute the GASP summary for *measured* on *axis*, render the chart,
        and write the per-run results folder.

        *manual_overrides* supplies project name and analysis-parameter values
        when there is no loaded image to read them from.

        *force_normalized* overrides the normalized-vs-full comparison choice
        (used by the manual-entry flow). ``None`` keeps the default behaviour
        (cropped always normalized; full follows the ``gasp_use_normalized_full``
        setting). Cropped data is always scored normalized regardless.
        """
        # Cropped sub-slices (not_full_slice) are compared against the cropped
        # reference; full MRI slices against the full-slice reference.
        examples_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Examples")
        full_csv = os.path.join(examples_dir, "gestational_week_reference.csv")
        cropped_csv = os.path.join(examples_dir, "gestational_week_cropped_reference.csv")
        is_cropped = self._is_cropped_slice(measured)
        csv_path = cropped_csv if is_cropped else full_csv
        if is_cropped and not os.path.isfile(csv_path):
            logger.warning(
                "Cropped reference not found (%s); using full-slice reference.", csv_path)
            csv_path = full_csv
        if not os.path.isfile(csv_path):
            QMessageBox.critical(self.mw, "Similarity Profile",
                f"Reference CSV not found:\n{csv_path}")
            return
        print(f"[Similarity Profile] slice_kind={'cropped' if is_cropped else 'full'}; "
              f"reference={os.path.basename(csv_path)}")

        from managers.visualization_settings import get_active as _get_viz
        viz = _get_viz()
        gasp_method = getattr(viz, "gasp_method", "gaussian")
        range_penalty = float(getattr(viz, "gasp_range_penalty", 0.0))
        oor_beta = float(getattr(viz, "gasp_oor_beta", 1.0))
        apply_penalty = bool(getattr(viz, "gasp_apply_penalty", True))
        weighted_global = bool(getattr(viz, "gasp_weighted_global", True))
        if force_normalized is not None:
            # Manual-entry choice; cropped data is always normalized regardless.
            use_normalized = bool(is_cropped) or bool(force_normalized)
        else:
            use_normalized = bool(is_cropped or getattr(viz, "gasp_use_normalized_full", False))

        ref_to_attr = {
            "area": "gasp_w_area",
            "perimeter": "gasp_w_perimeter",
            "lgi": "gasp_w_lgi",
            "compactness": "gasp_w_compactness",
            "primary_count": "gasp_w_primary_count",
            "secondary_count": "gasp_w_secondary_count",
            "tertiary_count": "gasp_w_tertiary_count",
            "primary_sulcus_values": "gasp_w_primary_sulcus_values",
            "secondary_sulcus_values": "gasp_w_secondary_sulcus_values",
            "tertiary_sulcus_values": "gasp_w_tertiary_sulcus_values",
            "primary_sulcus_values_normalized": "gasp_w_primary_sulcus_values",
            "secondary_sulcus_values_normalized": "gasp_w_secondary_sulcus_values",
            "tertiary_sulcus_values_normalized": "gasp_w_tertiary_sulcus_values",
            "primary_count_normalized": "gasp_w_primary_count",
            "secondary_count_normalized": "gasp_w_secondary_count",
            "tertiary_count_normalized": "gasp_w_tertiary_count",
            # Total sulcus count / depth (raw and normalized share one weight each).
            "sulci_count": "gasp_w_sulci_count",
            "sulci_depth": "gasp_w_sulci_depth",
            "sulci_depth_normalized": "gasp_w_sulci_depth",
        }
        weights_user = {k: float(getattr(viz, a, 1.0)) for k, a in ref_to_attr.items()}
        weights_user.setdefault("sulci_count_normalized", weights_user["sulci_count"])
        weights_user.setdefault("unclassified_count", 1.0)
        weights_user.setdefault("unclassified_count_normalized", 1.0)
        weights_user.setdefault("unclassified_sulcus_values", 1.0)
        weights_user.setdefault("unclassified_sulcus_values_normalized", 1.0)

        def _run(method: str) -> GASPSummary:
            if method.startswith("mahal") and not weighted_global:
                w = {k: 1.0 for k in weights_user}
            else:
                w = weights_user
            return compute_similarity_scores(
                measured, registry, axis,
                method=method, weights=w,
                apply_range_penalty=apply_penalty,
                beta=oor_beta,
                use_normalized=use_normalized,
            )

        try:
            registry = GestationalWeekProfile(csv_path)
            # Set the module-level Gaussian λ at call time so it reflects the
            # user's choice (RANGE_PENALTY is read inside compute_similarity_scores).
            import helpers.gestational_week_profile as _gwp
            _gwp.RANGE_PENALTY = range_penalty

            if gasp_method == "both":
                summary = _run("gaussian")
                summary_alt = _run("mahalanobis")
            else:
                summary = _run(gasp_method)
                summary_alt = None
        except Exception as ex:
            logger.error("Similarity profile failed: %s", ex)
            QMessageBox.critical(self.mw, "Similarity Profile",
                f"{type(ex).__name__}: {ex}")
            return

        if not summary.results:
            QMessageBox.information(self.mw, "Similarity Profile",
                "No reference weeks have data for the detected axis.")
            return

        sorted_by_week = sorted(summary.results, key=lambda r: r.week)
        sorted_alt_by_week = (
            sorted(summary_alt.results, key=lambda r: r.week)
            if summary_alt is not None else None
        )

        metric_labels = {
            "area": "Area", "perimeter": "Perimeter", "lgi": "LGI",
            "compactness": "Compactness",
            "primary_count": "Primary Count", "secondary_count": "Secondary Count",
            "tertiary_count": "Tertiary Count",
            "primary_sulcus_values": "Primary Depth",
            "secondary_sulcus_values": "Secondary Depth",
            "tertiary_sulcus_values": "Tertiary Depth",
            "unclassified_count": "Unclassified Count",
            "unclassified_sulcus_values": "Unclassified Depth",
            "sulci_count": "Sulci Count",
            "sulci_depth": "Sulci Depth",
            "sulci_depth_normalized": "Normalized Sulci Depth",
            "sulci_count_normalized": "Normalized Sulci Count",
        }

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        n_weeks = len(sorted_by_week)
        fig_h = max(4, n_weeks * 0.45)
        fig = Figure(figsize=(7, fig_h))
        ax = fig.add_subplot(111)

        week_labels = [str(r.week) for r in sorted_by_week]

        if sorted_alt_by_week is not None:
            # Grouped bars: Gaussian (top) + Mahalanobis (bottom) per week
            import numpy as _np
            y_idx = _np.arange(n_weeks)
            bar_h = 0.38
            scores_g = [r.gasp for r in sorted_by_week]
            scores_m = [r.gasp for r in sorted_alt_by_week]
            bars = ax.barh(y_idx - bar_h / 2, scores_g, height=bar_h,
                           color="#3498db", edgecolor="white",
                           linewidth=0.5, label="Gaussian")
            bars_alt = ax.barh(y_idx + bar_h / 2, scores_m, height=bar_h,
                               color="#9b59b6", edgecolor="white",
                               linewidth=0.5, label="Global Distance")
            ax.set_yticks(y_idx)
            ax.set_yticklabels(week_labels)
            ax.legend(loc="lower right", fontsize=8)
            method_label = "Both"
        else:
            scores = [r.gasp for r in sorted_by_week]
            colors = ["#2ecc71" if r.week == summary.best_week else "#3498db"
                      for r in sorted_by_week]
            bars = ax.barh(week_labels, scores, color=colors,
                           edgecolor="white", linewidth=0.5)
            bars_alt = None
            method_label = ("Global Distance" if gasp_method.startswith("mahal")
                            else "Gaussian Similarity")

        ax.set_ylabel("Gestational Week", fontsize=11)
        ax.set_xlabel("GASP Score", fontsize=11)

        gauss_color = "#1f6fa5"
        mahal_color = "#7d3c98"

        def _line(label: str, s: GASPSummary) -> str:
            return (f"  ■  {label}    │  "
                    f"best week {s.best_week}    GASP {s.max_gasp:.1%}    "
                    f"est. GA {s.estimated_ga:.1f} w    "
                    f"confidence: {s.confidence}")

        if summary_alt is not None:
            # Both methods: summary is Gaussian, summary_alt is Global Distance
            primary_line = _line("GAUSSIAN SIMILARITY", summary)
            primary_color = gauss_color
            secondary_line = _line("GLOBAL DISTANCE   ", summary_alt)
            secondary_color = mahal_color
        elif gasp_method.startswith("mahal"):
            primary_line = _line("GLOBAL DISTANCE", summary)
            primary_color = mahal_color
            secondary_line = None
        else:
            primary_line = _line("GAUSSIAN SIMILARITY", summary)
            primary_color = gauss_color
            secondary_line = None

        # Clear ax title — using fig.suptitle + colored fig.text instead so each
        # method's results are visually distinguishable.
        ax.set_title("")
        fig.suptitle(f"Gestational Age Similarity Profile  —  axis: {axis}",
                     fontsize=12, fontweight="bold", y=0.995)
        fig.text(0.5, 0.955, primary_line, ha="center", va="top", fontsize=9,
                 color=primary_color, fontweight="bold", family="monospace")
        if secondary_line is not None:
            fig.text(0.5, 0.920, secondary_line, ha="center", va="top", fontsize=9,
                     color=secondary_color, fontweight="bold", family="monospace")

        ax.set_xlim(0, 1.12)
        ax.axvline(x=summary.max_gasp, color="#e74c3c",
                    linestyle="--", linewidth=0.8, alpha=0.6)
        ax.invert_yaxis()

        all_bars = list(bars) + (list(bars_alt) if bars_alt is not None else [])
        for bar in all_bars:
            ax.text(bar.get_width() + 0.015,
                    bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():.0%}", ha="left", va="center", fontsize=7)

        top_rect = 0.88 if summary_alt is not None else 0.92
        fig.tight_layout(rect=[0, 0, 1, top_rect])

        tooltip = ax.annotate(
            "", xy=(0, 0), xytext=(15, 15),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", fc="#ffffcc",
                      ec="#999999", alpha=0.95),
            fontsize=9, visible=False)

        def _hit_bar(event):
            """Return (week_index, which_method) if the event falls on a bar."""
            for i, bar in enumerate(bars):
                if bar.contains(event)[0]:
                    return i, "primary"
            if bars_alt is not None:
                for i, bar in enumerate(bars_alt):
                    if bar.contains(event)[0]:
                        return i, "alt"
            return None, None

        def on_hover(event):
            if event.inaxes != ax:
                if tooltip.get_visible():
                    tooltip.set_visible(False)
                    canvas_qt.draw_idle()
                return
            i, which = _hit_bar(event)
            if i is None:
                if tooltip.get_visible():
                    tooltip.set_visible(False)
                    canvas_qt.draw_idle()
                return
            r = (sorted_alt_by_week[i] if which == "alt"
                 else sorted_by_week[i])
            method_tag = ("Global Distance" if which == "alt"
                          or (sorted_alt_by_week is None and gasp_method.startswith("mahal"))
                          else "Gaussian")
            lines = [f"Week {r.week}  ({method_tag})",
                     f"GASP: {r.gasp:.2%}", ""]
            for k, v in r.per_metric.items():
                z = r.z_scores.get(k, 0.0)
                oor = r.out_of_range.get(k, False)
                label = metric_labels.get(k, k)
                flag = " ⚠ OOR" if oor else ""
                lines.append(f"  {label}: {v:.2%} (z={z:.2f}){flag}")
            tooltip.set_text("\n".join(lines))
            tooltip.xy = (event.xdata, event.ydata)
            tooltip.set_visible(True)
            canvas_qt.draw_idle()

        def on_right_click(event):
            if event.inaxes != ax or event.button != 3:
                return
            i, _which = _hit_bar(event)
            if i is None:
                return
            r = sorted_by_week[i]
            r_alt = sorted_alt_by_week[i] if sorted_alt_by_week is not None else None
            ref = registry.get(r.week, axis)
            self._show_gasp_week_detail(
                r, ref, measured, metric_labels, r_alt)

        canvas_qt = FigureCanvasQTAgg(fig)
        canvas_qt.mpl_connect("motion_notify_event", on_hover)
        canvas_qt.mpl_connect("button_press_event", on_right_click)

        old = getattr(self.mw, "_similarity_canvas", None)
        if old is not None:
            old.setVisible(False)
            self.mw.display_box.removeWidget(old)
            old.deleteLater()

        self.mw._similarity_canvas = canvas_qt
        self.mw.display_box.addWidget(canvas_qt)
        self.mw.image_label.setVisible(False)
        self.mw.vtk_view.setVisible(False)
        canvas_qt.setVisible(True)
        self.mw._active_view = "chart"

        uid = uuid.uuid4().hex[:8]
        out_dir = os.path.join(self.mw.temp_dir, f"similarity_{uid}")
        os.makedirs(out_dir, exist_ok=True)
        self.mw.current_output_dir = out_dir
        fig.savefig(os.path.join(out_dir, "similarity_profile.png"),
                    dpi=150, bbox_inches="tight")

        if summary_alt is not None:
            method_banner = "Both methods (Gaussian Similarity + Global Distance)"
            primary_banner = "Gaussian Similarity"
        elif gasp_method.startswith("mahal"):
            method_banner = "Global Distance (Diagonal Mahalanobis)"
            primary_banner = "Global Distance"
        else:
            method_banner = "Metric-level Gaussian Similarity"
            primary_banner = "Gaussian Similarity"

        def _print_section(label: str, s: GASPSummary, sorted_results) -> None:
            print(f"[GASP] ── {label} ──")
            print(f"[GASP]   Best-matching week: {s.best_week}  "
                  f"(GASP = {s.max_gasp:.2%})")
            print(f"[GASP]   Estimated GA: {s.estimated_ga:.1f} weeks  "
                  f"|  confidence: {s.confidence}")
            print(f"[GASP]   Per-week scores:")
            for r in sorted_results:
                print(f"           Week {r.week}: {r.gasp:.2%}")

        print(f"[GASP] Method: {method_banner}")
        print(f"[GASP] Axis: {axis}")
        _print_section(primary_banner, summary, sorted_by_week)

        record_vals = dict(
            GASPAxis=axis,
            GASPMethod=method_banner,
            GASPBestWeek=summary.best_week,
            GASPScore=summary.max_gasp,
            EstimatedGA=summary.estimated_ga,
            GASPConfidence=summary.confidence,
        )
        if summary_alt is not None:
            _print_section("Global Distance", summary_alt, sorted_alt_by_week)
            record_vals.update(
                GASPBestWeek_GlobalDistance=summary_alt.best_week,
                GASPScore_GlobalDistance=summary_alt.max_gasp,
                EstimatedGA_GlobalDistance=summary_alt.estimated_ga,
            )

        # Final one-line summary reflecting the chosen method(s).
        if summary_alt is not None:
            print(f"[GASP] FINAL SUMMARY  —  Gaussian: week {summary.best_week} "
                  f"({summary.max_gasp:.1%}, GA {summary.estimated_ga:.1f}w)  "
                  f"|  Global Distance: week {summary_alt.best_week} "
                  f"({summary_alt.max_gasp:.1%}, GA {summary_alt.estimated_ga:.1f}w)")
        else:
            print(f"[GASP] FINAL SUMMARY ({primary_banner})  —  "
                  f"best week {summary.best_week}, GASP {summary.max_gasp:.1%}, "
                  f"estimated GA {summary.estimated_ga:.1f}w, "
                  f"confidence: {summary.confidence}")

        self.mw.metrics_store.record_metric_for(
            self.mw.current_path, **record_vals)

        try:
            from helpers.gasp_export import export_gasp_results
            settings = getattr(self.mw, "settings", None)
            overrides = manual_overrides or {}

            pixel_size = overrides.get("pixel_size")
            if pixel_size is None and settings is not None:
                pixel_size = settings.image_scales.get(
                    self.mw.current_path,
                    getattr(settings, "pixel_size", None),
                )
            pixel_size_units = overrides.get(
                "pixel_size_units",
                getattr(settings, "units_length", None))
            length_unit = overrides.get(
                "length_unit",
                getattr(settings, "units_length", None))
            kernel_size_mm = overrides.get(
                "kernel_size_mm",
                getattr(settings, "kernel_size_mm", None))
            kernel_size_px = None
            if settings is not None and pixel_size is not None:
                kernel_size_px = settings.kernel_size_px(pixel_size)

            override_name = overrides.get("project_name")
            if override_name:
                project_name = f"{override_name}_similarity_{uid}"
            else:
                base_name = (
                    os.path.splitext(os.path.basename(self.mw.current_path))[0]
                    if self.mw.current_path else "GASP"
                )
                project_name = (
                    getattr(self.mw, "custom_label", None) or base_name
                ) + f"_similarity_{uid}"

            source_path = overrides.get("source_path", self.mw.current_path)
            export_source = (
                overrides.get("source_label")
                if overrides.get("source_label") and not source_path
                else source_path
            )

            params = dict(
                method_banner=method_banner,
                range_penalty=range_penalty,
                oor_beta=oor_beta,
                apply_penalty=apply_penalty,
                weighted_global=weighted_global,
                normalized_comparison=use_normalized,
                reference_csv=os.path.basename(csv_path),
                kernel_size_mm=kernel_size_mm,
                kernel_size_px=kernel_size_px,
                pixel_size=pixel_size,
                pixel_size_units=pixel_size_units,
                length_unit=length_unit,
                filtered_threshold=getattr(settings, "cnt_threshold", None),
                contour_mode=getattr(settings, "contour_mode", None),
            )
            artifacts = export_gasp_results(
                out_dir,
                project_name=project_name,
                source_path=export_source,
                measured=measured,
                summary=summary,
                summary_alt=summary_alt,
                registry=registry,
                axis=axis,
                params=params,
                weights=weights_user,
            )
            print(f"[GASP] Wrote results folder: {out_dir}")
            for k, v in artifacts.items():
                print(f"[GASP]   {k}: {v}")
        except Exception as ex:
            logger.error("Failed to export GASP results: %s", ex, exc_info=True)
