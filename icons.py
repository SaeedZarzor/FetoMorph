"""Icon loader for the FetoMorph ribbon / toolbar actions.

Maps each ``QAction`` attribute name on the main window to the relative
path of its icon file under the *assets* directory.
"""

from deps import *


def set_icons(ui, assets: Path) -> None:
    """Assign icon files to every toolbar QAction on *ui*.

    Args:
        ui: The main window instance whose QAction attributes will be updated.
        assets: Root directory that contains the ``icons/`` subfolder.
    """
    icons = {
        "act_nav_import":       "icons/import.png",
        "act_nav_export":       "icons/export.png",
        "act_Reset":            "icons/rest.png",
        "act_close":            "icons/close.png",
        "act_quit":             "icons/quit.png",
        "act_imp_img":          "icons/image.png",
        "act_imp_vtk":          "icons/vtk.png",
        "act_imp_stl":          "icons/stl.png",
        "act_imp_nii":          "icons/nifti.png",
        "act_save":             "icons/screenshot.png",
        "act_save_data":        "icons/save.png",
        "act_export_metrics":   "icons/export_to_excel.png",
        "act_set_image_scale":  "icons/scale.png",
        "act_set_scale":        "icons/scale_bar.png",
        "act_kernel_size":      "icons/kernel.png",
        "act_annotate_square":  "icons/crop_1.png",
        "act_cnt_threshold":    "icons/threshold.png",
        "act_meas_allmarks":    "icons/hallmarks.png",
        "act_meas_area":        "icons/area.png",
        "act_meas_volumes":     "icons/volume.png",
        "act_meas_perimeter":   "icons/Perimeter.png",
        "act_meas_lgi":         "icons/LGI.png",
        "act_meas_sulci":       "icons/depth.png",
        "act_choose_regions":   "icons/labels.png",
        "act_show_results":     "icons/view_data.png",
        "act_set_custom_label": "icons/label.png",
        "act_slice_thickness":  "icons/thickness.png",
        "act_meas_curvature":   "icons/angle.png",
        "act_hausdorf":         "icons/hausdorff.png",
        "act_set_physical_dim": "icons/dim_set.png",
        "act_img_batch":        "icons/batch.png",
        "act_optimization":     "icons/optimization.png",
        "act_niftiextractor":   "icons/Extract.png",
        "act_pial_merge":       "icons/combine.png",
        "act_nitfi2png":        "icons/mask.png",
        "act_pial_to_stl":      "icons/convert.png",
        "act_view_surfacses":   "icons/Freesurfer.png",
        "act_view_morph_map":   "icons/morph_map.png",
        "act_meas_stright":     "icons/length.png",
        "act_meas_compactness": "icons/compactness.png",
        "act_meas_curve": "icons/curve.png",

    }

    for attr, rel in icons.items():
        action = getattr(ui, attr, None)
        if action:
            action.setIcon(QIcon(str(assets / rel)))
