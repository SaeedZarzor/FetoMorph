from pathlib import Path
from PySide6.QtGui import QIcon

def set_icons(ui, assets: Path):
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
    }

    for attr, rel in icons.items():
        action = getattr(ui, attr, None)
        if action:
            action.setIcon(QIcon(str(assets / rel)))
