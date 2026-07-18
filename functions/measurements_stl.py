"""STL mesh measurement functions for FetoMorph.

Pipeline: STL mesh → PyVista off-screen slice renders → OpenCV contour
analysis.  A red reference cube (10 % of Y extent) is rendered alongside
each cross-section and detected in the screenshot to calibrate mm/px.

Red-cube contours are excluded from brain measurements via
``contours_exclude`` so they don't pollute area / perimeter values.

Unit conversions:
    * ``/1000``: mm³ → cm³  (volume)
    * ``/100``:  mm² → cm²  (surface area)
    * ``/10``:   mm  → cm   (depth)
"""

from __future__ import annotations

from deps import *
from helpers.helpers import (
    compute_kernel_convex,
    contours_exclude,
    fill_section_polydata,
    split_inner_and_internal_contours,
    threshold_binary,
    calc_scale_with_metadata,
    get_red_rect_offset,
    make_scale_cube,
    prepare_orthographic_slice_render,
    slice_at,
    validate_scale_cube_sanity,
    compactness_2D,
    compactness_3D,
    mask_perimeter_mm,
    area_based_gi_3d,
    lateral_area_simpson,
    volume_simpson,
    total_surface_area_simpson,
    image_annotation_style,
    SULCUS_CLASS_COLORS,
    SULCUS_CLASSES,
    classify_sulcus_depth,
    sulcus_depth_min,
    empty_depth_sets,
    flatten_depth_sets,
    format_sulcus_class_summary,
    sulcus_export_columns,
    sulcus_export_cells,
    pad_row,
    drop_empty_columns,
)
from helpers.slice_kind_classifier import classify_slice_kind
from helpers.check_mesh import check_brain
from helpers.cavities import (
    SliceRecord, CavityCorrection, cavity_correction_tracking,
    make_slice_cavity, contour_to_mm,
)
from managers.visualization_settings import get_active as _get_viz
from constants import (
    BINARY_THRESHOLD_DEFAULT,
    DEFAULT_KERNEL_SIZE_MM,
    DEFECT_FIXED_POINT,
    SULCUS_TERTIARY_MIN_FRACTION,
    SULCUS_PRIMARY_MAX_FRACTION,
    DEFAULT_CAVITY_CORRECTION_ENABLED,
    DEFAULT_CAVITY_AREA_THRESHOLD_MM2,
    DEFAULT_FILL_CROSS_SECTION,
    DEFAULT_PERIMETER_METHOD,
    DEFAULT_SIMPLIFY_CONTOURS_FOR_PERIMETER,
    DEFAULT_CONTOUR_SIMPLIFY_EPSILON,
)

logger = logging.getLogger("fetomorph.stl")


def _to_kernel_px(kernel_size_mm: float, pixel_size_mm: float) -> int:
    px = max(3, int(round(float(kernel_size_mm) / max(float(pixel_size_mm), 1e-9))))
    return px


def _stl_slice_setup(mesh, brain_dim, Slice_direction: str, pixel_spacing: float = 0.1):
    """Compute direction-dependent slicing parameters for STL rendering.

    Returns a dict with keys: axis_index, pre_axis, next_axis, low, high,
    image_width, image_height, view_fn_name, cube_len, max_dim.
    """
    axis_index = {"X": 0, "Y": 1, "Z": 2}[Slice_direction]
    pre_axis = (axis_index - 1) % 3
    next_axis = (axis_index + 1) % 3

    bounds_pairs = list(zip(mesh.bounds[::2], mesh.bounds[1::2]))
    low, high = bounds_pairs[axis_index]

    image_width = int(np.clip(np.ceil(brain_dim[pre_axis] / pixel_spacing), 64, 4096))
    image_height = int(np.clip(np.ceil(brain_dim[next_axis] / pixel_spacing), 64, 4096))
    view_fn_name = {"X": "view_yz", "Y": "view_xz", "Z": "view_xy"}[Slice_direction]
    cube_len = max(1e-6, brain_dim[axis_index] / 10.0)
    max_dim = max(brain_dim[pre_axis], brain_dim[next_axis])

    return {
        "axis_index": axis_index,
        "pre_axis": pre_axis,
        "next_axis": next_axis,
        "low": low,
        "high": high,
        "image_width": image_width,
        "image_height": image_height,
        "view_fn_name": view_fn_name,
        "cube_len": cube_len,
        "max_dim": max_dim,
    }


def _require_scale_calibration(
    p,
    Slice_direction: str,
    cube_len: float,
    mesh_center,
    sample_slice: float,
    offset: float,
    view_fn_name: str,
    *,
    physical_cube_len: float | None = None,
) -> dict:
    scale_cube = make_scale_cube(Slice_direction, cube_len, mesh_center, sample_slice, offset)
    ok, metadata = validate_scale_cube_sanity(
        p,
        scale_cube,
        float(physical_cube_len if physical_cube_len is not None else cube_len),
        getattr(p, view_fn_name),
        background="white",
    )
    if not ok:
        raise ValueError(
            "Scale cube calibration failed before processing slices: "
            f"{metadata.get('calibration_status')}"
        )
    return metadata


# ----------------- main API -----------------
def compute_stl_allmarks(
    parent,
    file_path: str,
    out_dir: str,
    min_contour_area: float = 20.0,
    kernel_size_mm: float = DEFAULT_KERNEL_SIZE_MM,
    slice_thickness: float = 0.5,
    Slice_direction: str = "Y",
    cavity_correction_enabled: bool = DEFAULT_CAVITY_CORRECTION_ENABLED,
    cavity_area_threshold_mm2: float = DEFAULT_CAVITY_AREA_THRESHOLD_MM2,
    perimeter_method: str = DEFAULT_PERIMETER_METHOD,
    simplify_contours_for_perimeter: bool = DEFAULT_SIMPLIFY_CONTOURS_FOR_PERIMETER,
    contour_simplify_epsilon: float = DEFAULT_CONTOUR_SIMPLIFY_EPSILON,
    fill_cross_section: bool = DEFAULT_FILL_CROSS_SECTION):
    """Compute all hallmarks (volume, area, GI, sulci depths) from an STL mesh.

    Slices the mesh along Y, renders each cross-section with a red scale
    cube, then runs contour analysis on the screenshots.

    Args:
        parent: Qt parent widget for message boxes.
        file_path: Path to the ``.stl`` file.
        out_dir: Output directory for slice PNGs and Excel.
        min_contour_area: Minimum contour area (pixels) to keep.
        kernel_size_mm: Morph-close kernel diameter for outer contour, in mm.
        slice_thickness: Distance between slices (mm).

    Returns:
        Tuple of ``(label, dims_cm, area_cm2, volume_cm3, GI, depths_cm,
        saved_pngs, valid_slices)``.
    """
    # --- Load mesh
    
    dic = check_brain(file_path)
    if dic["label"] == "not_brain":
        reply = QMessageBox.question(parent,"Check measurement",
            "The imported mesh does not represent a human brain. Do you want to continue processing it?",   # message
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.No:
            return dic["label"],[],0,0,0,[],[],[]
        
    mesh = pv.read(str(file_path))
    print(f"[STL All Hallmarks] Loaded mesh: {mesh}")

    # --- Bounds / dims (mm)
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    brain_dim = [x_max - x_min, y_max - y_min, z_max - z_min]
    print(f"[STL All Hallmarks] mesh dimensions (mm): {brain_dim}")

    brain_dim_cm = [dim/10 for dim in brain_dim]
    brain_dim_cm = sorted(brain_dim_cm, reverse=True)

    # --- Direction-dependent slicing setup
    sd = _stl_slice_setup(mesh, brain_dim, Slice_direction)

    if slice_thickness <= 0:
        slice_thickness = 0.5
    slice_positions = np.arange(sd["low"], sd["high"], slice_thickness)
    N = len(slice_positions)
    if N == 0:
        print(f"[STL All Hallmarks] No slices to process (thickness too large vs. {Slice_direction} range).")
        return 0.0, [], []

    slice_thickness_eff = brain_dim[sd["axis_index"]] / N
    print(f"[STL All Hallmarks] Effective slice thickness: {slice_thickness_eff} mm")

    # --- Outputs
    os.makedirs(out_dir, exist_ok=True)
    out_dir_slices = os.path.join(out_dir, "stl_slices")
    os.makedirs(out_dir_slices, exist_ok=True)

    out_dir_origin = os.path.join(out_dir, "stl_orgin")
    os.makedirs(out_dir_origin, exist_ok=True)

    print(f"[STL All Hallmarks] Temp output dir: {out_dir}")

    window_size = (sd["image_width"], sd["image_height"])

    saved_pngs: list[str] = []
    valid_slices: list[int] = []
    rows = []
    kernel_px_values: list[int] = []
    slice_class_data: list = []
    sections_list: list[pv.PolyData] = []
    total_depth = []
    sum_inner_mm = 0.0
    sum_outer_mm = 0.0
    sum_area = 0.0
    # Raw per-slice data collected during the loop; the Simpson samples are built
    # AFTER the loop because the surface-connected cavity classification needs all
    # slices (tracking). Keyed by the loop index `idx`.
    lateral_samples: list[tuple[float, float]] = []
    volume_samples: list[tuple[float, float]] = []
    slice_records: list[SliceRecord] = []
    raw_lateral: list[tuple[int, float, float]] = []        # (idx, pos, inner_perim_mm)
    raw_volume: list[tuple[int, float, float]] = []         # (idx, pos, inner_area_mm)
    slice_png_by_idx: dict[int, tuple[str, int]] = {}       # idx -> (png path, line thickness)

    p = pv.Plotter(off_screen=True, window_size=window_size)
    p.set_background("white")
    p.parallel_projection = True
    cube_len = sd["cube_len"]
    max_dim = sd["max_dim"]
    _require_scale_calibration(
        p, Slice_direction, cube_len, mesh.center, float(slice_positions[0]),
        max_dim, sd["view_fn_name"],
    )
    calibration_metadata: list[dict] = []
    invalid_calibration_rows: list[dict] = []
    for idx, k in enumerate(slice_positions):
        normal, origin = slice_at(mesh, Slice_direction, k)
        section = mesh.slice(normal=normal, origin=origin)
        if section.n_points == 0:
            continue

        scale_cube = make_scale_cube(Slice_direction, cube_len, mesh.center, k, max_dim)

        plane = pv.Plane(center=origin,
                         direction=normal,
                         i_size=brain_dim[sd["pre_axis"]]*1.5, j_size=brain_dim[sd["next_axis"]]*1.5,
                         i_resolution=1, j_resolution=1)
        p.clear()
        p.add_mesh(scale_cube, color="red", lighting=False)
        p.add_mesh(fill_section_polydata(section) if fill_cross_section else section, color="#B4B4B4", lighting=False)
        prepare_orthographic_slice_render(p, getattr(p, sd["view_fn_name"]))

        # Screenshot (array for processing, file for debugging)
        img_rgb = p.screenshot(return_img=True, filename=os.path.join(out_dir_origin, f"image_{idx:03d}.png"))

        # Compute mm/px scale from the red cube
        mm_per_px, scale_meta, red_rect = calc_scale_with_metadata(img_rgb, cube_len)
        if mm_per_px is None:
            print(f"[STL Scale] Scale cube was not detected or failed validation. {scale_meta.get('calibration_status')}. Measurements for this slice were skipped.")
            invalid_calibration_rows.append({
                "Section": idx,
                "cube_len_mm": scale_meta.get("cube_len_mm"),
                "detected_cube_size_px": scale_meta.get("detected_cube_size_px"),
                "computed_mm_per_px": scale_meta.get("computed_mm_per_px"),
            })
            continue
        calibration_metadata.append(scale_meta)
        kernel_size_px = _to_kernel_px(kernel_size_mm, mm_per_px)


        # Prepare masks / contours (pixel space)
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h_img, w_img = bgr.shape[:2]
        thickness, _, radius_px = image_annotation_style(h_img, w_img, style="thin")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # Binary for contours. RETR_CCOMP so section holes/cavities are detected
        # as internal contours for the cavity correction; the top-level outer
        # contours match the previous RETR_EXTERNAL + contours_exclude result so
        # GI/perimeter are unchanged.
        bw = threshold_binary(gray, BINARY_THRESHOLD_DEFAULT, invert=True)
        contours, hierarchy = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        inner_filtered, internal_filtered = split_inner_and_internal_contours(
            contours, hierarchy, red_rect, bw.shape, float(min_contour_area) / (mm_per_px ** 2),
        )
        cv2.drawContours(bgr, inner_filtered, -1, tuple(_get_viz().contour_inner_color_bgr), thickness)

        # Outer contours: rebuild a mask from ONLY the kept inner contours so
        # noise blobs rejected by the inner filter can't produce spurious outer
        # components after morph-close.
        inner_mask = np.zeros_like(bw)
        cv2.drawContours(inner_mask, inner_filtered, -1, 255, thickness=cv2.FILLED)
        kernel = compute_kernel_convex(kernel_size_px)
        closed = cv2.morphologyEx(inner_mask, cv2.MORPH_CLOSE, kernel)
        outer_candidates, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outer_filtered = [c for c in outer_candidates if cv2.contourArea(c) * (mm_per_px ** 2) > float(min_contour_area)]
        cv2.drawContours(bgr, outer_filtered, -1, tuple(_get_viz().contour_outer_color_bgr), thickness)

        # Perimeters (mm). In-plane spacing is isotropic (single mm/px from the
        # cube), so Crofton resamples 1:1. arc_length keeps the exact prior value.
        area_px        = sum(cv2.contourArea(c)     for c in inner_filtered)
        inner_perim_px = sum(cv2.arcLength(c, True) for c in inner_filtered)
        outer_perim_px = sum(cv2.arcLength(c, True) for c in outer_filtered)
        if perimeter_method == "crofton":
            outer_envelope_mask = np.zeros_like(bw)
            cv2.drawContours(outer_envelope_mask, outer_filtered, -1, 255, thickness=cv2.FILLED)
            inner_perim_mm = mask_perimeter_mm(
                inner_mask, mm_per_px, mm_per_px, method="crofton",
                simplify=simplify_contours_for_perimeter, epsilon=contour_simplify_epsilon)
            outer_perim_mm = mask_perimeter_mm(
                outer_envelope_mask, mm_per_px, mm_per_px, method="crofton",
                simplify=simplify_contours_for_perimeter, epsilon=contour_simplify_epsilon)
        else:
            inner_perim_mm = inner_perim_px * mm_per_px
            outer_perim_mm = outer_perim_px * mm_per_px
        area_mm  = area_px * (mm_per_px ** 2)

        # Classify the rendered slice and gate the percent filter on it.
        # Slices that don't look like a full MRI fall back to the original
        # fixed-millimeter rule and stay "unclassified".
        slice_kind, slice_kind_conf = classify_slice_kind(bgr)
        use_percent_filter = slice_kind != "not_full_slice" and slice_kind_conf >= 0.7
        # Sulci classification: bin each kept defect by its depth as a
        # fraction of `max_dim` (longest brain extent in mm).
        depth_sets = empty_depth_sets()
        if inner_filtered:
            for cnt in inner_filtered:
                hull = cv2.convexHull(cnt, returnPoints=False, clockwise=True)  # Compute convex hull
                if hull is not None and np.all(np.diff(hull.ravel()) > 0) and len(hull) >= 3 and len(cnt) > 3:
                    defects = cv2.convexityDefects(cnt, hull)
                    if defects is not None:
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            start = tuple(cnt[s][0])
                            end = tuple(cnt[e][0])
                            far = tuple(cnt[f][0])
                            bgr = cv2.line(bgr, start, end, list(_get_viz().measurement_line_color_bgr), thickness)
                            if d > DEFECT_FIXED_POINT:
                                depth_mm = d * mm_per_px / DEFECT_FIXED_POINT
                                if use_percent_filter:
                                    keep = (SULCUS_TERTIARY_MIN_FRACTION * max_dim) < depth_mm < (SULCUS_PRIMARY_MAX_FRACTION * max_dim) and depth_mm > sulcus_depth_min("mm")
                                else:
                                    keep = depth_mm > sulcus_depth_min("mm")
                                if keep:
                                    if use_percent_filter:
                                        sulcus_class = classify_sulcus_depth(depth_mm, max_dim)
                                    else:
                                        sulcus_class = "unclassified"
                                    marker_color = SULCUS_CLASS_COLORS[sulcus_class]
                                    depth_sets[sulcus_class].append(depth_mm)
                                    bgr = cv2.circle(bgr, far, radius_px, marker_color, -1)

        depth = flatten_depth_sets(depth_sets)
        print(f"[STL All Hallmarks] slice {idx}: kind={slice_kind} ({slice_kind_conf:.2f}), {format_sulcus_class_summary(depth_sets)}")
        mean_depth = (sum(depth)/len(depth)) if depth else None
        total_depth.extend(depth)
        if use_percent_filter:
            per_class_cells = sulcus_export_cells(depth_sets)
            slice_class_data.append(depth_sets)
        else:
            per_class_cells = [None] * len(sulcus_export_columns("mm"))
            slice_class_data.append(None)
        rows.append([
            idx, slice_kind, len(inner_filtered), area_mm, inner_perim_mm, outer_perim_mm,
            len(depth),                         # n_defects
            (min(depth) if depth else None),    # min_depth_mm
            (max(depth) if depth else None),    # max_depth_mm
            mean_depth,                         # mean_depth_mm
            *per_class_cells,
        ])
        kernel_px_values.append(kernel_size_px)
        # Accumulate
        sum_inner_mm += inner_perim_mm
        sum_outer_mm += outer_perim_mm
        sum_area     += area_mm
        # Raw per-slice data + cavities/polys (physical mm relative to the cube
        # centre) for the cross-slice cavity tracker. STL is already in mm.
        raw_lateral.append((idx, float(k), inner_perim_mm))
        raw_volume.append((idx, float(k), area_mm))
        if cavity_correction_enabled:
            center_px = get_red_rect_offset(img_rgb)
            slice_records.append(SliceRecord(
                idx=idx, position_mm=float(k),
                cavities=[make_slice_cavity(c, mm_per_px, center_px) for c in internal_filtered],
                outer_polys_mm=[contour_to_mm(c, mm_per_px, center_px) for c in inner_filtered],
                hole_polys_mm=[contour_to_mm(c, mm_per_px, center_px) for c in internal_filtered],
            ))
#        GI_slice = (inner_perim_mm / outer_perim_mm) if outer_perim_mm > 0 else 0.0

        # Save annotated slice
        slice_path = os.path.join(out_dir_slices, f"slice_{idx:03d}.png")
        cv2.imwrite(slice_path, bgr)
        saved_pngs.append(slice_path)
        slice_png_by_idx[idx] = (slice_path, thickness)
        valid_slices.append(idx)
        plane["slice_idx"] = np.full(plane.n_points, idx, dtype=np.int32)
        section["slice_idx"] = np.full(section.n_points, idx, dtype=np.int32)
        sections_list.append(plane)
        sections_list.append(section)

    # ---- end with: plotter is fully and safely closed here ----

    # Surface-connected cavity correction (cross-slice tracking). Volume uses the
    # holes-filled area minus surface-connected cavities (enclosed voids stay
    # solid); the surface lateral gets the open-cavity wall perimeter added. GI is
    # untouched. When disabled, reproduces the previous behaviour exactly.
    if cavity_correction_enabled:
        cavity_corr = cavity_correction_tracking(
            slice_records, area_threshold_mm2=float(cavity_area_threshold_mm2))
        volume_samples = [(pos, max(0.0, area - cavity_corr.area_subtract(idx)))
                          for (idx, pos, area) in raw_volume]
        lateral_samples = [(pos, perim + cavity_corr.perim_add(idx))
                           for (idx, pos, perim) in raw_lateral]
        # Outline the surface-connected cavities in YELLOW on the saved slice PNGs
        # (classification is only known after tracking, so redraw here).
        for c_idx, c_contours in cavity_corr.surface_connected_by_idx.items():
            png_thick = slice_png_by_idx.get(c_idx)
            if png_thick is None:
                continue
            png_path, line_thick = png_thick
            img = cv2.imread(png_path)
            if img is None:
                continue
            cv2.drawContours(img, c_contours, -1, (0, 255, 255), int(max(1, line_thick)))
            cv2.imwrite(png_path, img)
    else:
        cavity_corr = CavityCorrection.empty()
        volume_samples = [(pos, area) for (_idx, pos, area) in raw_volume]
        lateral_samples = [(pos, perim) for (_idx, pos, perim) in raw_lateral]

    # Totals
    # Volume = ∫ cross-section area dh via Simpson's rule (mm³ → /1000 cm³).
    # The exact mesh.volume is kept only for verification.
    brain_volume = volume_simpson(volume_samples) / 1000.0
    _mesh_volume_cm3 = abs(float(mesh.volume)) / 1000.0
    print(f"[STL All Hallmarks] Volume={brain_volume:.3f} cm³")
    # Surface area = Simpson lateral (∫ exterior perimeter dh) + top & bottom caps.
    # The exact mesh.area (mm²) is kept only for verification.
    total_area_mm2, inner_area_slice_sum_mm2, caps_area_mm2 = total_surface_area_simpson(
        lateral_samples, volume_samples)
    mesh_area_mm2 = float(mesh.area)
    Area = total_area_mm2 / 100.0
    print(f"[STL All Hallmarks] Surface area: lateral+caps={Area:.3f} cm²")
    if cavity_correction_enabled:
        print(f"[STL All Hallmarks] Cavity correction: "
              f"{cavity_corr.n_surface_connected} surface-connected, "
              f"{cavity_corr.n_enclosed} enclosed; "
              f"area removed={cavity_corr.total_cavity_area_mm2 / 100:.3f} cm²")
    GI_total = (sum_inner_mm / sum_outer_mm) if sum_outer_mm > 0 else 0.0
    comp_3D  = compactness_3D(brain_volume, Area)
    # 3-D area-based gyrification index = exact mesh surface ÷ convex-hull
    # surface (separate from the 2-D perimeter GI). STL is already in mm.
    gi_3d, _gi3d_area, hull_area_mm2 = area_based_gi_3d(mesh)

    # Snapshot mm-unit depths for the per-mm Excel summary row before
    # the in-place cm conversion below.
    total_depth_mm = list(total_depth)
    total_depth = [x/10 for x in total_depth]

    mean_total = (sum(total_depth)/ len(total_depth))  if len(total_depth)>0 else None
    if sections_list:
        all_slices_mesh = pv.merge(sections_list)
#        all_slices_mesh.active_scalars_name = "RGB"
        slice_mesh_path = os.path.join(out_dir, "all_slices_mesh.vtk")
        all_slices_mesh.save(slice_mesh_path)
    else:
        all_slices_mesh = pv.PolyData()

    # Save per-slice + totals to Excel using the shared spec layout.
    try:
        from helpers.results_excel_format import (
            ResultsSheet, write_results_workbook, subtype_mean,
            gi_3d_note,
        )

        overall_n = len(total_depth_mm)
        overall_mean = (sum(total_depth_mm) / overall_n) if overall_n else None

        results_rows = []
        for r, dsets, png_path, kernel_px, cal_meta in zip(rows, slice_class_data, saved_pngs, kernel_px_values, calibration_metadata):
            idx, _kind, _ncont, area_mm, inner_perim_mm, outer_perim_mm = r[:6]
            lgi = ((inner_perim_mm / outer_perim_mm)
                   if outer_perim_mm else None)
            compact = (compactness_2D(area_mm, inner_perim_mm)
                       if inner_perim_mm else None)
            d = dsets if isinstance(dsets, dict) else {}
            results_rows.append({
                "Section": idx,
                "Kernel px": int(kernel_px),
                "Cube length (mm)": cal_meta.get("cube_len_mm"),
                "Detected cube size (px)": cal_meta.get("detected_cube_size_px"),
                "Computed mm per px": cal_meta.get("computed_mm_per_px"),
                "Area": area_mm,
                "Perimeter": inner_perim_mm,
                "LGI": lgi,
                "Compactness": compact,
                "PrimarySulciCount": len(d.get("primary", []) or []),
                "SecondarySulciCount": len(d.get("secondary", []) or []),
                "TertiarySulciCount": len(d.get("tertiary", []) or []),
                "UnclassifiedSulciCount": len(d.get("unclassified", []) or []),
                "PrimaryMeanDepth": subtype_mean(
                    None, d.get("primary", []) or []),
                "SecondaryMeanDepth": subtype_mean(
                    None, d.get("secondary", []) or []),
                "TertiaryMeanDepth": subtype_mean(
                    None, d.get("tertiary", []) or []),
                "UnclassifiedMeanDepth": subtype_mean(
                    None, d.get("unclassified", []) or []),
                "_section_link": png_path,
            })
        results_rows.extend(invalid_calibration_rows)

        parameters = {
            "Kernel size (mm)": float(kernel_size_mm),
            "Kernel size (px)": (
                f"per-slice, range {min(kernel_px_values)}-{max(kernel_px_values)}"
                if kernel_px_values else None),
            "Pixel spacing": "0.1 mm/pixel (rendering); per-slice from cube",
            "Slice thickness": float(slice_thickness),
            "Filtered threshold (mm²)": float(min_contour_area),
            "Sulcus depth threshold (mm)": float(sulcus_depth_min("mm")),
            "Slice direction": Slice_direction,
            "Length unit": "cm",
            "Perimeter method": perimeter_method,
            "Contour simplification enabled": bool(simplify_contours_for_perimeter),
            "Contour simplification epsilon": float(contour_simplify_epsilon),
            "Cavity correction": "on" if cavity_correction_enabled else "off",
        }
        if cavity_correction_enabled:
            parameters["Cavity area threshold (mm²)"] = float(cavity_area_threshold_mm2)
        totals = {
            "Volume (cm^3)": round(float(brain_volume), 4),
            "Surface Area (cm^2)": round(float(Area), 4),
            "Area lateral surface (cm^2)": round(inner_area_slice_sum_mm2 / 100, 4),
            "Surface area caps (cm^2)": round(caps_area_mm2 / 100, 4),
            "GI": round(float(GI_total), 4),
            "GI 3D (convex hull)": (round(float(gi_3d), 4) if gi_3d is not None else None),
            "Convex hull area (cm^2)": (round(hull_area_mm2 / 100, 4) if hull_area_mm2 else None),
            "Compactness": round(float(comp_3D), 4),
            "Total sulci count": int(overall_n),
            "Mean sulci depth (mm)": (round(float(overall_mean), 4)
                                       if overall_mean is not None else None),
        }
        if cavity_correction_enabled:
            totals.update({
                "Number of surface-connected cavities": int(cavity_corr.n_surface_connected),
                "Number of enclosed cavities": int(cavity_corr.n_enclosed),
                "Surface connected cavity area (cm^2)": round(cavity_corr.total_cavity_area_mm2 / 100, 4),
            })
        sheet = ResultsSheet(
            sheet_name=os.path.basename(file_path) or "Results",
            file_name=os.path.basename(file_path),
            folder=os.path.dirname(file_path) or None,
            parameters=parameters,
            rows=results_rows,
            extra_columns=(
                "Kernel px",
                "Cube length (mm)",
                "Detected cube size (px)",
                "Computed mm per px",
            ),
            totals=totals,
            totals_notes={
                "GI 3D (convex hull)": gi_3d_note(gi_3d),
            },
            drop_empty_columns=True,
        )
        xlsx_path = os.path.join(out_dir, "Mesh_Allmarks.xlsx")
        write_results_workbook(xlsx_path, [sheet])
        print(f"[STL All Hallmarks] Saved Excel → {xlsx_path}")
    except Exception as ex:
        print(f"[STL All Hallmarks] WARN: could not save Excel: {ex}")


    # Always return a 3-tuple
    return dic["label"], brain_dim_cm, Area, brain_volume, GI_total, comp_3D ,total_depth ,saved_pngs, valid_slices 


def compute_stl_lGI(
    parent,
    file_path: str,
    out_dir: str,
    min_contour_area: float = 20.0,
    kernel_size_mm: float = DEFAULT_KERNEL_SIZE_MM,
    slice_thickness: float = 0.5,
    build_solid: bool = False,
    Slice_direction: str = "Y",
    perimeter_method: str = DEFAULT_PERIMETER_METHOD,
    simplify_contours_for_perimeter: bool = DEFAULT_SIMPLIFY_CONTOURS_FOR_PERIMETER,
    contour_simplify_epsilon: float = DEFAULT_CONTOUR_SIMPLIFY_EPSILON,
    fill_cross_section: bool = DEFAULT_FILL_CROSS_SECTION,
):
    """Compute the gyrification index (GI) from an STL mesh via slice rendering.

    GI = total exterior perimeter / total closed-envelope perimeter across all Y slices.
    Optionally builds an extruded 3-D solid from the outer contours.

    Args:
        parent: Qt parent widget for message boxes.
        file_path: Path to the ``.stl`` file.
        out_dir: Output directory.
        min_contour_area: Minimum contour area (pixels) to keep.
        kernel_size_mm: Morph-close kernel diameter, in mm.
        slice_thickness: Distance between slices (mm).
        build_solid: If True, extrude outer contours into a solid mesh.

    Returns:
        Tuple of ``(label, dims_cm, GI_total, saved_pngs, valid_slices)``.
    """
    
    dic = check_brain(file_path)
    

    if dic["label"] == "not_brain":
        reply = QMessageBox.question(parent,"Check measurement",
            "The imported mesh does not represent a human brain. Do you want to continue processing it?",   # message
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.No:
            return dic["label"],[],0,[],[]
    # --- Load mesh
    mesh = pv.read(str(file_path))
    print(f"[STL lGI] Loaded mesh: {mesh}")

    # --- Bounds / dims (mm)
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    brain_dim = [x_max - x_min, y_max - y_min, z_max - z_min]
    print(f"[STL lGI] mesh dimensions (mm): {brain_dim}")

    brain_dim_cm = [dim/10 for dim in brain_dim]
    brain_dim_cm = sorted(brain_dim_cm, reverse=True)

    # --- Direction-dependent slicing setup
    sd = _stl_slice_setup(mesh, brain_dim, Slice_direction)

    if slice_thickness <= 0:
        slice_thickness = 0.5
    slice_positions = np.arange(sd["low"], sd["high"], slice_thickness)
    N = len(slice_positions)
    if N == 0:
        print(f"[STL lGI] No slices to process (thickness too large vs. {Slice_direction} range).")
        return dic["label"],[],0,[],[]

    slice_thickness_eff = brain_dim[sd["axis_index"]] / N
    print(f"[STL lGI] Effective slice thickness: {slice_thickness_eff} mm")

    # --- Outputs
    os.makedirs(out_dir, exist_ok=True)
    out_dir_slices = os.path.join(out_dir, "stl_slices")
    os.makedirs(out_dir_slices, exist_ok=True)

    out_dir_origin = os.path.join(out_dir, "stl_orgin")
    os.makedirs(out_dir_origin, exist_ok=True)

    print(f"[STL lGI] Temp output dir: {out_dir}")

    window_size = (sd["image_width"], sd["image_height"])

    saved_pngs: list[str] = []
    valid_slices: list[int] = []
    rows = []
    all_3d_contours = []
    sections_list: list[pv.PolyData] = []
    sum_inner_mm = 0.0
    sum_outer_mm = 0.0
    lateral_samples: list[tuple[float, float]] = []  # (position mm, exterior perimeter mm)

    p = pv.Plotter(off_screen=True, window_size=window_size)
    p.set_background("white")
    p.parallel_projection = True
    cube_len = sd["cube_len"]
    _require_scale_calibration(
        p, Slice_direction, cube_len, mesh.center, float(slice_positions[0]),
        sd["max_dim"], sd["view_fn_name"],
    )

    for idx, k in enumerate(slice_positions):
        normal, origin = slice_at(mesh, Slice_direction, k)
        section = mesh.slice(normal=normal, origin=origin)
        if section.n_points == 0:
            continue

        scale_cube = make_scale_cube(Slice_direction, cube_len, mesh.center, k, sd["max_dim"])

        plane = pv.Plane(center=origin,
                         direction=normal,
                         i_size=brain_dim[sd["pre_axis"]]*1.5, j_size=brain_dim[sd["next_axis"]]*1.5,
                         i_resolution=1, j_resolution=1)
        p.clear()
        p.add_mesh(scale_cube, color="red", lighting=False)
        p.add_mesh(fill_section_polydata(section) if fill_cross_section else section, color="#B4B4B4", lighting=False)
        prepare_orthographic_slice_render(p, getattr(p, sd["view_fn_name"]))

        # Screenshot (array for processing, file for debugging)
        img_rgb = p.screenshot(return_img=True, filename=os.path.join(out_dir_origin, f"image_{idx:03d}.png"))

        # Compute mm/px scale from the red cube
        mm_per_px, scale_meta, red_rect = calc_scale_with_metadata(img_rgb, cube_len)
        if mm_per_px is None:
            print(f"[STL Scale] Scale cube was not detected or failed validation. {scale_meta.get('calibration_status')}. Measurements for this slice were skipped.")
            continue
        kernel_size_px = _to_kernel_px(kernel_size_mm, mm_per_px)


        # Prepare masks / contours (pixel space). Mirror the All-Hallmarks
        # pipeline so the enclosed (outer) and inner surface contours are drawn
        # identically here — RETR_CCOMP + split so section holes/cavities are
        # detected the same way. Sulci markers are intentionally NOT drawn.
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h_img, w_img = bgr.shape[:2]
        thickness, _, _ = image_annotation_style(h_img, w_img, style="thin")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # Binary for contours
        bw = threshold_binary(gray, BINARY_THRESHOLD_DEFAULT, invert=True)
        contours, hierarchy = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # Inner surface contours: exclude red ref + area filter
        inner_filtered, _internal_filtered = split_inner_and_internal_contours(
            contours, hierarchy, red_rect, bw.shape, float(min_contour_area) / (mm_per_px ** 2),
        )
        cv2.drawContours(bgr, inner_filtered, -1, tuple(_get_viz().contour_inner_color_bgr), thickness)

        # Outer contours: rebuild a mask from ONLY the kept inner contours so
        # noise blobs rejected by the inner filter can't produce spurious outer
        # components after morph-close.
        inner_mask = np.zeros_like(bw)
        cv2.drawContours(inner_mask, inner_filtered, -1, 255, thickness=cv2.FILLED)
        kernel = compute_kernel_convex(kernel_size_px)
        closed = cv2.morphologyEx(inner_mask, cv2.MORPH_CLOSE, kernel)
        outer_candidates, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outer_filtered = [c for c in outer_candidates if cv2.contourArea(c) * (mm_per_px ** 2) > float(min_contour_area)]
        cv2.drawContours(bgr, outer_filtered, -1, tuple(_get_viz().contour_outer_color_bgr), thickness)

        # Perimeters (mm) — Crofton on the filled masks when selected.
        inner_perim_px = sum(cv2.arcLength(c, True) for c in inner_filtered)
        outer_perim_px = sum(cv2.arcLength(c, True) for c in outer_filtered)
        if perimeter_method == "crofton":
            outer_envelope_mask = np.zeros_like(bw)
            cv2.drawContours(outer_envelope_mask, outer_filtered, -1, 255, thickness=cv2.FILLED)
            inner_perim_mm = mask_perimeter_mm(
                inner_mask, mm_per_px, mm_per_px, method="crofton",
                simplify=simplify_contours_for_perimeter, epsilon=contour_simplify_epsilon)
            outer_perim_mm = mask_perimeter_mm(
                outer_envelope_mask, mm_per_px, mm_per_px, method="crofton",
                simplify=simplify_contours_for_perimeter, epsilon=contour_simplify_epsilon)
        else:
            inner_perim_mm = inner_perim_px * mm_per_px
            outer_perim_mm = outer_perim_px * mm_per_px

        # Accumulate
        sum_inner_mm += inner_perim_mm
        sum_outer_mm += outer_perim_mm
        lateral_samples.append((float(k), inner_perim_mm))
        GI_slice = (inner_perim_mm / outer_perim_mm) if outer_perim_mm > 0 else 0.0
        rows.append([idx, len(inner_filtered), kernel_size_px, inner_perim_mm, outer_perim_mm, GI_slice, scale_meta])

        # Save annotated slice
        slice_path = os.path.join(out_dir_slices, f"slice_{idx:03d}.png")
        cv2.imwrite(slice_path, bgr)
        saved_pngs.append(slice_path)
        valid_slices.append(idx)
        plane["slice_idx"] = np.full(plane.n_points, idx, dtype=np.int32)
        section["slice_idx"] = np.full(section.n_points, idx, dtype=np.int32)
        sections_list.append(section)
        sections_list.append(plane)

            # Store 3D outer contours in mm coords for optional solid build
        if build_solid and outer_filtered:
            offset_2d = get_red_rect_offset(img_rgb)  # center of red cube in px
            for cnt in outer_filtered:
                pts = cnt.squeeze()
                if pts.ndim != 2 or pts.shape[0] < 3:
                    continue
                # XZ from pixels -> mm; Y from slice index
                aligned = pts - offset_2d
                cnt_3d = np.column_stack([
                    aligned[:, 0] * mm_per_px,                       # X
                    np.full((pts.shape[0],), idx * slice_thickness_eff),  # Y
                    aligned[:, 1] * mm_per_px,                       # Z
                ])
                all_3d_contours.append(cnt_3d)

    # ---- end with: plotter is fully and safely closed here ----

    # Totals
    GI_total = (sum_inner_mm / sum_outer_mm) if sum_outer_mm > 0 else 0.0
    print(f"[STL lGI] GI_total = {GI_total:.6f}")

    # Lateral surface area = ∫ exterior perimeter dh via Simpson's rule.
    inner_area_slice_sum_mm2 = lateral_area_simpson(lateral_samples)
    # 3-D area-based GI = exact mesh surface ÷ convex-hull surface.
    gi_3d, _gi3d_area, hull_area_mm2 = area_based_gi_3d(mesh)

    if sections_list:
        all_slices_mesh = pv.merge(sections_list)
#        all_slices_mesh.active_scalars_name = "RGB"
        slice_mesh_path = os.path.join(out_dir, "all_slices_mesh.vtk")
        all_slices_mesh.save(slice_mesh_path)
    else:
        all_slices_mesh = pv.PolyData()

    # Save per-slice + total to Excel (spec layout, like All-hallmarks).
    try:
        from helpers.results_excel_format import (
            build_measurement_sheet, write_results_workbook, gi_3d_note)
        results_rows = [{
            "Section": r[0], "Contours": r[1], "Kernel px": r[2],
            "Cube length (mm)": r[-1].get("cube_len_mm"),
            "Detected cube size (px)": r[-1].get("detected_cube_size_px"),
            "Computed mm per px": r[-1].get("computed_mm_per_px"),
            "Perimeter": r[3], "Closed-envelope perimeter": r[4], "LGI": r[5],
        } for r in rows]
        parameters = {
            "Kernel size (mm)": float(kernel_size_mm),
            "Slice thickness": float(slice_thickness),
            "Slice direction": Slice_direction,
            "Filtered threshold (mm²)": float(min_contour_area),
        }
        totals = {
            "GI": round(float(GI_total), 4),
            "GI 3D (convex hull)": (round(float(gi_3d), 4) if gi_3d is not None else None),
            "Convex hull area (cm^2)": (round(hull_area_mm2 / 100, 4) if hull_area_mm2 else None),
            "Area lateral surface (cm^2)": round(inner_area_slice_sum_mm2 / 100, 4),
        }
        sheet = build_measurement_sheet(
            file_path, "LGI", results_rows, parameters, totals,
            extra_columns=("Contours", "Kernel px", "Cube length (mm)",
                           "Detected cube size (px)", "Computed mm per px",
                           "Closed-envelope perimeter"),
            totals_notes={"GI 3D (convex hull)": gi_3d_note(gi_3d)})
        xlsx_path = os.path.join(out_dir, "STL_lGI.xlsx")
        write_results_workbook(xlsx_path, [sheet])
        print(f"[STL lGI] Saved Excel → {xlsx_path}")
    except Exception as ex:
        print(f"[STL lGI] WARN: could not save Excel: {ex}")

    # Optional: build extruded solid (can be fragile on some macOS OpenGL stacks)
    if build_solid and all_3d_contours:
        try:
            solids = []
            for cnt in all_3d_contours:
                if cnt.shape[0] < 3:
                    continue
                n = cnt.shape[0]
                faces = np.hstack(([n], np.arange(n, dtype=np.int64))).astype(np.int64)
                surf = pv.PolyData(cnt, faces=faces).triangulate()
                extruded = surf.extrude([0, slice_thickness_eff, 0], capping=True)
                solids.append(extruded)
            if solids:
                merged = solids[0].copy()
                for part in solids[1:]:
                    merged = merged.merge(part)
                solid_path = os.path.join(out_dir, "solid_volume_convex.stl")
                merged.save(solid_path)
                print(f"[STL lGI] Saved extruded solid → {solid_path}")
        except Exception as ex:
            print(f"[STL lGI] NOTE: solid build skipped: {ex}")

    # Always return a 3-tuple
    return dic["label"], brain_dim_cm, float(GI_total), saved_pngs, valid_slices


def compute_stl_volume(
    parent,
    file_path: str,
    out_dir: str,
    min_contour_area: float = 20.0,
    slice_thickness: float = 0.5,
    Slice_direction: str = "Y",
    fill_cross_section: bool = DEFAULT_FILL_CROSS_SECTION,
    cavity_correction_enabled: bool = DEFAULT_CAVITY_CORRECTION_ENABLED,
    cavity_area_threshold_mm2: float = DEFAULT_CAVITY_AREA_THRESHOLD_MM2):
    """Compute brain volume (cm³) from an STL mesh by integrating slice cross-section areas.

    Args:
        parent: Qt parent widget for message boxes.
        file_path: Path to the ``.stl`` file.
        out_dir: Output directory.
        min_contour_area: Minimum contour area (pixels) to keep.
        slice_thickness: Distance between slices (mm).

    Returns:
        Tuple of ``(label, dims_cm, volume_cm3, saved_pngs, valid_slices)``.
    """
    # --- Load mesh
    
    dic = check_brain(file_path)
    
    if dic["label"] == "not_brain":
        reply = QMessageBox.question(parent,"Check measurement",
            "The imported mesh does not represent a human brain. Do you want to continue processing it?",   # message
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.No:
            return dic["label"],[],0,[],[]
        
    mesh = pv.read(str(file_path))
    print(f"[STL Volume] Loaded mesh: {mesh}")

    # --- Bounds / dims (mm)
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    brain_dim = [x_max - x_min, y_max - y_min, z_max - z_min]
    print(f"[STL Volume] mesh dimensions (mm): {brain_dim}")

    brain_dim_cm = [dim/10 for dim in brain_dim]
    brain_dim_cm = sorted(brain_dim_cm, reverse=True)

    # --- Direction-dependent slicing setup
    sd = _stl_slice_setup(mesh, brain_dim, Slice_direction)

    if slice_thickness <= 0:
        slice_thickness = 0.5
    slice_positions = np.arange(sd["low"], sd["high"], slice_thickness)
    N = len(slice_positions)
    if N == 0:
        print(f"[STL Volume] No slices to process (thickness too large vs. {Slice_direction} range).")
        return dic["label"],[],0,[],[]

    slice_thickness_eff = brain_dim[sd["axis_index"]] / N
    print(f"[STL Volume] Effective slice thickness: {slice_thickness_eff} mm")

    # --- Outputs
    os.makedirs(out_dir, exist_ok=True)
    out_dir_slices = os.path.join(out_dir, "stl_slices")
    os.makedirs(out_dir_slices, exist_ok=True)

    out_dir_origin = os.path.join(out_dir, "stl_orgin")
    os.makedirs(out_dir_origin, exist_ok=True)

    print(f"[STL Volume] Temp output dir: {out_dir}")

    window_size = (sd["image_width"], sd["image_height"])

    saved_pngs: list[str] = []
    valid_slices: list[int] = []
    rows = []
    sections_list: list[pv.PolyData] = []
    sum_area = 0.0
    # Raw per-slice data for the surface-connected cavity correction (applied
    # after the loop). volume_samples is built from raw_volume post-correction.
    raw_volume: list[tuple[int, float, float]] = []   # (idx, pos mm, area mm²)
    slice_records: list[SliceRecord] = []
    slice_png_by_idx: dict[int, tuple[str, int]] = {}  # idx -> (png path, line thickness)

    p = pv.Plotter(off_screen=True, window_size=window_size)
    p.set_background("white")
    p.parallel_projection = True
    cube_len = sd["cube_len"]
    _require_scale_calibration(
        p, Slice_direction, cube_len, mesh.center, float(slice_positions[0]),
        sd["max_dim"], sd["view_fn_name"],
    )

    for idx, k in enumerate(slice_positions):
        normal, origin = slice_at(mesh, Slice_direction, k)
        section = mesh.slice(normal=normal, origin=origin)
        if section.n_points == 0:
            continue

        scale_cube = make_scale_cube(Slice_direction, cube_len, mesh.center, k, sd["max_dim"])
        plane = pv.Plane(center=origin,
                         direction=normal,
                         i_size=brain_dim[sd["pre_axis"]]*1.5, j_size=brain_dim[sd["next_axis"]]*1.5,
                         i_resolution=1, j_resolution=1)
        p.clear()
        p.add_mesh(scale_cube, color="red", lighting=False)
        p.add_mesh(fill_section_polydata(section) if fill_cross_section else section, color="#B4B4B4", lighting=False)
        prepare_orthographic_slice_render(p, getattr(p, sd["view_fn_name"]))

        # Screenshot (array for processing, file for debugging)
        img_rgb = p.screenshot(return_img=True, filename=os.path.join(out_dir_origin, f"image_{idx:03d}.png"))

        # Compute mm/px scale from the red cube
        mm_per_px, scale_meta, red_rect = calc_scale_with_metadata(img_rgb, cube_len)
        if mm_per_px is None:
            print(f"[STL Scale] Scale cube was not detected or failed validation. {scale_meta.get('calibration_status')}. Measurements for this slice were skipped.")
            continue


        # Prepare masks / contours (pixel space)
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h_img, w_img = bgr.shape[:2]
        thickness, _, _ = image_annotation_style(h_img, w_img, style="thin")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # Binary for contours. RETR_CCOMP so holes are detected as internal
        # contours for the surface-connected cavity correction; the top-level
        # contours match the previous RETR_EXTERNAL result so area is unchanged.
        bw = threshold_binary(gray, BINARY_THRESHOLD_DEFAULT, invert=True)
        contours, hierarchy = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        inner_filtered, internal_filtered = split_inner_and_internal_contours(
            contours, hierarchy, red_rect, bw.shape, float(min_contour_area) / (mm_per_px ** 2),
        )
        cv2.drawContours(bgr, inner_filtered, -1, tuple(_get_viz().contour_inner_color_bgr), thickness)

        # Cross-section area (holes-filled outer region; surface-connected
        # cavities are subtracted after the loop, enclosed voids stay solid).
        area_px  = sum(cv2.contourArea(c)     for c in inner_filtered)
        area_mm  = area_px * (mm_per_px ** 2)

        rows.append([idx, len(inner_filtered), area_mm, scale_meta])
        # Accumulate
        sum_area     += area_mm
        raw_volume.append((idx, float(k), area_mm))
        if cavity_correction_enabled:
            center_px = get_red_rect_offset(img_rgb)
            slice_records.append(SliceRecord(
                idx=idx, position_mm=float(k),
                cavities=[make_slice_cavity(c, mm_per_px, center_px) for c in internal_filtered],
                outer_polys_mm=[contour_to_mm(c, mm_per_px, center_px) for c in inner_filtered],
                hole_polys_mm=[contour_to_mm(c, mm_per_px, center_px) for c in internal_filtered],
            ))

        # Save annotated slice
        slice_path = os.path.join(out_dir_slices, f"slice_{idx:03d}.png")
        cv2.imwrite(slice_path, bgr)
        saved_pngs.append(slice_path)
        slice_png_by_idx[idx] = (slice_path, thickness)
        valid_slices.append(idx)
        plane["slice_idx"] = np.full(plane.n_points, idx, dtype=np.int32)
        section["slice_idx"] = np.full(section.n_points, idx, dtype=np.int32)
        sections_list.append(section)
        sections_list.append(plane)


    # ---- end with: plotter is fully and safely closed here ----

    # Surface-connected cavity correction: subtract open-cavity area from the
    # volume integral (enclosed voids stay solid); yellow-outline open cavities.
    if cavity_correction_enabled:
        cavity_corr = cavity_correction_tracking(
            slice_records, area_threshold_mm2=float(cavity_area_threshold_mm2))
        volume_samples = [(pos, max(0.0, area - cavity_corr.area_subtract(idx)))
                          for (idx, pos, area) in raw_volume]
        for c_idx, c_contours in cavity_corr.surface_connected_by_idx.items():
            png_thick = slice_png_by_idx.get(c_idx)
            if png_thick is None:
                continue
            png_path, line_thick = png_thick
            img = cv2.imread(png_path)
            if img is None:
                continue
            cv2.drawContours(img, c_contours, -1, (0, 255, 255), int(max(1, line_thick)))
            cv2.imwrite(png_path, img)
    else:
        cavity_corr = CavityCorrection.empty()
        volume_samples = [(pos, area) for (_idx, pos, area) in raw_volume]

    # Totals — Volume = ∫ cross-section area dh via Simpson's rule (mm³ → cm³).
    # Exact mesh.volume kept only for verification.
    brain_volume = volume_simpson(volume_samples) / 1000.0
    _mesh_volume_cm3 = abs(float(mesh.volume)) / 1000.0
    print(f"[STL Volume] Volume={brain_volume:.3f} cm³")
    if cavity_correction_enabled:
        print(f"[STL Volume] Cavity correction: "
              f"{cavity_corr.n_surface_connected} surface-connected, "
              f"{cavity_corr.n_enclosed} enclosed; "
              f"area removed={cavity_corr.total_cavity_area_mm2:.3f} mm²")
    if sections_list:
        all_slices_mesh = pv.merge(sections_list)
#        all_slices_mesh.active_scalars_name = "RGB"
        slice_mesh_path = os.path.join(out_dir, "all_slices_mesh.vtk")
        all_slices_mesh.save(slice_mesh_path)
    else:
        all_slices_mesh = pv.PolyData()
        
    # Save per-slice + total to Excel (spec layout, like All-hallmarks).
    try:
        from helpers.results_excel_format import build_measurement_sheet, write_results_workbook
        results_rows = [{
            "Section": r[0], "Contours": r[1],
            "Cube length (mm)": r[-1].get("cube_len_mm"),
            "Detected cube size (px)": r[-1].get("detected_cube_size_px"),
            "Computed mm per px": r[-1].get("computed_mm_per_px"),
            "Area": r[2],
        } for r in rows]
        parameters = {
            "Slice thickness": float(slice_thickness),
            "Slice direction": Slice_direction,
            "Filtered threshold (mm²)": float(min_contour_area),
            "Cavity correction": "on" if cavity_correction_enabled else "off",
        }
        if cavity_correction_enabled:
            parameters["Cavity area threshold (mm²)"] = float(cavity_area_threshold_mm2)
        totals = {"Volume (cm^3)": round(float(brain_volume), 4)}
        if cavity_correction_enabled:
            totals.update({
                "Number of surface-connected cavities": int(cavity_corr.n_surface_connected),
                "Number of enclosed cavities": int(cavity_corr.n_enclosed),
                "Cavity area removed (mm²)": round(float(cavity_corr.total_cavity_area_mm2), 4),
            })
        sheet = build_measurement_sheet(
            file_path, "Volume", results_rows, parameters, totals,
            extra_columns=("Contours", "Cube length (mm)",
                           "Detected cube size (px)", "Computed mm per px"))
        xlsx_path = os.path.join(out_dir, "Mesh_Volume.xlsx")
        write_results_workbook(xlsx_path, [sheet])
        print(f"[STL Volume] Saved Excel → {xlsx_path}")
    except Exception as ex:
        print(f"[STL Volume] WARN: could not save Excel: {ex}")


    # Always return a 3-tuple
    return dic["label"], brain_dim_cm, brain_volume,saved_pngs, valid_slices


def compute_stl_area(
    parent,
    file_path: str,
    out_dir: str,
    min_contour_area: float = 20.0,
    slice_thickness: float = 0.5,
    Slice_direction: str = "Y",
    perimeter_method: str = DEFAULT_PERIMETER_METHOD,
    simplify_contours_for_perimeter: bool = DEFAULT_SIMPLIFY_CONTOURS_FOR_PERIMETER,
    contour_simplify_epsilon: float = DEFAULT_CONTOUR_SIMPLIFY_EPSILON,
    fill_cross_section: bool = DEFAULT_FILL_CROSS_SECTION,
    cavity_correction_enabled: bool = DEFAULT_CAVITY_CORRECTION_ENABLED,
    cavity_area_threshold_mm2: float = DEFAULT_CAVITY_AREA_THRESHOLD_MM2):
    """Compute brain surface area (cm²) from an STL mesh.

    Sums exterior-contour perimeters across Y slices and multiplies by
    effective slice thickness.

    Args:
        parent: Qt parent widget for message boxes.
        file_path: Path to the ``.stl`` file.
        out_dir: Output directory.
        min_contour_area: Minimum contour area (pixels) to keep.
        slice_thickness: Distance between slices (mm).

    Returns:
        Tuple of ``(label, dims_cm, area_cm2, saved_pngs, valid_slices)``.
    """
    # --- Load mesh
    
    dic = check_brain(file_path)

    if dic["label"] == "not_brain":
        reply = QMessageBox.question(parent,"Check measurement",
            "The imported mesh does not represent a human brain. Do you want to continue processing it?",   # message
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.No:
            return dic["label"],[],0,[],[]
        
    mesh = pv.read(str(file_path))
    print(f"[STL Area] Loaded mesh: {mesh}")

    # --- Bounds / dims (mm)
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    brain_dim = [x_max - x_min, y_max - y_min, z_max - z_min]
    print(f"[STL Area] mesh dimensions (mm): {brain_dim}")

    brain_dim_cm = [dim/10 for dim in brain_dim]
    brain_dim_cm = sorted(brain_dim_cm, reverse=True)

    # --- Direction-dependent slicing setup
    sd = _stl_slice_setup(mesh, brain_dim, Slice_direction)

    if slice_thickness <= 0:
        slice_thickness = 0.5
    slice_positions = np.arange(sd["low"], sd["high"], slice_thickness)
    N = len(slice_positions)
    if N == 0:
        print(f"[STL Area] No slices to process (thickness too large vs. {Slice_direction} range).")
        return dic["label"],[],0,[],[]

    slice_thickness_eff = brain_dim[sd["axis_index"]] / N
    print(f"[STL Area] Effective slice thickness: {slice_thickness_eff} mm")

    # --- Outputs
    os.makedirs(out_dir, exist_ok=True)
    out_dir_slices = os.path.join(out_dir, "stl_slices")
    os.makedirs(out_dir_slices, exist_ok=True)

    out_dir_origin = os.path.join(out_dir, "stl_orgin")
    os.makedirs(out_dir_origin, exist_ok=True)

    print(f"[STL Area] Temp output dir: {out_dir}")

    window_size = (sd["image_width"], sd["image_height"])

    saved_pngs: list[str] = []
    valid_slices: list[int] = []
    rows = []
    sum_inner_mm = 0.0
    # Raw per-slice data for the surface-connected cavity correction (applied
    # after the loop): the open-cavity wall perimeter is added to the lateral
    # surface and the open-cavity area is subtracted from the cap area.
    raw_lateral: list[tuple[int, float, float]] = []   # (idx, pos mm, inner_perim_mm)
    raw_volume: list[tuple[int, float, float]] = []    # (idx, pos mm, inner_area_mm)
    slice_records: list[SliceRecord] = []
    slice_png_by_idx: dict[int, tuple[str, int]] = {}  # idx -> (png path, line thickness)
    sections_list: list[pv.PolyData] = []

    p = pv.Plotter(off_screen=True, window_size=window_size)
    p.set_background("white")
    p.parallel_projection = True
    cube_len = sd["cube_len"]
    _require_scale_calibration(
        p, Slice_direction, cube_len, mesh.center, float(slice_positions[0]),
        sd["max_dim"], sd["view_fn_name"],
    )

    for idx, k in enumerate(slice_positions):
        normal, origin = slice_at(mesh, Slice_direction, k)
        section = mesh.slice(normal=normal, origin=origin)
        if section.n_points == 0:
            continue

        scale_cube = make_scale_cube(Slice_direction, cube_len, mesh.center, k, sd["max_dim"])
        plane = pv.Plane(center=origin,
                         direction=normal,
                         i_size=brain_dim[sd["pre_axis"]]*1.5, j_size=brain_dim[sd["next_axis"]]*1.5,
                         i_resolution=1, j_resolution=1)
        p.clear()
        p.add_mesh(scale_cube, color="red", lighting=False)
        p.add_mesh(fill_section_polydata(section) if fill_cross_section else section, color="#B4B4B4", lighting=False)
        prepare_orthographic_slice_render(p, getattr(p, sd["view_fn_name"]))

        # Screenshot (array for processing, file for debugging)
        img_rgb = p.screenshot(return_img=True, filename=os.path.join(out_dir_origin, f"image_{idx:03d}.png"))

        # Compute mm/px scale from the red cube
        mm_per_px, scale_meta, red_rect = calc_scale_with_metadata(img_rgb, cube_len)
        if mm_per_px is None:
            print(f"[STL Scale] Scale cube was not detected or failed validation. {scale_meta.get('calibration_status')}. Measurements for this slice were skipped.")
            continue


        # Prepare masks / contours (pixel space)
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h_img, w_img = bgr.shape[:2]
        thickness, _, _ = image_annotation_style(h_img, w_img, style="thin")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # Binary for contours. RETR_CCOMP so holes are detected as internal
        # contours for the surface-connected cavity correction; the top-level
        # contours match the previous RETR_EXTERNAL result so perimeter is unchanged.
        bw = threshold_binary(gray, BINARY_THRESHOLD_DEFAULT, invert=True)
        contours, hierarchy = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        inner_filtered, internal_filtered = split_inner_and_internal_contours(
            contours, hierarchy, red_rect, bw.shape, float(min_contour_area) / (mm_per_px ** 2),
        )
        cv2.drawContours(bgr, inner_filtered, -1, tuple(_get_viz().contour_inner_color_bgr), thickness)


        # Perimeters (mm) — Crofton on the filled mask when selected.
        inner_perim_px = sum(cv2.arcLength(c, True) for c in inner_filtered)
        if perimeter_method == "crofton":
            inner_mask = np.zeros_like(bw)
            cv2.drawContours(inner_mask, inner_filtered, -1, 255, thickness=cv2.FILLED)
            inner_perim_mm = mask_perimeter_mm(
                inner_mask, mm_per_px, mm_per_px, method="crofton",
                simplify=simplify_contours_for_perimeter, epsilon=contour_simplify_epsilon)
        else:
            inner_perim_mm = inner_perim_px * mm_per_px
        inner_area_mm = sum(cv2.contourArea(c) for c in inner_filtered) * (mm_per_px ** 2)

        rows.append([idx, len(inner_filtered), inner_perim_mm, scale_meta])
        # Accumulate
        sum_inner_mm += inner_perim_mm
        raw_lateral.append((idx, float(k), inner_perim_mm))
        raw_volume.append((idx, float(k), inner_area_mm))
        if cavity_correction_enabled:
            center_px = get_red_rect_offset(img_rgb)
            slice_records.append(SliceRecord(
                idx=idx, position_mm=float(k),
                cavities=[make_slice_cavity(c, mm_per_px, center_px) for c in internal_filtered],
                outer_polys_mm=[contour_to_mm(c, mm_per_px, center_px) for c in inner_filtered],
                hole_polys_mm=[contour_to_mm(c, mm_per_px, center_px) for c in internal_filtered],
            ))


        # Save annotated slice
        slice_path = os.path.join(out_dir_slices, f"slice_{idx:03d}.png")
        cv2.imwrite(slice_path, bgr)
        saved_pngs.append(slice_path)
        slice_png_by_idx[idx] = (slice_path, thickness)
        valid_slices.append(idx)
        plane["slice_idx"] = np.full(plane.n_points, idx, dtype=np.int32)
        section["slice_idx"] = np.full(section.n_points, idx, dtype=np.int32)
        sections_list.append(section)
        sections_list.append(plane)


    # ---- end with: plotter is fully and safely closed here ----

    # Surface-connected cavity correction: add the open-cavity wall perimeter to
    # the lateral surface and subtract the open-cavity area from the cap area
    # (enclosed voids stay solid). Yellow-outline the open cavities once known.
    if cavity_correction_enabled:
        cavity_corr = cavity_correction_tracking(
            slice_records, area_threshold_mm2=float(cavity_area_threshold_mm2))
        lateral_samples = [(pos, perim + cavity_corr.perim_add(idx))
                           for (idx, pos, perim) in raw_lateral]
        volume_samples = [(pos, max(0.0, area - cavity_corr.area_subtract(idx)))
                          for (idx, pos, area) in raw_volume]
        for c_idx, c_contours in cavity_corr.surface_connected_by_idx.items():
            png_thick = slice_png_by_idx.get(c_idx)
            if png_thick is None:
                continue
            png_path, line_thick = png_thick
            img = cv2.imread(png_path)
            if img is None:
                continue
            cv2.drawContours(img, c_contours, -1, (0, 255, 255), int(max(1, line_thick)))
            cv2.imwrite(png_path, img)
    else:
        cavity_corr = CavityCorrection.empty()
        lateral_samples = [(pos, perim) for (_idx, pos, perim) in raw_lateral]
        volume_samples = [(pos, area) for (_idx, pos, area) in raw_volume]

    # Totals — surface area = Simpson lateral (∫ perimeter dh) + top & bottom
    # caps (mm² → /100 cm²). The exact mesh.area is kept only for verification.
    total_area_mm2, lateral_mm2, caps_mm2 = total_surface_area_simpson(
        lateral_samples, volume_samples)
    mesh_area_mm2 = float(mesh.area)
    Area = total_area_mm2 / 100
    print(f"[STL Area] Surface area: lateral+caps={Area:.3f} cm²")
    if cavity_correction_enabled:
        print(f"[STL Area] Cavity correction: "
              f"{cavity_corr.n_surface_connected} surface-connected, "
              f"{cavity_corr.n_enclosed} enclosed; "
              f"wall perim added={cavity_corr.total_wall_perim_mm:.3f} mm")
    if sections_list:
        all_slices_mesh = pv.merge(sections_list)
#        all_slices_mesh.active_scalars_name = "RGB"
        slice_mesh_path = os.path.join(out_dir, "all_slices_mesh.vtk")
        all_slices_mesh.save(slice_mesh_path)
    else:
        all_slices_mesh = pv.PolyData()
        
    # Save per-slice + total to Excel (spec layout, like All-hallmarks).
    try:
        from helpers.results_excel_format import build_measurement_sheet, write_results_workbook
        results_rows = [{
            "Section": r[0], "Contours": r[1],
            "Cube length (mm)": r[-1].get("cube_len_mm"),
            "Detected cube size (px)": r[-1].get("detected_cube_size_px"),
            "Computed mm per px": r[-1].get("computed_mm_per_px"),
            "Perimeter": r[2],
        } for r in rows]
        parameters = {
            "Slice thickness": float(slice_thickness),
            "Slice direction": Slice_direction,
            "Filtered threshold (mm²)": float(min_contour_area),
            "Perimeter method": perimeter_method,
            "Surface area method": "lateral + caps",
            "Cavity correction": "on" if cavity_correction_enabled else "off",
        }
        if cavity_correction_enabled:
            parameters["Cavity area threshold (mm²)"] = float(cavity_area_threshold_mm2)
        totals = {
            "Surface Area (cm^2)": round(float(Area), 4),
            "Surface area lateral (cm^2)": round(lateral_mm2 / 100, 4),
            "Surface area caps (cm^2)": round(caps_mm2 / 100, 4),
        }
        if cavity_correction_enabled:
            totals.update({
                "Number of surface-connected cavities": int(cavity_corr.n_surface_connected),
                "Number of enclosed cavities": int(cavity_corr.n_enclosed),
                "Cavity wall perimeter added (mm)": round(float(cavity_corr.total_wall_perim_mm), 4),
            })
        sheet = build_measurement_sheet(
            file_path, "Area", results_rows, parameters, totals,
            extra_columns=("Contours", "Cube length (mm)",
                           "Detected cube size (px)", "Computed mm per px"))
        xlsx_path = os.path.join(out_dir, "Mesh_Area.xlsx")
        write_results_workbook(xlsx_path, [sheet])
        print(f"[STL Area] Saved Excel → {xlsx_path}")
    except Exception as ex:
        print(f"[STL Area] WARN: could not save Excel: {ex}")


    # Always return a 3-tuple
    return dic["label"], brain_dim_cm, Area, saved_pngs, valid_slices



def compute_stl_sulci_depth(
    parent,
    file_path: str,
    out_dir: str,
    min_contour_area: float = 20.0,
    slice_thickness: float = 0.5,
    Slice_direction: str = "Y",
    fill_cross_section: bool = DEFAULT_FILL_CROSS_SECTION):
    """Compute sulci depths from convexity defects across STL mesh slices.

    Args:
        parent: Qt parent widget for message boxes.
        file_path: Path to the ``.stl`` file.
        out_dir: Output directory.
        min_contour_area: Minimum contour area (pixels) to keep.
        slice_thickness: Distance between slices (mm).

    Returns:
        Tuple of ``(label, dims_cm, depths_mm, saved_pngs, valid_slices)``.
    """
    # --- Load mesh
    
    dic = check_brain(file_path)
    
    if dic["label"] == "not_brain":
        reply = QMessageBox.question(parent,"Check measurement",
            "The imported mesh does not represent a human brain. Do you want to continue processing it?",   # message
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.No:
            return dic["label"],[],[],[],[]
        
    mesh = pv.read(str(file_path))
    print(f"[STL Sulci depth] Loaded mesh: {mesh}")

    # --- Bounds / dims (mm)
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    brain_dim = [x_max - x_min, y_max - y_min, z_max - z_min]
    print(f"[STL Sulci depth] mesh dimensions (mm): {brain_dim}")

    brain_dim_cm = [dim/10 for dim in brain_dim]
    brain_dim_cm = sorted(brain_dim_cm, reverse=True)

    # --- Direction-dependent slicing setup
    sd = _stl_slice_setup(mesh, brain_dim, Slice_direction)

    if slice_thickness <= 0:
        slice_thickness = 0.5
    slice_positions = np.arange(sd["low"], sd["high"], slice_thickness)
    N = len(slice_positions)
    if N == 0:
        print(f"[STL Sulci depth] No slices to process (thickness too large vs. {Slice_direction} range).")
        return dic["label"],[],[],[],[]

    slice_thickness_eff = brain_dim[sd["axis_index"]] / N
    print(f"[STL Sulci depth] Effective slice thickness: {slice_thickness_eff} mm")

    # --- Outputs
    os.makedirs(out_dir, exist_ok=True)
    out_dir_slices = os.path.join(out_dir, "stl_slices")
    os.makedirs(out_dir_slices, exist_ok=True)

    out_dir_origin = os.path.join(out_dir, "stl_orgin")
    os.makedirs(out_dir_origin, exist_ok=True)

    print(f"[STL Sulci depth] Temp output dir: {out_dir}")

    window_size = (sd["image_width"], sd["image_height"])

    saved_pngs: list[str] = []
    valid_slices: list[int] = []
    rows = []
    slice_class_data: list = []
    total_depth = []
    sections_list: list[pv.PolyData] = []

    p = pv.Plotter(off_screen=True, window_size=window_size)
    p.set_background("white")
    p.parallel_projection = True
    cube_len = sd["cube_len"]
    max_dim = sd["max_dim"]
    _require_scale_calibration(
        p, Slice_direction, cube_len, mesh.center, float(slice_positions[0]),
        max_dim, sd["view_fn_name"],
    )
    for idx, k in enumerate(slice_positions):
        normal, origin = slice_at(mesh, Slice_direction, k)
        section = mesh.slice(normal=normal, origin=origin)
        if section.n_points == 0:
            continue

        scale_cube = make_scale_cube(Slice_direction, cube_len, mesh.center, k, max_dim)
        plane = pv.Plane(center=origin,
                         direction=normal,
                         i_size=brain_dim[sd["pre_axis"]]*1.5, j_size=brain_dim[sd["next_axis"]]*1.5,
                         i_resolution=1, j_resolution=1)
        p.clear()
        p.add_mesh(scale_cube, color="red", lighting=False)
        p.add_mesh(fill_section_polydata(section) if fill_cross_section else section, color="#B4B4B4", lighting=False)
        prepare_orthographic_slice_render(p, getattr(p, sd["view_fn_name"]))

        # Screenshot (array for processing, file for debugging)
        img_rgb = p.screenshot(return_img=True, filename=os.path.join(out_dir_origin, f"image_{idx:03d}.png"))

        # Compute mm/px scale from the red cube
        mm_per_px, scale_meta, red_rect = calc_scale_with_metadata(img_rgb, cube_len)
        if mm_per_px is None:
            print(f"[STL Scale] Scale cube was not detected or failed validation. {scale_meta.get('calibration_status')}. Measurements for this slice were skipped.")
            continue


        # Prepare masks / contours (pixel space). Mirror the All-Hallmarks
        # render + inner-contour extraction (filled gray section, RETR_CCOMP +
        # split) so the inner surface contour matches. The enclosed (outer)
        # envelope is intentionally NOT drawn here; sulci markers go on top.
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h_img, w_img = bgr.shape[:2]
        thickness, _, radius_px = image_annotation_style(h_img, w_img, style="thin")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # Binary for contours
        bw = threshold_binary(gray, BINARY_THRESHOLD_DEFAULT, invert=True)
        contours, hierarchy = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # Inner surface contours: exclude red ref + area filter
        inner_filtered, _internal_filtered = split_inner_and_internal_contours(
            contours, hierarchy, red_rect, bw.shape, float(min_contour_area) / (mm_per_px ** 2),
        )
        cv2.drawContours(bgr, inner_filtered, -1, tuple(_get_viz().contour_inner_color_bgr), thickness)

        # Classify the rendered slice and gate the percent filter on it.
        slice_kind, slice_kind_conf = classify_slice_kind(bgr)
        use_percent_filter = slice_kind != "not_full_slice" and slice_kind_conf >= 0.7
        # Sulci classification: bin each kept defect by its depth as a
        # fraction of `max_dim` (longest brain extent in mm).
        depth_sets = empty_depth_sets()
        if inner_filtered:
            for cnt in inner_filtered:
                hull = cv2.convexHull(cnt, returnPoints=False, clockwise=True)  # Compute convex hull
                if hull is not None and np.all(np.diff(hull.ravel()) > 0) and len(hull) >= 3 and len(cnt) > 3:
                    defects = cv2.convexityDefects(cnt, hull)
                    if defects is not None:
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            start = tuple(cnt[s][0])
                            end = tuple(cnt[e][0])
                            far = tuple(cnt[f][0])
                            bgr = cv2.line(bgr, start, end, list(_get_viz().measurement_line_color_bgr), thickness)
                            if d > DEFECT_FIXED_POINT:
                                mm_per_fixed = mm_per_px / DEFECT_FIXED_POINT
                                depth_mm = d *mm_per_fixed
                                if use_percent_filter:
                                    keep = (SULCUS_TERTIARY_MIN_FRACTION * max_dim) < depth_mm < (SULCUS_PRIMARY_MAX_FRACTION * max_dim) and depth_mm > sulcus_depth_min("mm")
                                else:
                                    keep = depth_mm > sulcus_depth_min("mm")
                                if keep:
                                    if use_percent_filter:
                                        sulcus_class = classify_sulcus_depth(depth_mm, max_dim)
                                    else:
                                        sulcus_class = "unclassified"
                                    marker_color = SULCUS_CLASS_COLORS[sulcus_class]
                                    depth_sets[sulcus_class].append(depth_mm)
                                    bgr = cv2.circle(bgr, far, radius_px, marker_color, -1)

        depth = flatten_depth_sets(depth_sets)
        print(f"[STL Sulci depth] slice {idx}: kind={slice_kind} ({slice_kind_conf:.2f}), {format_sulcus_class_summary(depth_sets)}")
        mean_depth = (sum(depth)/len(depth)) if depth else None
        total_depth.extend(depth)
        if use_percent_filter:
            per_class_cells = sulcus_export_cells(depth_sets)
            slice_class_data.append(depth_sets)
        else:
            per_class_cells = [None] * len(sulcus_export_columns("mm"))
            slice_class_data.append(None)
        rows.append([
            idx, slice_kind, len(inner_filtered),
            len(depth),                         # n_defects
            (min(depth) if depth else None),    # min_depth_mm
            (max(depth) if depth else None),    # max_depth_mm
            mean_depth,                         # mean_depth_mm
            *per_class_cells,
            scale_meta,                         # calibration metadata (last)
        ])


        # Save annotated slice
        slice_path = os.path.join(out_dir_slices, f"slice_{idx:03d}.png")
        cv2.imwrite(slice_path, bgr)
        saved_pngs.append(slice_path)
        valid_slices.append(idx)
        section["slice_idx"] = np.full(section.n_points, idx, dtype=np.int32)
        plane["slice_idx"] = np.full(plane.n_points, idx, dtype=np.int32)
        sections_list.append(section)
        sections_list.append(plane)
      
    # ---- end with: plotter is fully and safely closed here ----

    mean_total = (sum(total_depth)/ len(total_depth))  if len(total_depth)>0 else None
    
    if sections_list:
        all_slices_mesh = pv.merge(sections_list)
#        all_slices_mesh.active_scalars_name = "RGB"
        slice_mesh_path = os.path.join(out_dir, "all_slices_mesh.vtk")
        all_slices_mesh.save(slice_mesh_path)
    else:
        all_slices_mesh = pv.PolyData()
        
    # Save per-slice + total to Excel (spec layout, like All-hallmarks).
    try:
        from helpers.results_excel_format import (
            build_measurement_sheet, write_results_workbook, subtype_mean)
        results_rows = []
        for r, dsets in zip(rows, slice_class_data):
            d = dsets if isinstance(dsets, dict) else {}
            results_rows.append({
                "Section": r[0],
                "Contours": r[2],
                "Cube length (mm)": r[-1].get("cube_len_mm"),
                "Detected cube size (px)": r[-1].get("detected_cube_size_px"),
                "Computed mm per px": r[-1].get("computed_mm_per_px"),
                "PrimarySulciCount": len(d.get("primary", []) or []),
                "SecondarySulciCount": len(d.get("secondary", []) or []),
                "TertiarySulciCount": len(d.get("tertiary", []) or []),
                "UnclassifiedSulciCount": len(d.get("unclassified", []) or []),
                "PrimaryMeanDepth": subtype_mean(None, d.get("primary", []) or []),
                "SecondaryMeanDepth": subtype_mean(None, d.get("secondary", []) or []),
                "TertiaryMeanDepth": subtype_mean(None, d.get("tertiary", []) or []),
                "UnclassifiedMeanDepth": subtype_mean(None, d.get("unclassified", []) or []),
            })
        overall_n = len(total_depth)
        overall_mean = (sum(total_depth) / overall_n) if overall_n else None
        parameters = {
            "Slice thickness": float(slice_thickness),
            "Slice direction": Slice_direction,
            "Filtered threshold (mm²)": float(min_contour_area),
            "Sulcus depth threshold (mm)": float(sulcus_depth_min("mm")),
        }
        totals = {
            "Total sulci count": int(overall_n),
            "Min sulci depth (mm)": (round(float(min(total_depth)), 4) if total_depth else None),
            "Max sulci depth (mm)": (round(float(max(total_depth)), 4) if total_depth else None),
            "Mean sulci depth (mm)": (round(float(overall_mean), 4) if overall_mean is not None else None),
        }
        sheet = build_measurement_sheet(
            file_path, "Sulci depth", results_rows, parameters, totals,
            extra_columns=("Contours", "Cube length (mm)",
                           "Detected cube size (px)", "Computed mm per px"))
        xlsx_path = os.path.join(out_dir, "Mesh_Sulci_depth.xlsx")
        write_results_workbook(xlsx_path, [sheet])
        print(f"[STL Sulci depth] Saved Excel → {xlsx_path}")
    except Exception as ex:
        print(f"[STL Sulci depth] WARN: could not save Excel: {ex}")


    # Always return a 3-tuple
    return dic["label"], brain_dim_cm, total_depth ,saved_pngs, valid_slices


def compute_compactness_stl(
    parent,
    file_path: str,
    out_dir: str,
    min_contour_area: float = 20.0,
    slice_thickness: float = 0.5,
    Slice_direction: str = "Y",
    cavity_correction_enabled: bool = DEFAULT_CAVITY_CORRECTION_ENABLED,
    cavity_area_threshold_mm2: float = DEFAULT_CAVITY_AREA_THRESHOLD_MM2):
    """Compute 3D compactness (sphericity) from an STL mesh.

    Uses the exact STL ``mesh.volume`` and ``mesh.area`` totals, then applies
    ``compactness_3D(V, SA)``.

    Args:
        parent: Qt parent widget for message boxes.
        file_path: Path to the ``.stl`` file.
        out_dir: Output directory.
        min_contour_area: Minimum contour area (pixels) to keep.
        slice_thickness: Distance between slices (mm).

    Returns:
        Tuple of ``(label, dims_cm, compactness, saved_pngs, valid_slices)``.
    """
    # --- Load mesh
    dic = check_brain(file_path)

    if dic["label"] == "not_brain":
        reply = QMessageBox.question(parent, "Check measurement",
            "The imported mesh does not represent a human brain. Do you want to continue processing it?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.No:
            return dic["label"], [], 0, [], []

    mesh = pv.read(str(file_path))
    print(f"[STL Compactness] Loaded mesh: {mesh}")

    # --- Bounds / dims (mm)
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    brain_dim = [x_max - x_min, y_max - y_min, z_max - z_min]
    print(f"[STL Compactness] mesh dimensions (mm): {brain_dim}")

    brain_dim_cm = [dim / 10 for dim in brain_dim]
    brain_dim_cm = sorted(brain_dim_cm, reverse=True)

    # --- Direction-dependent slicing setup
    sd = _stl_slice_setup(mesh, brain_dim, Slice_direction)

    if slice_thickness <= 0:
        slice_thickness = 0.5
    slice_positions = np.arange(sd["low"], sd["high"], slice_thickness)
    N = len(slice_positions)
    if N == 0:
        print(f"[STL Compactness] No slices to process (thickness too large vs. {Slice_direction} range).")
        return dic["label"], [], 0, [], []

    slice_thickness_eff = brain_dim[sd["axis_index"]] / N
    print(f"[STL Compactness] Effective slice thickness: {slice_thickness_eff} mm")

    # --- Outputs
    os.makedirs(out_dir, exist_ok=True)
    out_dir_slices = os.path.join(out_dir, "stl_slices")
    os.makedirs(out_dir_slices, exist_ok=True)

    out_dir_origin = os.path.join(out_dir, "stl_orgin")
    os.makedirs(out_dir_origin, exist_ok=True)

    print(f"[STL Compactness] Temp output dir: {out_dir}")

    window_size = (sd["image_width"], sd["image_height"])

    saved_pngs: list[str] = []
    valid_slices: list[int] = []
    rows = []
    sections_list: list[pv.PolyData] = []
    sum_area_mm2 = 0.0
    sum_inner_mm = 0.0
    # Raw per-slice data for the surface-connected cavity correction (applied
    # after the loop) so compactness mirrors the Area/Volume tools.
    raw_lateral: list[tuple[int, float, float]] = []   # (idx, pos mm, inner_perim_mm)
    raw_volume: list[tuple[int, float, float]] = []    # (idx, pos mm, inner_area_mm)
    slice_records: list[SliceRecord] = []
    slice_png_by_idx: dict[int, tuple[str, int]] = {}  # idx -> (png path, line thickness)

    p = pv.Plotter(off_screen=True, window_size=window_size)
    p.set_background("white")
    p.parallel_projection = True
    cube_len = sd["cube_len"]
    _require_scale_calibration(
        p, Slice_direction, cube_len, mesh.center, float(slice_positions[0]),
        sd["max_dim"], sd["view_fn_name"],
    )

    for idx, k in enumerate(slice_positions):
        normal, origin = slice_at(mesh, Slice_direction, k)
        section = mesh.slice(normal=normal, origin=origin)
        if section.n_points == 0:
            continue

        scale_cube = make_scale_cube(Slice_direction, cube_len, mesh.center, k, sd["max_dim"])
        plane = pv.Plane(center=origin,
                         direction=normal,
                         i_size=brain_dim[sd["pre_axis"]] * 1.5, j_size=brain_dim[sd["next_axis"]] * 1.5,
                         i_resolution=1, j_resolution=1)
        p.clear()
        p.add_mesh(scale_cube, color="red", lighting=False)
        p.add_mesh(section, color="black", lighting=False)
        prepare_orthographic_slice_render(p, getattr(p, sd["view_fn_name"]))

        # Screenshot
        img_rgb = p.screenshot(return_img=True, filename=os.path.join(out_dir_origin, f"image_{idx:03d}.png"))

        # Compute mm/px scale from the red cube
        mm_per_px, scale_meta, red_rect = calc_scale_with_metadata(img_rgb, cube_len)
        if mm_per_px is None:
            print(f"[STL Scale] Scale cube was not detected or failed validation. {scale_meta.get('calibration_status')}. Measurements for this slice were skipped.")
            continue

        # Prepare masks / contours (pixel space)
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h_img, w_img = bgr.shape[:2]
        thickness, _, _ = image_annotation_style(h_img, w_img, style="thin")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # Binary for contours. RETR_CCOMP so holes feed the cavity correction;
        # top-level contours match the previous RETR_EXTERNAL result.
        bw = threshold_binary(gray, BINARY_THRESHOLD_DEFAULT, invert=True)
        contours, hierarchy = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        inner_filtered, internal_filtered = split_inner_and_internal_contours(
            contours, hierarchy, red_rect, bw.shape, float(min_contour_area) / (mm_per_px ** 2),
        )
        cv2.drawContours(bgr, inner_filtered, -1, tuple(_get_viz().contour_inner_color_bgr), thickness)

        # Convert pixel measurements to physical units (holes-filled outer region).
        area_px = sum(cv2.contourArea(c) for c in inner_filtered)
        inner_perim_px = sum(cv2.arcLength(c, True) for c in inner_filtered)
        area_mm2 = area_px * (mm_per_px ** 2)
        inner_perim_mm = inner_perim_px * mm_per_px
        comp_2D = compactness_2D(area_mm2, inner_perim_mm)

        # Accumulate
        sum_area_mm2 += area_mm2
        sum_inner_mm += inner_perim_mm
        raw_lateral.append((idx, float(k), inner_perim_mm))
        raw_volume.append((idx, float(k), area_mm2))
        if cavity_correction_enabled:
            center_px = get_red_rect_offset(img_rgb)
            slice_records.append(SliceRecord(
                idx=idx, position_mm=float(k),
                cavities=[make_slice_cavity(c, mm_per_px, center_px) for c in internal_filtered],
                outer_polys_mm=[contour_to_mm(c, mm_per_px, center_px) for c in inner_filtered],
                hole_polys_mm=[contour_to_mm(c, mm_per_px, center_px) for c in internal_filtered],
            ))

        # Save annotated slice
        slice_path = os.path.join(out_dir_slices, f"slice_{idx:03d}.png")
        cv2.imwrite(slice_path, bgr)
        saved_pngs.append(slice_path)
        slice_png_by_idx[idx] = (slice_path, thickness)
        valid_slices.append(idx)
        rows.append([idx, len(inner_filtered), comp_2D, scale_meta])
        plane["slice_idx"] = np.full(plane.n_points, idx, dtype=np.int32)
        section["slice_idx"] = np.full(section.n_points, idx, dtype=np.int32)
        sections_list.append(section)
        sections_list.append(plane)

    # ---- end of slice loop ----

    # Surface-connected cavity correction (mirrors the Area/Volume tools): wall
    # perimeter added to the surface lateral, open-cavity area subtracted from
    # volume + caps. Enclosed voids stay solid; yellow-outline open cavities.
    if cavity_correction_enabled:
        cavity_corr = cavity_correction_tracking(
            slice_records, area_threshold_mm2=float(cavity_area_threshold_mm2))
        lateral_samples = [(pos, perim + cavity_corr.perim_add(idx))
                           for (idx, pos, perim) in raw_lateral]
        volume_samples = [(pos, max(0.0, area - cavity_corr.area_subtract(idx)))
                          for (idx, pos, area) in raw_volume]
        for c_idx, c_contours in cavity_corr.surface_connected_by_idx.items():
            png_thick = slice_png_by_idx.get(c_idx)
            if png_thick is None:
                continue
            png_path, line_thick = png_thick
            img = cv2.imread(png_path)
            if img is None:
                continue
            cv2.drawContours(img, c_contours, -1, (0, 255, 255), int(max(1, line_thick)))
            cv2.imwrite(png_path, img)
    else:
        cavity_corr = CavityCorrection.empty()
        lateral_samples = [(pos, perim) for (_idx, pos, perim) in raw_lateral]
        volume_samples = [(pos, area) for (_idx, pos, area) in raw_volume]

    # Totals (mm → cm conversions). Volume = ∫ area dh via Simpson's rule;
    # exact mesh.volume kept only for verification.
    Volume = volume_simpson(volume_samples) / 1000.0      # cm³
    _mesh_volume_cm3 = abs(float(mesh.volume)) / 1000.0
    # Surface area = Simpson lateral + caps (cm²); exact mesh.area for verification.
    _total_area_mm2, _lat_mm2, _caps_mm2 = total_surface_area_simpson(
        lateral_samples, volume_samples)
    Area = _total_area_mm2 / 100.0                        # cm²
    print(f"[STL Compactness] Volume={Volume:.3f} cm³ "
          f"Surface area: lateral+caps={Area:.3f} cm²")
    comp_3D = compactness_3D(Volume, Area)

    if sections_list:
        all_slices_mesh = pv.merge(sections_list)
        slice_mesh_path = os.path.join(out_dir, "all_slices_mesh.vtk")
        all_slices_mesh.save(slice_mesh_path)
    else:
        all_slices_mesh = pv.PolyData()

    # Save per-slice + total to Excel (spec layout, like All-hallmarks).
    try:
        from helpers.results_excel_format import build_measurement_sheet, write_results_workbook
        results_rows = [{
            "Section": r[0], "Contours": r[1],
            "Cube length (mm)": r[-1].get("cube_len_mm"),
            "Detected cube size (px)": r[-1].get("detected_cube_size_px"),
            "Computed mm per px": r[-1].get("computed_mm_per_px"),
            "Compactness": r[2],
        } for r in rows]
        parameters = {
            "Slice thickness": float(slice_thickness),
            "Slice direction": Slice_direction,
            "Filtered threshold (mm²)": float(min_contour_area),
            "Cavity correction": "on" if cavity_correction_enabled else "off",
        }
        if cavity_correction_enabled:
            parameters["Cavity area threshold (mm²)"] = float(cavity_area_threshold_mm2)
        totals = {
            "Compactness": round(float(comp_3D), 4),
            "Volume (cm^3)": round(float(Volume), 4),
            "Surface Area (cm^2)": round(float(Area), 4),
        }
        if cavity_correction_enabled:
            totals.update({
                "Number of surface-connected cavities": int(cavity_corr.n_surface_connected),
                "Number of enclosed cavities": int(cavity_corr.n_enclosed),
            })
        sheet = build_measurement_sheet(
            file_path, "Compactness", results_rows, parameters, totals,
            extra_columns=("Contours", "Cube length (mm)",
                           "Detected cube size (px)", "Computed mm per px"))
        xlsx_path = os.path.join(out_dir, "Mesh_Compactness.xlsx")
        write_results_workbook(xlsx_path, [sheet])
        print(f"[STL Compactness] Saved Excel → {xlsx_path}")
    except Exception as ex:
        print(f"[STL Compactness] WARN: could not save Excel: {ex}")

    # Always return a 5-tuple
    return dic["label"], brain_dim_cm, comp_3D, saved_pngs, valid_slices
