"""VTK mesh measurement functions for FetoMorph.

Same render-and-measure pipeline as the STL module, but adapted for VTK
meshes that may use arbitrary model units.  A ``Physical_dim`` parameter
supplies the real-world dimensions so that a ``mesh_dim_scaled`` factor
converts model units → physical mm.

Key differences from the STL pipeline:
    * ``BINARY_THRESHOLD_VTK = 150`` (vs 200 for STL) because VTK renders
      use a black background instead of white.
    * Slice direction is configurable (X / Y / Z) via cyclic axis indexing:
      ``pre_axis = (axis_index - 1) % 3``, ``next_axis = (axis_index + 1) % 3``.
"""

from __future__ import annotations

from deps import *
from helpers.helpers import (
    compute_kernel_convex,
    contours_exclude,
    split_inner_and_internal_contours,
    threshold_binary,
    mask_perimeter_mm,
    fill_section_polydata,
    calc_scale_with_metadata,
    get_red_rect_offset,
    slice_at,
    make_scale_cube,
    prepare_orthographic_slice_render,
    validate_scale_cube_sanity,
    compactness_3D,
    compactness_2D,
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
from helpers.check_mesh import check_brain
from helpers.slice_kind_classifier import classify_slice_kind
from helpers.cavities import (
    SliceRecord, CavityCorrection, cavity_correction_tracking,
    make_slice_cavity, contour_to_mm,
)
from managers.visualization_settings import get_active as _get_viz
from constants import (
    BINARY_THRESHOLD_VTK,
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

logger = logging.getLogger("fetomorph.vtk")
VTK_RENDER_WINDOW_SIZE = (1024, 1024)


def _to_kernel_px(kernel_size_mm: float, pixel_size_mm: float) -> int:
    px = max(3, int(round(float(kernel_size_mm) / max(float(pixel_size_mm), 1e-9))))
    return px


def _mesh_area_volume_physical(mesh, mesh_dim_scaled) -> tuple[float, float]:
    """Return physical ``(surface_area, volume)`` for a VTK mesh.

    Raw VTK datasets may be volumetric or unstructured, where ``.area`` can be
    zero because the dataset itself does not expose polygonal surface cells.
    Scale the geometry first, then measure the extracted external surface.
    """
    scale = np.asarray(mesh_dim_scaled, dtype=float)
    if scale.shape != (3,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError(f"Invalid VTK physical scale factors: {scale!r}")

    scaled_mesh = mesh.copy(deep=True)
    scaled_mesh.points = np.asarray(scaled_mesh.points, dtype=float) * scale

    surface = scaled_mesh.extract_surface().triangulate()
    area = float(surface.area)
    volume = abs(float(scaled_mesh.volume))
    if volume <= 0 and surface.n_cells > 0:
        volume = abs(float(surface.volume))
    if area <= 0:
        raise ValueError(
            "VTK mesh surface area is 0 after extract_surface(); "
            "the input appears to have no polygonal surface cells to measure."
        )
    return area, volume


def _view_fn(p, Slice_direction: str):
    return {"X": p.view_yz, "Y": p.view_xz, "Z": p.view_xy}[Slice_direction]


def _require_scale_calibration(
    p,
    Slice_direction: str,
    cube_len: float,
    physical_cube_len: float,
    mesh_center,
    sample_slice: float,
    offset: float,
) -> dict:
    scale_cube = make_scale_cube(Slice_direction, cube_len, mesh_center, sample_slice, offset)
    ok, metadata = validate_scale_cube_sanity(
        p,
        scale_cube,
        float(physical_cube_len),
        _view_fn(p, Slice_direction),
        background="black",
    )
    if not ok:
        raise ValueError(
            "Scale cube calibration failed before processing slices: "
            f"{metadata.get('calibration_status')}"
        )
    return metadata

    
# ----------------- main API -----------------
def compute_vtk_allmarks(
    parent,
    file_path: str,
    out_dir: str,
    min_contour_area: float = 20.0,
    kernel_size_mm: float = DEFAULT_KERNEL_SIZE_MM,
    Slice_direction: Literal["X", "Y", "Z"] = "Y",
    Physical_dim: Sequence[int] | None = None,
    unit: str = "mm",
    slice_thickness: float = 0.5,
    cavity_correction_enabled: bool = DEFAULT_CAVITY_CORRECTION_ENABLED,
    cavity_area_threshold_mm2: float = DEFAULT_CAVITY_AREA_THRESHOLD_MM2,
    perimeter_method: str = DEFAULT_PERIMETER_METHOD,
    simplify_contours_for_perimeter: bool = DEFAULT_SIMPLIFY_CONTOURS_FOR_PERIMETER,
    contour_simplify_epsilon: float = DEFAULT_CONTOUR_SIMPLIFY_EPSILON,
    fill_cross_section: bool = DEFAULT_FILL_CROSS_SECTION):
    """Compute all hallmarks (area, volume, GI, sulci depths) from a VTK mesh.

    Args:
        parent: Qt parent widget for message boxes.
        file_path: Path to the ``.vtk`` file.
        out_dir: Output directory.
        min_contour_area: Minimum contour area (pixels) to keep.
        kernel_size_mm: Morph-close kernel diameter, in mm.
        Slice_direction: Axis to slice along.
        Physical_dim: Real-world ``[X, Y, Z]`` dimensions for unit scaling.
        unit: Label for output units.
        slice_thickness: Distance between slices in model units.

    Returns:
        Tuple of ``(area, volume, GI, depths, saved_pngs, valid_slices)``.
    """
    # --- Load mesh
    mesh = pv.read(str(file_path))
    print(f"[VTK All Hallmarks] Loaded mesh: {mesh}")

    # --- Bounds / dims (mm)
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    mesh_dim = [x_max - x_min, y_max - y_min, z_max - z_min]
    # Scale factor: converts model units to physical mm per axis.
    # E.g. if the mesh is 50 units wide but physically 100 mm, scale = 2.0.
    mesh_dim_scaled = np.array(Physical_dim) / np.array(mesh_dim)

    axis_bounds = {
        "X": (x_min, x_max),
        "Y": (y_min, y_max),
        "Z": (z_min, z_max),
    }

    # --- Slice positions along the chosen axis
    if slice_thickness <= 0:
        slice_thickness = 0.05

    low, high = axis_bounds[Slice_direction]
    slice_positions = np.arange(low, high, slice_thickness)

    N = len(slice_positions)
    if N == 0:
        print(f"[VTK All Hallmarks] No slices to process (thickness too large vs. {Slice_direction} range).")
        return 0.0, [], []

    # Effective slice thickness in physical units (model units × scale).
    axis_index = {"X": 0, "Y": 1, "Z": 2}.get(Slice_direction)
    slice_thickness_eff = mesh_dim[axis_index] / N
    slice_thickness_eff *= mesh_dim_scaled[axis_index]
    print(f"[VTK All Hallmarks] Effective slice thickness: {slice_thickness_eff} {unit}")

    # --- Outputs
    os.makedirs(out_dir, exist_ok=True)
    out_dir_slices = os.path.join(out_dir, "vtk_slices")
    os.makedirs(out_dir_slices, exist_ok=True)
    
    out_dir_origin = os.path.join(out_dir, "vtk_orgin")
    os.makedirs(out_dir_origin, exist_ok=True)
    
    print(f"[VTK All Hallmarks] Temp output dir: {out_dir}")

    saved_pngs: list[str] = []
    valid_slices: list[int] = []
    rows = []
    kernel_px_values: list[int] = []
    sections_list: list[pv.PolyData] = []
    total_depth = []
    slice_class_data: list = []
    sum_inner_mm = 0.0
    sum_outer_mm = 0.0
    sum_area = 0.0
    # `k` is in mesh units → scale onto the physical slice axis. Raw per-slice
    # records are collected during the loop; the Simpson samples are built AFTER
    # the loop because the surface-connected cavity classification needs all
    # slices (tracking). Keyed by the loop index `idx`.
    axis_scale = float(mesh_dim_scaled[axis_index])
    lateral_samples: list[tuple[float, float]] = []
    volume_samples: list[tuple[float, float]] = []
    slice_records: list[SliceRecord] = []
    raw_lateral: list[tuple[int, float, float]] = []   # (idx, pos, inner_perim_mm)
    raw_volume: list[tuple[int, float, float, float]] = []  # (idx, pos, inner_area_mm, area_perim_mm)
    slice_png_by_idx: dict[int, tuple[str, int]] = {}  # idx -> (png path, line thickness)

    # VTK renders use a black background → threshold is BINARY_THRESHOLD_VTK (150)
    # instead of the 200 used for white-background STL renders.
    p = pv.Plotter(off_screen=True, window_size=VTK_RENDER_WINDOW_SIZE)
    p.set_background("black")
    p.parallel_projection = True

    # Cyclic axis indexing: for a given slice axis, pre_axis and next_axis
    # give the two in-plane axes used for plane sizing and cube placement.
    pre_axis = (axis_index - 1) % 3
    next_axis = (axis_index + 1) % 3
    cube_len = max(1e-6, mesh_dim[0] / 10.0)
    scale_cube_offset = max(mesh_dim[pre_axis], mesh_dim[next_axis])
    physical_cube_len = cube_len * mesh_dim_scaled[0]
    _require_scale_calibration(
        p, Slice_direction, cube_len, physical_cube_len, mesh.center,
        float(slice_positions[0]), scale_cube_offset,
    )
    calibration_metadata: list[dict] = []
    invalid_calibration_rows: list[dict] = []

    for idx, k in enumerate(slice_positions):
        normal, origin = slice_at(mesh, Slice_direction, k)
        section = mesh.slice(normal=normal,origin=origin)

        if section.n_points == 0:
            continue

        scale_cube = make_scale_cube(Slice_direction, cube_len, mesh.center, k, scale_cube_offset)

        plane = pv.Plane(center=origin,
                 direction=normal,
                 i_size=mesh_dim[pre_axis]*1.5, j_size=mesh_dim[next_axis]*1.5,
                 i_resolution=1, j_resolution=1)
        p.clear()
        p.add_mesh(fill_section_polydata(section) if fill_cross_section else section, color="#B4B4B4", opacity=1, lighting=False)
        p.add_mesh(scale_cube, color="red", lighting=False)
        prepare_orthographic_slice_render(p, _view_fn(p, Slice_direction))

        img_rgb = p.screenshot(return_img=True, filename=os.path.join(out_dir_origin, f"image_{idx:03d}.png"))

        # Scale cube is rendered in physical mm (cube_len × mesh_dim_scaled[0]).
        mm_per_px, scale_meta, red_rect = calc_scale_with_metadata(img_rgb, physical_cube_len)
        if mm_per_px is None:
            print(f"[VTK Scale] Scale cube was not detected or failed validation. {scale_meta.get('calibration_status')}. Measurements for this slice were skipped.")
            invalid_calibration_rows.append({
                "Section": idx,
                "cube_len_mm": scale_meta.get("cube_len_mm"),
                "detected_cube_size_px": scale_meta.get("detected_cube_size_px"),
                "computed_mm_per_px": scale_meta.get("computed_mm_per_px"),
            })
            continue
        calibration_metadata.append(scale_meta)
        kernel_size_px = _to_kernel_px(kernel_size_mm, mm_per_px)

        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h_img, w_img = bgr.shape[:2]
        thickness, _, radius_px = image_annotation_style(h_img, w_img, style="thin")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # Use BINARY_THRESHOLD_VTK (150) for black-background VTK renders.
        bw = threshold_binary(gray, BINARY_THRESHOLD_VTK, invert=False)
        contours, hierarchy = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # internal_filtered (holes) is retained for the surface-connected cavity
        # correction below; hole accounting is owned by that correction, not by a
        # manual contour mode.
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

        area_perim_px = sum(cv2.contourArea(c) for c in inner_filtered)
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
        area_perim_mm  = area_perim_px * (mm_per_px ** 2)


        # Classify the rendered slice and gate the percent filter on it.
        slice_kind, slice_kind_conf = classify_slice_kind(bgr)
        use_percent_filter = slice_kind != "not_full_slice" and slice_kind_conf >= 0.7
        # Sulci classification: bin each kept defect by its depth as a
        # fraction of the brain's longest physical extent (mm).
        max_dim_mm = float(max(Physical_dim)) if Physical_dim else float(max(mesh_dim))
        depth_sets = empty_depth_sets()
        if inner_filtered:
            for cnt in inner_filtered:
                hull = cv2.convexHull(cnt, returnPoints=False, clockwise=True)
                if hull is not None and np.all(np.diff(hull.ravel()) > 0) and len(hull) >= 3 and len(cnt) > 3:
                    defects = cv2.convexityDefects(cnt, hull)
                    if defects is not None:
                        for i in range(defects.shape[0]):
                            # s=start, e=end, f=farthest, d=depth (8.8 fixed-point)
                            s, e, f, d = defects[i, 0]
                            start = tuple(cnt[s][0])
                            end = tuple(cnt[e][0])
                            far = tuple(cnt[f][0])
                            bgr = cv2.line(bgr, start, end, list(_get_viz().measurement_line_color_bgr), thickness)
                            if d > DEFECT_FIXED_POINT:
                                depth_mm = d * mm_per_px / DEFECT_FIXED_POINT
                                if use_percent_filter:
                                    keep = (SULCUS_TERTIARY_MIN_FRACTION * max_dim_mm) < depth_mm < (SULCUS_PRIMARY_MAX_FRACTION * max_dim_mm) and depth_mm > sulcus_depth_min("mm")
                                else:
                                    keep = depth_mm > sulcus_depth_min("mm")
                                if keep:
                                    if use_percent_filter:
                                        sulcus_class = classify_sulcus_depth(depth_mm, max_dim_mm)
                                    else:
                                        sulcus_class = "unclassified"
                                    marker_color = SULCUS_CLASS_COLORS[sulcus_class]
                                    depth_sets[sulcus_class].append(depth_mm)
                                    bgr = cv2.circle(bgr, far, radius_px, marker_color, -1)

        depth = flatten_depth_sets(depth_sets)
        print(f"[VTK All Hallmarks] slice {idx}: kind={slice_kind} ({slice_kind_conf:.2f}), {format_sulcus_class_summary(depth_sets)}")
        mean_depth = (sum(depth)/len(depth)) if depth else None
        total_depth.extend(depth)
        if use_percent_filter:
            per_class_cells = sulcus_export_cells(depth_sets)
            slice_class_data.append(depth_sets)
        else:
            per_class_cells = [None] * len(sulcus_export_columns(unit))
            slice_class_data.append(None)
        rows.append([idx, len(inner_filtered), area_perim_mm, inner_perim_mm, outer_perim_mm,
            len(depth),                         # n_defects
            (min(depth) if depth else None),    # min_depth_{unit}
            (max(depth) if depth else None),    # max_depth_{unit}
            mean_depth,                         # mean_depth_{unit}
            *per_class_cells,
            ])
        kernel_px_values.append(kernel_size_px)
        # Accumulate
        sum_inner_mm += inner_perim_mm
        sum_outer_mm += outer_perim_mm
        sum_area     += area_perim_mm
        # Collect raw per-slice data; cavities + tissue/hole polygons in physical
        # mm (relative to the red-cube centre, a consistent in-plane origin) for
        # the cross-slice tracker. inner_area_mm is the holes-filled baseline.
        pos = float(k) * axis_scale
        inner_area_mm = area_perim_mm  # outer-region area; holes handled by cavity correction
        raw_lateral.append((idx, pos, inner_perim_mm))
        raw_volume.append((idx, pos, inner_area_mm, area_perim_mm))
        if cavity_correction_enabled:
            center_px = get_red_rect_offset(img_rgb)
            slice_records.append(SliceRecord(
                idx=idx, position_mm=pos,
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
        section["slice_idx"] = np.full(section.n_points, idx, dtype=np.int32)
        plane["slice_idx"] = np.full(plane.n_points, idx, dtype=np.int32)
        sections_list.append(section)
        sections_list.append(plane)
    # ---- end with: plotter is fully and safely closed here ----

    # Surface-connected cavity correction (cross-slice tracking). Volume uses the
    # holes-filled area minus surface-connected cavities (enclosed voids stay
    # solid); the surface lateral gets the open-cavity wall perimeter added. GI is
    # untouched. When disabled, reproduce the previous behaviour exactly.
    if cavity_correction_enabled:
        cavity_corr = cavity_correction_tracking(
            slice_records, area_threshold_mm2=float(cavity_area_threshold_mm2))
        volume_samples = [(pos, max(0.0, inner_area - cavity_corr.area_subtract(idx)))
                          for (idx, pos, inner_area, _area_perim) in raw_volume]
        lateral_samples = [(pos, perim + cavity_corr.perim_add(idx))
                           for (idx, pos, perim) in raw_lateral]
        # Outline the surface-connected cavities in YELLOW on the saved slice PNGs
        # (classification is only known after tracking, so the slices are redrawn
        # here). Enclosed voids are left unmarked.
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
        volume_samples = [(pos, area_perim) for (_idx, pos, _inner, area_perim) in raw_volume]
        lateral_samples = [(pos, perim) for (_idx, pos, perim) in raw_lateral]

    # Totals. Surface area = Simpson lateral (∫ perimeter dh) + top & bottom caps;
    # Volume = ∫ cross-section area dh via Simpson's rule. The exact physical
    # mesh.area and mesh.volume are kept only for verification.
    mesh_area_phys, _mesh_volume_phys = _mesh_area_volume_physical(mesh, mesh_dim_scaled)
    Volume = volume_simpson(volume_samples)
    Area, inner_area_slice_sum, caps_area = total_surface_area_simpson(
        lateral_samples, volume_samples)
    print(f"[VTK All Hallmarks] Volume={Volume:.3f} {unit}³")
    print(f"[VTK All Hallmarks] Surface area: lateral+caps={Area:.3f} {unit}²")
    if cavity_correction_enabled:
        print(f"[VTK All Hallmarks] Cavity correction: "
              f"{cavity_corr.n_surface_connected} surface-connected, "
              f"{cavity_corr.n_enclosed} enclosed; "
              f"area removed={cavity_corr.total_cavity_area_mm2:.3f} {unit}², "
              f"wall perim added={cavity_corr.total_wall_perim_mm:.3f} {unit}")
    GI_total = (sum_inner_mm / sum_outer_mm) if sum_outer_mm > 0 else 0.0
    comp = compactness_3D(Volume, Area)
    # 3-D area-based GI = exact mesh surface ÷ convex-hull surface, both on the
    # physically-scaled mesh (separate from the 2-D perimeter GI).
    gi_3d, _gi3d_area, hull_area_phys = area_based_gi_3d(mesh, scale=mesh_dim_scaled)

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

        overall_n = len(total_depth)
        overall_mean = (sum(total_depth) / overall_n) if overall_n else None

        results_rows = []
        for r, dsets, png_path, kernel_px, cal_meta in zip(rows, slice_class_data, saved_pngs, kernel_px_values, calibration_metadata):
            idx, _ncont, area_u, inner_perim_u, outer_perim_u = r[:5]
            lgi = ((inner_perim_u / outer_perim_u)
                   if outer_perim_u else None)
            compact = (compactness_2D(area_u, inner_perim_u)
                       if inner_perim_u else None)
            d = dsets if isinstance(dsets, dict) else {}
            results_rows.append({
                "Section": idx,
                "Kernel px": int(kernel_px),
                "Cube length (mm)": cal_meta.get("cube_len_mm"),
                "Detected cube size (px)": cal_meta.get("detected_cube_size_px"),
                "Computed mm per px": cal_meta.get("computed_mm_per_px"),
                "Area": area_u,
                "Perimeter": inner_perim_u,
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
            "Pixel spacing": f"per-slice from cube ({unit}/pixel)",
            "Slice thickness": float(slice_thickness),
            "Filtered threshold (mm²)": float(min_contour_area),
            "Slice direction": Slice_direction,
            "Length unit": unit,
            "Perimeter method": perimeter_method,
            "Contour simplification enabled": bool(simplify_contours_for_perimeter),
            "Contour simplification epsilon": float(contour_simplify_epsilon),
            "Cavity correction": "on" if cavity_correction_enabled else "off",
        }
        if cavity_correction_enabled:
            parameters["Cavity area threshold (mm²)"] = float(cavity_area_threshold_mm2)
        totals = {
            f"Volume ({unit}^3)": round(float(Volume), 4),
            f"Surface Area ({unit}^2)": round(float(Area), 4),
            f"Area lateral surface ({unit}^2)": round(float(inner_area_slice_sum), 4),
            f"Surface area caps ({unit}^2)": round(float(caps_area), 4),
            "GI": round(float(GI_total), 4),
            "GI 3D (convex hull)": (round(float(gi_3d), 4) if gi_3d is not None else None),
            f"Convex hull area ({unit}^2)": (round(float(hull_area_phys), 4) if hull_area_phys else None),
            "Compactness": round(float(comp), 4),
            "Total sulci count": int(overall_n),
            f"Mean sulci depth ({unit})": (
                round(float(overall_mean), 4)
                if overall_mean is not None else None),
        }
        if cavity_correction_enabled:
            totals.update({
                "Number of surface-connected cavities": int(cavity_corr.n_surface_connected),
                "Number of enclosed cavities": int(cavity_corr.n_enclosed),
                f"Surface connected cavity area ({unit}^2)": round(float(cavity_corr.total_cavity_area_mm2), 4),
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
        xlsx_path = os.path.join(
            out_dir, f"Mesh_Allmarks_{Slice_direction}.xlsx")
        write_results_workbook(xlsx_path, [sheet])
        print(f"[VTK All Hallmarks] Saved Excel → {xlsx_path}")
    except Exception as ex:
        print(f"[VTK All Hallmarks] WARN: could not save Excel: {ex}")


    # Always return a 3-tuple
    return Area, Volume, GI_total, comp ,total_depth ,saved_pngs, valid_slices


def compute_vtk_lGI(
    parent,
    file_path: str,
    out_dir: str,
    min_contour_area: float = 20.0,
    kernel_size_mm: float = DEFAULT_KERNEL_SIZE_MM,
    Slice_direction: Literal["X", "Y", "Z"] = "Y",
    Physical_dim: Sequence[int] | None = None,
    unit: str = "mm",
    slice_thickness: float = 0.5,
    perimeter_method: str = DEFAULT_PERIMETER_METHOD,
    simplify_contours_for_perimeter: bool = DEFAULT_SIMPLIFY_CONTOURS_FOR_PERIMETER,
    contour_simplify_epsilon: float = DEFAULT_CONTOUR_SIMPLIFY_EPSILON):
    """Compute the gyrification index (GI) from a VTK mesh.

    Args:
        parent: Qt parent widget for message boxes.
        file_path: Path to the ``.vtk`` file.
        out_dir: Output directory.
        min_contour_area: Minimum contour area (pixels) to keep.
        kernel_size_mm: Morph-close kernel diameter, in mm.
        Slice_direction: Axis to slice along.
        Physical_dim: Real-world ``[X, Y, Z]`` dimensions for unit scaling.
        unit: Label for output units.
        slice_thickness: Distance between slices in model units.

    Returns:
        Tuple of ``(GI_total, saved_pngs, valid_slices)``.
    """
    # --- Load mesh
    
    mesh = pv.read(str(file_path))
    print(f"[VTK lGI] Loaded mesh: {mesh}")

    # --- Bounds / dims (mm)
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    mesh_dim = [x_max - x_min, y_max - y_min, z_max - z_min]
    mesh_dim_scaled = np.array(Physical_dim) / np.array(mesh_dim)

    axis_bounds = {
        "X": (x_min, x_max),
        "Y": (y_min, y_max),
        "Z": (z_min, z_max),
    }

    # --- Slice positions along Y
    if slice_thickness <= 0:
        slice_thickness = 0.05
        
    low, high = axis_bounds[Slice_direction]
    slice_positions = np.arange(low, high, slice_thickness)
        
    N = len(slice_positions)
    if N == 0:
        print(f"[VTK lGI] No slices to process (thickness too large vs. {Slice_direction} range).")
        return 0.0, [], []

    # Effective thickness to exactly span Y-extent
    axis_index = {"X": 0, "Y": 1, "Z": 2}.get(Slice_direction)
    slice_thickness_eff = mesh_dim[axis_index] / N
    slice_thickness_eff *= mesh_dim_scaled[axis_index]
    print(f"[VTK lGI] Effective slice thickness: {slice_thickness_eff} {unit}")

    # --- Outputs
    os.makedirs(out_dir, exist_ok=True)
    out_dir_slices = os.path.join(out_dir, "vtk_slices")
    os.makedirs(out_dir_slices, exist_ok=True)
    
    out_dir_origin = os.path.join(out_dir, "vtk_orgin")
    os.makedirs(out_dir_origin, exist_ok=True)
    
    print(f"[VTK lGI] Temp output dir: {out_dir}")

    saved_pngs: list[str] = []
    valid_slices: list[int] = []
    sections_list: list[pv.PolyData] = []
    rows = []
    sum_inner_mm = 0.0
    sum_outer_mm = 0.0
    axis_scale = float(mesh_dim_scaled[axis_index])
    lateral_samples: list[tuple[float, float]] = []  # (physical position, exterior perimeter)

    # --- Use a context manager so the plotter is *guaranteed* to be closed safely
    p = pv.Plotter(off_screen=True, window_size=VTK_RENDER_WINDOW_SIZE)
    p.set_background("black")
    p.parallel_projection = True
    
    pre_axis = (axis_index - 1) % 3
    next_axis = (axis_index + 1) % 3
    cube_len = max(1e-6, mesh_dim[0] / 10.0)
    scale_cube_offset = max(mesh_dim[pre_axis], mesh_dim[next_axis])
    physical_cube_len = cube_len * mesh_dim_scaled[0]
    _require_scale_calibration(
        p, Slice_direction, cube_len, physical_cube_len, mesh.center,
        float(slice_positions[0]), scale_cube_offset,
    )

    for idx, k in enumerate(slice_positions):
        # Cross-section slice
        normal, origin = slice_at(mesh, Slice_direction, k)
        section = mesh.slice(normal=normal,origin=origin)
        
        if section.n_points == 0:
            continue

        # Red cube reference (10% of X extent)
        scale_cube = make_scale_cube(Slice_direction, cube_len, mesh.center, k, scale_cube_offset)
        
        plane = pv.Plane(center=origin,
                 direction=normal,  # plane normal
                 i_size=mesh_dim[pre_axis]*1.5, j_size=mesh_dim[next_axis]*1.5,  # side lengths
                 i_resolution=1, j_resolution=1)        # Render: section + scale cube
        p.clear()
        p.add_mesh(section, color="#ffffff", opacity=1, lighting=False)
        p.add_mesh(scale_cube, color="red", lighting=False)
        prepare_orthographic_slice_render(p, _view_fn(p, Slice_direction))

        # Screenshot (array for processing, file for debugging)
        img_rgb = p.screenshot(return_img=True, filename=os.path.join(out_dir_origin, f"image_{idx:03d}.png"))

        # Compute mm/px scale from the red cube
        mm_per_px, scale_meta, red_rect = calc_scale_with_metadata(img_rgb, physical_cube_len)
        if mm_per_px is None:
            print(f"[VTK Scale] Scale cube was not detected or failed validation. {scale_meta.get('calibration_status')}. Measurements for this slice were skipped.")
            continue
        kernel_size_px = _to_kernel_px(kernel_size_mm, mm_per_px)

        # Prepare masks / contours (pixel space)
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h_img, w_img = bgr.shape[:2]
        thickness, _, _ = image_annotation_style(h_img, w_img, style="thin")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # Binary for contours
        bw = threshold_binary(gray, BINARY_THRESHOLD_VTK, invert=False)
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # Inner contours: exclude red ref + area filter
        inner_candidates = contours_exclude(contours, red_rect, bw.shape)
        inner_filtered = [c for c in inner_candidates if cv2.contourArea(c) * (mm_per_px ** 2) > float(min_contour_area)]
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
        lateral_samples.append((float(k) * axis_scale, inner_perim_mm))
        GI_slice = (inner_perim_mm / outer_perim_mm) if outer_perim_mm > 0 else 0.0

        # Save annotated slice
        slice_path = os.path.join(out_dir_slices, f"slice_{idx:03d}.png")
        cv2.imwrite(slice_path, bgr)
        saved_pngs.append(slice_path)
        valid_slices.append(idx)
        rows.append([idx, len(inner_filtered), kernel_size_px, inner_perim_mm, outer_perim_mm, GI_slice, scale_meta])
        section["slice_idx"] = np.full(section.n_points, idx, dtype=np.int32)
        plane["slice_idx"] = np.full(plane.n_points, idx, dtype=np.int32)
        sections_list.append(section)
        sections_list.append(plane)
      
    # ---- end with: plotter is fully and safely closed here ----

    GI_total = (sum_inner_mm / sum_outer_mm) if sum_outer_mm > 0 else 0.0

    # Lateral surface area = ∫ exterior perimeter dh via Simpson's rule.
    inner_area_slice_sum = lateral_area_simpson(lateral_samples)
    # 3-D area-based GI = exact mesh surface ÷ convex-hull surface (scaled mesh).
    gi_3d, _gi3d_area, hull_area_phys = area_based_gi_3d(mesh, scale=mesh_dim_scaled)

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
            "Length unit": unit,
        }
        totals = {
            "GI": round(float(GI_total), 4),
            "GI 3D (convex hull)": (round(float(gi_3d), 4) if gi_3d is not None else None),
            f"Convex hull area ({unit}^2)": (round(float(hull_area_phys), 4) if hull_area_phys else None),
            f"Area lateral surface ({unit}^2)": round(float(inner_area_slice_sum), 4),
        }
        sheet = build_measurement_sheet(
            file_path, "LGI", results_rows, parameters, totals,
            extra_columns=("Contours", "Kernel px", "Cube length (mm)",
                           "Detected cube size (px)", "Computed mm per px",
                           "Closed-envelope perimeter"),
            totals_notes={"GI 3D (convex hull)": gi_3d_note(gi_3d)})
        xlsx_path = os.path.join(out_dir, f"Mesh_lGI_{Slice_direction}.xlsx")
        write_results_workbook(xlsx_path, [sheet])
        print(f"[VTK lGI] Saved Excel → {xlsx_path}")
    except Exception as ex:
        print(f"[VTK lGI] WARN: could not save Excel: {ex}")


    # Always return a 3-tuple
    return  GI_total, saved_pngs, valid_slices


def compute_vtk_volume(
    parent,
    file_path: str,
    out_dir: str,
    min_contour_area: float = 0.0,
    Slice_direction: Literal["X", "Y", "Z"] = "Y",
    Physical_dim: Sequence[int] | None = None,
    unit: str = "mm",
    slice_thickness: float = 0.5,
    fill_cross_section: bool = DEFAULT_FILL_CROSS_SECTION,
    cavity_correction_enabled: bool = DEFAULT_CAVITY_CORRECTION_ENABLED,
    cavity_area_threshold_mm2: float = DEFAULT_CAVITY_AREA_THRESHOLD_MM2):
    """Compute volume from a VTK mesh by integrating slice cross-section areas.

    Args:
        parent: Qt parent widget for message boxes.
        file_path: Path to the ``.vtk`` file.
        out_dir: Output directory.
        min_contour_area: Minimum contour area (pixels) to keep.
        Slice_direction: Axis to slice along.
        Physical_dim: Real-world ``[X, Y, Z]`` dimensions for unit scaling.
        unit: Label for output units.
        slice_thickness: Distance between slices in model units.

    Returns:
        Tuple of ``(volume, saved_pngs, valid_slices)``.
    """
    # --- Load mesh
    
        
    mesh = pv.read(str(file_path))
    print(f"[VTK Volume] Loaded mesh: {mesh}")
    # --- Bounds / dims (mm)
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    mesh_dim = [x_max - x_min, y_max - y_min, z_max - z_min]
    mesh_dim_scaled = np.array(Physical_dim) / np.array(mesh_dim)
    mesh_center = mesh.center

    axis_bounds = {
        "X": (x_min, x_max),
        "Y": (y_min, y_max),
        "Z": (z_min, z_max),
    }

    # --- Slice positions along Y
    if slice_thickness <= 0:
        slice_thickness = 0.05
        
    low, high = axis_bounds[Slice_direction]
    slice_positions = np.arange(low, high, slice_thickness)
        
    N = len(slice_positions)
    if N == 0:
        print(f"[VTK Volume] No slices to process (thickness too large vs. {Slice_direction} range).")
        return 0.0, [], []

    # Effective thickness to exactly span Y-extent
    axis_index = {"X": 0, "Y": 1, "Z": 2}.get(Slice_direction)
    slice_thickness_eff = mesh_dim[axis_index] / N
    slice_thickness_eff *= mesh_dim_scaled[axis_index]
    print(f"[VTK Volume] Effective slice thickness: {slice_thickness_eff} {unit}")

    # --- Outputs
    os.makedirs(out_dir, exist_ok=True)
    out_dir_slices = os.path.join(out_dir, "vtk_slices")
    os.makedirs(out_dir_slices, exist_ok=True)
    
    out_dir_origin = os.path.join(out_dir, "vtk_orgin")
    os.makedirs(out_dir_origin, exist_ok=True)
    
    print(f"[VTK Volume] Temp output dir: {out_dir}")

    saved_pngs: list[str] = []
    valid_slices: list[int] = []
    sections_list: list[pv.PolyData] = []
    rows = []
    sum_area = 0.0
    axis_scale = float(mesh_dim_scaled[axis_index])
    # Raw per-slice data for the surface-connected cavity correction (applied
    # after the loop). volume_samples is built from raw_volume post-correction.
    raw_volume: list[tuple[int, float, float]] = []   # (idx, pos, area_mm)
    slice_records: list[SliceRecord] = []
    slice_png_by_idx: dict[int, tuple[str, int]] = {}  # idx -> (png path, line thickness)

    # --- Use a context manager so the plotter is *guaranteed* to be closed safely
    p = pv.Plotter(off_screen=True, window_size=VTK_RENDER_WINDOW_SIZE)
    p.set_background("black")
    p.parallel_projection = True

    pre_axis = (axis_index - 1) % 3
    next_axis = (axis_index + 1) % 3
    cube_len = max(1e-6, mesh_dim[0] / 10.0)
    scale_cube_offset = max(mesh_dim[pre_axis], mesh_dim[next_axis])
    physical_cube_len = cube_len * mesh_dim_scaled[0]
    _require_scale_calibration(
        p, Slice_direction, cube_len, physical_cube_len, mesh.center,
        float(slice_positions[0]), scale_cube_offset,
    )

    for idx, k in enumerate(slice_positions):
        # Cross-section slice
        normal, origin = slice_at(mesh, Slice_direction, k)
        section = mesh.slice(normal=normal,origin=origin)
        
        if section.n_points == 0:
            continue

        # Red cube reference (10% of X extent)
        scale_cube = make_scale_cube(Slice_direction, cube_len, mesh.center, k, scale_cube_offset)
        
        plane = pv.Plane(center=origin,
                 direction=normal,  # plane normal
                 i_size=mesh_dim[pre_axis]*1.5, j_size=mesh_dim[next_axis]*1.5,  # side lengths
                 i_resolution=1, j_resolution=1)
        # Render: section + scale cube
        p.clear()
        p.add_mesh(fill_section_polydata(section) if fill_cross_section else section, color="#B4B4B4", opacity=1, lighting=False)
        p.add_mesh(scale_cube, color="red", lighting=False)
        prepare_orthographic_slice_render(p, _view_fn(p, Slice_direction))

        # Screenshot (array for processing, file for debugging)
        img_rgb = p.screenshot(return_img=True, filename=os.path.join(out_dir_origin, f"image_{idx:03d}.png"))

        # Compute mm/px scale from the red cube
        mm_per_px, scale_meta, red_rect = calc_scale_with_metadata(img_rgb, physical_cube_len)
        if mm_per_px is None:
            print(f"[VTK Scale] Scale cube was not detected or failed validation. {scale_meta.get('calibration_status')}. Measurements for this slice were skipped.")
            continue


        # Prepare masks / contours (pixel space)
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h_img, w_img = bgr.shape[:2]
        thickness, _, _ = image_annotation_style(h_img, w_img, style="thin")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # Binary for contours
        bw = threshold_binary(gray, BINARY_THRESHOLD_VTK, invert=False)
        contours, hierarchy = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # internal_filtered (holes) is retained for the surface-connected cavity
        # correction; hole accounting is owned by that correction.
        inner_filtered, internal_filtered = split_inner_and_internal_contours(
            contours, hierarchy, red_rect, bw.shape, float(min_contour_area) / (mm_per_px ** 2),
        )
        cv2.drawContours(bgr, inner_filtered, -1, tuple(_get_viz().contour_inner_color_bgr), thickness)

        # Cross-section area (holes-filled outer region; surface-connected
        # cavities are subtracted after the loop, enclosed voids stay solid).
        area_perim_px = sum(cv2.contourArea(c) for c in inner_filtered)
        area_perim_mm  = area_perim_px * (mm_per_px ** 2)
        sum_area      += area_perim_mm
        pos = float(k) * axis_scale
        raw_volume.append((idx, pos, area_perim_mm))
        if cavity_correction_enabled:
            center_px = get_red_rect_offset(img_rgb)
            slice_records.append(SliceRecord(
                idx=idx, position_mm=pos,
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
        rows.append([idx, len(inner_filtered), area_perim_mm, scale_meta])
        section["slice_idx"] = np.full(section.n_points, idx, dtype=np.int32)
        plane["slice_idx"] = np.full(plane.n_points, idx, dtype=np.int32)
        sections_list.append(section)
        sections_list.append(plane)
      
    # ---- end with: plotter is fully and safely closed here ----

    # Surface-connected cavity correction: subtract open-cavity area from the
    # volume integral (enclosed voids stay solid). Yellow-outline the open
    # cavities on the saved slice PNGs once classification is known.
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

    # Totals — Volume = ∫ cross-section area dh via Simpson's rule; exact
    # mesh.volume kept only for verification.
    _mesh_area_phys, _mesh_volume_phys = _mesh_area_volume_physical(mesh, mesh_dim_scaled)
    Volume = volume_simpson(volume_samples)
    print(f"[VTK Volume] Volume={Volume:.3f} {unit}³")
    if cavity_correction_enabled:
        print(f"[VTK Volume] Cavity correction: "
              f"{cavity_corr.n_surface_connected} surface-connected, "
              f"{cavity_corr.n_enclosed} enclosed; "
              f"area removed={cavity_corr.total_cavity_area_mm2:.3f} {unit}²")
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
            "Length unit": unit,
            "Cavity correction": "on" if cavity_correction_enabled else "off",
        }
        if cavity_correction_enabled:
            parameters["Cavity area threshold (mm²)"] = float(cavity_area_threshold_mm2)
        totals = {f"Volume ({unit}^3)": round(float(Volume), 4)}
        if cavity_correction_enabled:
            totals.update({
                "Number of surface-connected cavities": int(cavity_corr.n_surface_connected),
                "Number of enclosed cavities": int(cavity_corr.n_enclosed),
                f"Cavity area removed ({unit}^2)": round(float(cavity_corr.total_cavity_area_mm2), 4),
            })
        sheet = build_measurement_sheet(
            file_path, "Volume", results_rows, parameters, totals,
            extra_columns=("Contours", "Cube length (mm)",
                           "Detected cube size (px)", "Computed mm per px"))
        xlsx_path = os.path.join(out_dir, f"Mesh_Volume_{Slice_direction}.xlsx")
        write_results_workbook(xlsx_path, [sheet])
        print(f"[VTK Volume] Saved Excel → {xlsx_path}")
    except Exception as ex:
        print(f"[VTK Volume] WARN: could not save Excel: {ex}")

    return Volume, saved_pngs, valid_slices

def compute_vtk_area(
    parent,
    file_path: str,
    out_dir: str,
    min_contour_area: float = 20.0,
    Slice_direction: Literal["X", "Y", "Z"] = "Y",
    Physical_dim: Sequence[int] | None = None,
    unit: str = "mm",
    slice_thickness: float = 0.5,
    perimeter_method: str = DEFAULT_PERIMETER_METHOD,
    simplify_contours_for_perimeter: bool = DEFAULT_SIMPLIFY_CONTOURS_FOR_PERIMETER,
    contour_simplify_epsilon: float = DEFAULT_CONTOUR_SIMPLIFY_EPSILON,
    fill_cross_section: bool = DEFAULT_FILL_CROSS_SECTION,
    cavity_correction_enabled: bool = DEFAULT_CAVITY_CORRECTION_ENABLED,
    cavity_area_threshold_mm2: float = DEFAULT_CAVITY_AREA_THRESHOLD_MM2):
    """Compute surface area from a VTK mesh by summing slice perimeters.

    Args:
        parent: Qt parent widget for message boxes.
        file_path: Path to the ``.vtk`` file.
        out_dir: Output directory.
        min_contour_area: Minimum contour area (pixels) to keep.
        Slice_direction: Axis to slice along.
        Physical_dim: Real-world ``[X, Y, Z]`` dimensions for unit scaling.
        unit: Label for output units.
        slice_thickness: Distance between slices in model units.

    Returns:
        Tuple of ``(area, saved_pngs, valid_slices)``.
    """
    # --- Load mesh
    
        
    mesh = pv.read(str(file_path))
    print(f"[VTK Area] Loaded mesh: {mesh}")

    # --- Bounds / dims (mm)
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    mesh_dim = [x_max - x_min, y_max - y_min, z_max - z_min]
    mesh_dim_scaled = np.array(Physical_dim) / np.array(mesh_dim)

    axis_bounds = {
        "X": (x_min, x_max),
        "Y": (y_min, y_max),
        "Z": (z_min, z_max),
    }

    # --- Slice positions along Y
    if slice_thickness <= 0:
        slice_thickness = 0.05
        
    low, high = axis_bounds[Slice_direction]
    slice_positions = np.arange(low, high, slice_thickness)
        
    N = len(slice_positions)
    if N == 0:
        print(f"[VTK Area] No slices to process (thickness too large vs. {Slice_direction} range).")
        return 0.0, [], []

    # Effective thickness to exactly span Y-extent
    axis_index = {"X": 0, "Y": 1, "Z": 2}.get(Slice_direction)
    slice_thickness_eff = mesh_dim[axis_index] / N
    slice_thickness_eff *= mesh_dim_scaled[axis_index]
    print(f"[VTK Area] Effective slice thickness: {slice_thickness_eff} {unit}")

    # --- Outputs
    os.makedirs(out_dir, exist_ok=True)
    out_dir_slices = os.path.join(out_dir, "vtk_slices")
    os.makedirs(out_dir_slices, exist_ok=True)
    
    out_dir_origin = os.path.join(out_dir, "vtk_orgin")
    os.makedirs(out_dir_origin, exist_ok=True)
    
    print(f"[VTK Area] Temp output dir: {out_dir}")

    saved_pngs: list[str] = []
    valid_slices: list[int] = []
    sections_list: list[pv.PolyData] = []
    rows = []
    sum_inner_mm = 0.0
    axis_scale = float(mesh_dim_scaled[axis_index])
    # Raw per-slice data for the surface-connected cavity correction (applied
    # after the loop): the open-cavity wall perimeter is added to the lateral
    # surface and the open-cavity area is subtracted from the cap area.
    raw_lateral: list[tuple[int, float, float]] = []   # (idx, pos, inner_perim_mm)
    raw_volume: list[tuple[int, float, float]] = []    # (idx, pos, inner_area_mm)
    slice_records: list[SliceRecord] = []
    slice_png_by_idx: dict[int, tuple[str, int]] = {}  # idx -> (png path, line thickness)

    # --- Use a context manager so the plotter is *guaranteed* to be closed safely
    p = pv.Plotter(off_screen=True, window_size=VTK_RENDER_WINDOW_SIZE)
    p.set_background("black")
    p.parallel_projection = True

    pre_axis = (axis_index - 1) % 3
    next_axis = (axis_index + 1) % 3
    cube_len = max(1e-6, mesh_dim[0] / 10.0)
    scale_cube_offset = max(mesh_dim[pre_axis], mesh_dim[next_axis])
    physical_cube_len = cube_len * mesh_dim_scaled[0]
    _require_scale_calibration(
        p, Slice_direction, cube_len, physical_cube_len, mesh.center,
        float(slice_positions[0]), scale_cube_offset,
    )

    for idx, k in enumerate(slice_positions):
        # Cross-section slice
        normal, origin = slice_at(mesh, Slice_direction, k)
        section = mesh.slice(normal=normal,origin=origin)
        
        if section.n_points == 0:
            continue

        # Red cube reference (10% of X extent)
        scale_cube = make_scale_cube(Slice_direction, cube_len, mesh.center, k, scale_cube_offset)
        
        plane = pv.Plane(center=origin,
                 direction=normal,  # plane normal
                 i_size=mesh_dim[pre_axis]*1.5, j_size=mesh_dim[next_axis]*1.5,  # side lengths
                 i_resolution=1, j_resolution=1)        # Render: section + scale cube
        p.clear()
        p.add_mesh(fill_section_polydata(section) if fill_cross_section else section, color="#B4B4B4", opacity=1, lighting=False)
        p.add_mesh(scale_cube, color="red", lighting=False)
        prepare_orthographic_slice_render(p, _view_fn(p, Slice_direction))

        # Screenshot (array for processing, file for debugging)
        img_rgb = p.screenshot(return_img=True, filename=os.path.join(out_dir_origin, f"image_{idx:03d}.png"))

        # Compute mm/px scale from the red cube
        mm_per_px, scale_meta, red_rect = calc_scale_with_metadata(img_rgb, physical_cube_len)
        if mm_per_px is None:
            print(f"[VTK Scale] Scale cube was not detected or failed validation. {scale_meta.get('calibration_status')}. Measurements for this slice were skipped.")
            continue

        # Prepare masks / contours (pixel space)
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h_img, w_img = bgr.shape[:2]
        thickness, _, _ = image_annotation_style(h_img, w_img, style="thin")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # Binary for contours. RETR_CCOMP so holes are detected as internal
        # contours for the surface-connected cavity correction; the top-level
        # contours match the previous RETR_EXTERNAL result so the perimeter is
        # unchanged.
        bw = threshold_binary(gray, BINARY_THRESHOLD_VTK, invert=False)
        contours, hierarchy = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        inner_filtered, internal_filtered = split_inner_and_internal_contours(
            contours, hierarchy, red_rect, bw.shape, float(min_contour_area) / (mm_per_px ** 2),
        )
        cv2.drawContours(bgr, inner_filtered, -1, tuple(_get_viz().contour_inner_color_bgr), thickness)

        # Perimeters (physical `unit`) — Crofton on the filled mask when selected.
        inner_perim_px = sum(cv2.arcLength(c, True) for c in inner_filtered)
        if perimeter_method == "crofton":
            inner_mask = np.zeros_like(bw)
            cv2.drawContours(inner_mask, inner_filtered, -1, 255, thickness=cv2.FILLED)
            inner_perim_mm = mask_perimeter_mm(
                inner_mask, mm_per_px, mm_per_px, method="crofton",
                simplify=simplify_contours_for_perimeter, epsilon=contour_simplify_epsilon)
        else:
            inner_perim_mm = inner_perim_px * mm_per_px
        sum_inner_mm += inner_perim_mm
        inner_area_phys = sum(cv2.contourArea(c) for c in inner_filtered) * (mm_per_px ** 2)
        pos = float(k) * axis_scale
        raw_lateral.append((idx, pos, inner_perim_mm))
        raw_volume.append((idx, pos, inner_area_phys))
        if cavity_correction_enabled:
            center_px = get_red_rect_offset(img_rgb)
            slice_records.append(SliceRecord(
                idx=idx, position_mm=pos,
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
        rows.append([idx, len(inner_filtered), inner_perim_mm, scale_meta])
        section["slice_idx"] = np.full(section.n_points, idx, dtype=np.int32)
        plane["slice_idx"] = np.full(plane.n_points, idx, dtype=np.int32)
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
    # caps (physical unit²). The exact physical mesh.area is kept only for
    # verification (matches the All-hallmarks workflow).
    _mesh_area_phys, _volume = _mesh_area_volume_physical(mesh, mesh_dim_scaled)
    Area, lateral_phys, caps_phys = total_surface_area_simpson(
        lateral_samples, volume_samples)
    print(f"[VTK Area] Surface area: lateral+caps={Area:.3f} {unit}²")
    if cavity_correction_enabled:
        print(f"[VTK Area] Cavity correction: "
              f"{cavity_corr.n_surface_connected} surface-connected, "
              f"{cavity_corr.n_enclosed} enclosed; "
              f"wall perim added={cavity_corr.total_wall_perim_mm:.3f} {unit}")
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
            "Length unit": unit,
            "Perimeter method": perimeter_method,
            "Surface area method": "lateral + caps",
            "Cavity correction": "on" if cavity_correction_enabled else "off",
        }
        if cavity_correction_enabled:
            parameters["Cavity area threshold (mm²)"] = float(cavity_area_threshold_mm2)
        totals = {
            f"Surface Area ({unit}^2)": round(float(Area), 4),
            f"Surface area lateral ({unit}^2)": round(float(lateral_phys), 4),
            f"Surface area caps ({unit}^2)": round(float(caps_phys), 4),
        }
        if cavity_correction_enabled:
            totals.update({
                "Number of surface-connected cavities": int(cavity_corr.n_surface_connected),
                "Number of enclosed cavities": int(cavity_corr.n_enclosed),
                f"Cavity wall perimeter added ({unit})": round(float(cavity_corr.total_wall_perim_mm), 4),
            })
        sheet = build_measurement_sheet(
            file_path, "Area", results_rows, parameters, totals,
            extra_columns=("Contours", "Cube length (mm)",
                           "Detected cube size (px)", "Computed mm per px"))
        xlsx_path = os.path.join(out_dir, f"Mesh_Area_{Slice_direction}.xlsx")
        write_results_workbook(xlsx_path, [sheet])
        print(f"[VTK Area] Saved Excel → {xlsx_path}")
    except Exception as ex:
        print(f"[VTK Area] WARN: could not save Excel: {ex}")


    # Always return a 3-tuple
    return Area, saved_pngs, valid_slices



def compute_vtk_sulci_depth(
    parent,
    file_path: str,
    out_dir: str,
    min_contour_area: float = 20.0,
    Slice_direction: Literal["X", "Y", "Z"] = "Y",
    Physical_dim: Sequence[int] | None = None,
    unit: str = "mm",
    slice_thickness: float = 0.5):
    """Compute sulci depths from convexity defects across VTK mesh slices.

    Args:
        parent: Qt parent widget for message boxes.
        file_path: Path to the ``.vtk`` file.
        out_dir: Output directory.
        min_contour_area: Minimum contour area (pixels) to keep.
        Slice_direction: Axis to slice along.
        Physical_dim: Real-world ``[X, Y, Z]`` dimensions for unit scaling.
        unit: Label for output units.
        slice_thickness: Distance between slices in model units.

    Returns:
        Tuple of ``(depths, saved_pngs, valid_slices)``.
    """
    # --- Load mesh
    
        
    mesh = pv.read(str(file_path))
    print(f"[VTK Sulci depth] Loaded mesh: {mesh}")

    # --- Bounds / dims (mm)
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    mesh_dim = [x_max - x_min, y_max - y_min, z_max - z_min]
    mesh_dim_scaled = np.array(Physical_dim) / np.array(mesh_dim)

    axis_bounds = {
        "X": (x_min, x_max),
        "Y": (y_min, y_max),
        "Z": (z_min, z_max),
    }

    # --- Slice positions along Y
    if slice_thickness <= 0:
        slice_thickness = 0.05
        
    low, high = axis_bounds[Slice_direction]
    slice_positions = np.arange(low, high, slice_thickness)
        
    N = len(slice_positions)
    if N == 0:
        print(f"[VTK Sulci depth] No slices to process (thickness too large vs. {Slice_direction} range).")
        return 0.0, [], []

    # Effective thickness to exactly span Y-extent
    axis_index = {"X": 0, "Y": 1, "Z": 2}.get(Slice_direction)
    slice_thickness_eff = mesh_dim[axis_index] / N
    slice_thickness_eff *= mesh_dim_scaled[axis_index]
    print(f"[VTK Sulci depth] Effective slice thickness: {slice_thickness_eff} {unit}")

    # --- Outputs
    os.makedirs(out_dir, exist_ok=True)
    out_dir_slices = os.path.join(out_dir, "vtk_slices")
    os.makedirs(out_dir_slices, exist_ok=True)
    
    out_dir_origin = os.path.join(out_dir, "vtk_orgin")
    os.makedirs(out_dir_origin, exist_ok=True)
    
    print(f"[VTK Sulci depth] Temp output dir: {out_dir}")

    saved_pngs: list[str] = []
    valid_slices: list[int] = []
    sections_list: list[pv.PolyData] = []
    rows = []
    total_depth = []
    slice_class_data: list = []

    # --- Use a context manager so the plotter is *guaranteed* to be closed safely
    p = pv.Plotter(off_screen=True, window_size=VTK_RENDER_WINDOW_SIZE)
    p.set_background("black")
    p.parallel_projection = True

    pre_axis = (axis_index - 1) % 3
    next_axis = (axis_index + 1) % 3
    cube_len = max(1e-6, mesh_dim[0] / 10.0)
    scale_cube_offset = max(mesh_dim[pre_axis], mesh_dim[next_axis])
    physical_cube_len = cube_len * mesh_dim_scaled[0]
    _require_scale_calibration(
        p, Slice_direction, cube_len, physical_cube_len, mesh.center,
        float(slice_positions[0]), scale_cube_offset,
    )

    for idx, k in enumerate(slice_positions):
        # Cross-section slice
        normal, origin = slice_at(mesh, Slice_direction, k)
        section = mesh.slice(normal=normal,origin=origin)

        if section.n_points == 0:
            continue

        # Red cube reference (10% of X extent)
        scale_cube = make_scale_cube(Slice_direction, cube_len, mesh.center, k, scale_cube_offset)

        plane = pv.Plane(center=origin,
                 direction=normal,  # plane normal
                 i_size=mesh_dim[pre_axis]*1.5, j_size=mesh_dim[next_axis]*1.5,  # side lengths
                 i_resolution=1, j_resolution=1)        # Render: section + scale cube
        p.clear()
        p.add_mesh(section, color="#ffffff", opacity=1, lighting=False)
        p.add_mesh(scale_cube, color="red", lighting=False)
        prepare_orthographic_slice_render(p, _view_fn(p, Slice_direction))

        # Screenshot (array for processing, file for debugging)
        img_rgb = p.screenshot(return_img=True, filename=os.path.join(out_dir_origin, f"image_{idx:03d}.png"))

        # Compute mm/px scale from the red cube
        mm_per_px, scale_meta, red_rect = calc_scale_with_metadata(img_rgb, physical_cube_len)
        if mm_per_px is None:
            print(f"[VTK Scale] Scale cube was not detected or failed validation. {scale_meta.get('calibration_status')}. Measurements for this slice were skipped.")
            continue

        # Prepare masks / contours (pixel space)
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h_img, w_img = bgr.shape[:2]
        thickness, _, radius_px = image_annotation_style(h_img, w_img, style="thin")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # Binary for contours
        bw = threshold_binary(gray, BINARY_THRESHOLD_VTK, invert=False)
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # Inner contours: exclude red ref + area filter
        inner_candidates = contours_exclude(contours, red_rect, bw.shape)
        inner_filtered = [c for c in inner_candidates if cv2.contourArea(c) * (mm_per_px ** 2) > float(min_contour_area)]
        cv2.drawContours(bgr, inner_filtered, -1, tuple(_get_viz().contour_inner_color_bgr), thickness)

        # Classify the rendered slice and gate the percent filter on it.
        slice_kind, slice_kind_conf = classify_slice_kind(bgr)
        use_percent_filter = slice_kind != "not_full_slice" and slice_kind_conf >= 0.7
        # Sulci classification: bin each kept defect by its depth as a
        # fraction of the brain's longest physical extent (mm).
        max_dim_mm = float(max(Physical_dim)) if Physical_dim else float(max(mesh_dim))
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
                                    keep = (SULCUS_TERTIARY_MIN_FRACTION * max_dim_mm) < depth_mm < (SULCUS_PRIMARY_MAX_FRACTION * max_dim_mm) and depth_mm > sulcus_depth_min("mm")
                                else:
                                    keep = depth_mm > sulcus_depth_min("mm")
                                if keep:
                                    if use_percent_filter:
                                        sulcus_class = classify_sulcus_depth(depth_mm, max_dim_mm)
                                    else:
                                        sulcus_class = "unclassified"
                                    marker_color = SULCUS_CLASS_COLORS[sulcus_class]
                                    depth_sets[sulcus_class].append(depth_mm)
                                    bgr = cv2.circle(bgr, far, radius_px, marker_color, -1)

        depth = flatten_depth_sets(depth_sets)
        print(f"[VTK Sulci depth] slice {idx}: kind={slice_kind} ({slice_kind_conf:.2f}), {format_sulcus_class_summary(depth_sets)}")
        mean_depth = (sum(depth)/len(depth)) if depth else None
        total_depth.extend(depth)
        if use_percent_filter:
            per_class_cells = sulcus_export_cells(depth_sets)
            slice_class_data.append(depth_sets)
        else:
            per_class_cells = [None] * len(sulcus_export_columns(unit))
            slice_class_data.append(None)
        rows.append([idx, len(inner_filtered),
            len(depth),                         # n_defects
            (min(depth) if depth else None),    # min_depth_{unit}
            (max(depth) if depth else None),    # max_depth_{unit}
            mean_depth,                         # mean_depth_{unit}
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
                "Contours": r[1],
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
            "Length unit": unit,
        }
        totals = {
            "Total sulci count": int(overall_n),
            f"Min sulci depth ({unit})": (round(float(min(total_depth)), 4) if total_depth else None),
            f"Max sulci depth ({unit})": (round(float(max(total_depth)), 4) if total_depth else None),
            f"Mean sulci depth ({unit})": (round(float(overall_mean), 4) if overall_mean is not None else None),
        }
        sheet = build_measurement_sheet(
            file_path, "Sulci depth", results_rows, parameters, totals,
            extra_columns=("Contours", "Cube length (mm)",
                           "Detected cube size (px)", "Computed mm per px"))
        xlsx_path = os.path.join(out_dir, f"Mesh_Sulci_depth_{Slice_direction}.xlsx")
        write_results_workbook(xlsx_path, [sheet])
        print(f"[VTK Sulci depth] Saved Excel → {xlsx_path}")
    except Exception as ex:
        print(f"[VTK Sulci depth] WARN: could not save Excel: {ex}")


    # Always return a 3-tuple
    return total_depth ,saved_pngs, valid_slices

def compute_compactness_vtk(parent,
    file_path: str,
    out_dir: str,
    min_contour_area: float = 20.0,
    Slice_direction: Literal["X", "Y", "Z"] = "Y",
    Physical_dim: Sequence[int] | None = None,
    unit: str = "mm",
    slice_thickness: float = 0.5,
    cavity_correction_enabled: bool = DEFAULT_CAVITY_CORRECTION_ENABLED,
    cavity_area_threshold_mm2: float = DEFAULT_CAVITY_AREA_THRESHOLD_MM2):

    # --- Load mesh

    mesh = pv.read(str(file_path))
    print(f"[VTK Compactness] Loaded mesh: {mesh}")

    # --- Bounds / dims (mm)
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    mesh_dim = [x_max - x_min, y_max - y_min, z_max - z_min]
    mesh_dim_scaled = np.array(Physical_dim) / np.array(mesh_dim)

    axis_bounds = {
        "X": (x_min, x_max),
        "Y": (y_min, y_max),
        "Z": (z_min, z_max),
    }

    # --- Slice positions along the chosen axis
    if slice_thickness <= 0:
        slice_thickness = 0.05

    low, high = axis_bounds[Slice_direction]
    slice_positions = np.arange(low, high, slice_thickness)

    N = len(slice_positions)
    if N == 0:
        print(f"[VTK Compactness] No slices to process (thickness too large vs. {Slice_direction} range).")
        return 0.0, [], []

    # Effective thickness to exactly span the axis extent, scaled to physical units
    axis_index = {"X": 0, "Y": 1, "Z": 2}.get(Slice_direction)
    slice_thickness_eff = mesh_dim[axis_index] / N
    slice_thickness_eff *= mesh_dim_scaled[axis_index]
    print(f"[VTK Compactness] Effective slice thickness: {slice_thickness_eff} {unit}")

    # --- Outputs
    os.makedirs(out_dir, exist_ok=True)
    out_dir_slices = os.path.join(out_dir, "vtk_slices")
    os.makedirs(out_dir_slices, exist_ok=True)

    out_dir_origin = os.path.join(out_dir, "vtk_orgin")
    os.makedirs(out_dir_origin, exist_ok=True)

    print(f"[VTK Compactness] Temp output dir: {out_dir}")

    saved_pngs: list[str] = []
    valid_slices: list[int] = []
    sections_list: list[pv.PolyData] = []
    rows = []
    sum_inner_mm = 0.0
    sum_area_mm2 = 0.0
    axis_scale = float(mesh_dim_scaled[axis_index])
    # Raw per-slice data for the surface-connected cavity correction (applied
    # after the loop) so compactness mirrors the Area/Volume tools.
    raw_lateral: list[tuple[int, float, float]] = []   # (idx, pos, inner_perim_mm)
    raw_volume: list[tuple[int, float, float]] = []    # (idx, pos, inner_area_mm)
    slice_records: list[SliceRecord] = []
    slice_png_by_idx: dict[int, tuple[str, int]] = {}  # idx -> (png path, line thickness)

    # --- Use a context manager so the plotter is *guaranteed* to be closed safely
    p = pv.Plotter(off_screen=True, window_size=VTK_RENDER_WINDOW_SIZE)
    p.set_background("black")
    p.parallel_projection = True

    pre_axis = (axis_index - 1) % 3
    next_axis = (axis_index + 1) % 3
    cube_len = max(1e-6, mesh_dim[0] / 10.0)
    scale_cube_offset = max(mesh_dim[pre_axis], mesh_dim[next_axis])
    physical_cube_len = cube_len * mesh_dim_scaled[0]
    _require_scale_calibration(
        p, Slice_direction, cube_len, physical_cube_len, mesh.center,
        float(slice_positions[0]), scale_cube_offset,
    )

    for idx, k in enumerate(slice_positions):
        # Cross-section slice
        normal, origin = slice_at(mesh, Slice_direction, k)
        section = mesh.slice(normal=normal,origin=origin)

        if section.n_points == 0:
            continue

        # Red cube reference (10% of X extent)
        scale_cube = make_scale_cube(Slice_direction, cube_len, mesh.center, k, scale_cube_offset)

        plane = pv.Plane(center=origin,
                 direction=normal,  # plane normal
                 i_size=mesh_dim[pre_axis]*1.5, j_size=mesh_dim[next_axis]*1.5,  # side lengths
                 i_resolution=1, j_resolution=1)        # Render: section + scale cube
        p.clear()
        p.add_mesh(section, color="#ffffff", opacity=1, lighting=False)
        p.add_mesh(scale_cube, color="red", lighting=False)
        prepare_orthographic_slice_render(p, _view_fn(p, Slice_direction))

        # Screenshot (array for processing, file for debugging)
        img_rgb = p.screenshot(return_img=True, filename=os.path.join(out_dir_origin, f"image_{idx:03d}.png"))

        # Compute mm/px scale from the red cube
        mm_per_px, scale_meta, red_rect = calc_scale_with_metadata(img_rgb, physical_cube_len)
        if mm_per_px is None:
            print(f"[VTK Scale] Scale cube was not detected or failed validation. {scale_meta.get('calibration_status')}. Measurements for this slice were skipped.")
            continue

        # Prepare masks / contours (pixel space)
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h_img, w_img = bgr.shape[:2]
        thickness, _, _ = image_annotation_style(h_img, w_img, style="thin")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # Binary for contours
        bw = threshold_binary(gray, BINARY_THRESHOLD_VTK, invert=False)
        contours, hierarchy = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        inner_filtered, internal_filtered = split_inner_and_internal_contours(
            contours, hierarchy, red_rect, bw.shape, float(min_contour_area) / (mm_per_px ** 2),
        )
        cv2.drawContours(bgr, inner_filtered, -1, tuple(_get_viz().contour_inner_color_bgr), thickness)

        # Convert pixel measurements to physical units (holes-filled outer region;
        # surface-connected cavities are corrected after the loop).
        inner_perim_px = sum(cv2.arcLength(c, True) for c in inner_filtered)
        area_px = sum(cv2.contourArea(c) for c in inner_filtered)
        inner_perim_mm = inner_perim_px * mm_per_px
        area_mm2 = area_px * (mm_per_px ** 2)
        comp_2D = compactness_2D(area_mm2, inner_perim_mm)

        # Accumulate in physical units
        sum_inner_mm += inner_perim_mm
        sum_area_mm2 += area_mm2
        pos = float(k) * axis_scale
        raw_lateral.append((idx, pos, inner_perim_mm))
        raw_volume.append((idx, pos, area_mm2))
        if cavity_correction_enabled:
            center_px = get_red_rect_offset(img_rgb)
            slice_records.append(SliceRecord(
                idx=idx, position_mm=pos,
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
        section["slice_idx"] = np.full(section.n_points, idx, dtype=np.int32)
        plane["slice_idx"] = np.full(plane.n_points, idx, dtype=np.int32)
        sections_list.append(section)
        sections_list.append(plane)

    # ---- end with: plotter is fully and safely closed here ----

    # Surface-connected cavity correction (mirrors the Area/Volume tools): the
    # open-cavity wall perimeter is added to the surface lateral and the
    # open-cavity area is subtracted from the volume + caps. Enclosed voids stay
    # solid. Yellow-outline the open cavities on the saved slices.
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

    # Surface area = Simpson lateral + caps; Volume = ∫ area dh via Simpson's
    # rule. Exact mesh.area / mesh.volume kept only for verification.
    _mesh_area_phys, _mesh_volume_phys = _mesh_area_volume_physical(mesh, mesh_dim_scaled)
    Volume = volume_simpson(volume_samples)
    Area, _lat_phys, _caps_phys = total_surface_area_simpson(lateral_samples, volume_samples)
    print(f"[VTK Compactness] Volume={Volume:.3f} {unit}³ "
          f"Surface area: lateral+caps={Area:.3f} {unit}²")
    if cavity_correction_enabled:
        print(f"[VTK Compactness] Cavity correction: "
              f"{cavity_corr.n_surface_connected} surface-connected, "
              f"{cavity_corr.n_enclosed} enclosed")
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
            "Length unit": unit,
            "Cavity correction": "on" if cavity_correction_enabled else "off",
        }
        if cavity_correction_enabled:
            parameters["Cavity area threshold (mm²)"] = float(cavity_area_threshold_mm2)
        totals = {
            "Compactness": round(float(comp_3D), 4),
            f"Volume ({unit}^3)": round(float(Volume), 4),
            f"Surface Area ({unit}^2)": round(float(Area), 4),
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
        xlsx_path = os.path.join(out_dir, f"Mesh_compactness_{Slice_direction}.xlsx")
        write_results_workbook(xlsx_path, [sheet])
        print(f"[VTK Compactness] Saved Excel → {xlsx_path}")
    except Exception as ex:
        print(f"[VTK Compactness] WARN: could not save Excel: {ex}")


    # Always return a 3-tuple
    return  comp_3D, saved_pngs, valid_slices
