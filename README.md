FetoMorph -- a desktop application for morphometric analysis of fetal brain data.

Supports 2-D histological images (PNG/JPEG/TIFF), NIfTI volumetric scans,
VTK legacy meshes, and STL surface meshes.  The GUI is built with PySide6
and VTK, exposing measurement routines (area, volume, perimeter, sulci
depth, LGI, curvature, Hausdorff distance) through both menus and a
ribbon toolbar.  Results are collected in an in-memory metrics store and
can be exported to Excel.

Typical workflow:
    1. Import a file (image, NIfTI, VTK, or STL).
    2. Adjust parameters (pixel scale, kernel size, ROI selection, etc.).
    3. Run one or more measurements from the Process menu.
    4. Review results in the Metrics dock and export to Excel.

