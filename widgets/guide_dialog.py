"""Scrollable in-app User Guide dialog.

Shows a rich-text walkthrough of the main FetoMorph workflows in a
``QTextBrowser`` (so it scrolls and supports headings/links), sized as a
normal resizable window rather than a cramped message box.
"""

from deps import *
# QTextBrowser is not re-exported by deps.py's __all__; import it directly.
from PySide6.QtWidgets import QTextBrowser

try:
    from theme import BASE, TEXT, ACCENT, BORDER
except Exception:  # pragma: no cover - theme is always present in the app
    BASE, TEXT, ACCENT, BORDER = "#23272e", "#e6e6e6", "#2dd4bf", "#454b55"

_MUTED = "#9aa0a6"


_GUIDE_HTML = f"""
<html><body style="color:{TEXT}; font-size:13px; line-height:1.5;">
<h2 style="color:{ACCENT};">FetoMorph — User Guide</h2>
<p>FetoMorph measures the shape of the developing fetal brain and scores how
closely a brain (real <i>or</i> simulated) matches each gestational week. Its
purpose is to <b>validate computational models of brain development</b>: a
simulated brain is measured with the same pipeline used on real data, then
compared against age-specific references.</p>
<p>The <b>ribbon</b> at the top mirrors the menus; when a tab is crowded, use the
chevron arrows on either side to scroll to more tools. Most tools stay disabled
until data of the matching kind (image / mesh / volume) is loaded.</p>

<h3 style="color:{ACCENT};">Quick start</h3>
<ol>
  <li><b>Load data</b> — <i>File → Import</i>, or pick a bundled dataset from the
      <i>Examples</i> menu.</li>
  <li><b>Set the scale</b> — <i>Adjustments → Set Image Scale…</i> (2D) or
      <i>Mesh dimensions…</i> (VTK) so results come out in real units.</li>
  <li><b>Measure</b> — <i>Process → Measure → All hallmarks</i> computes
      everything at once.</li>
  <li><b>Validate</b> — <i>Process → Analysis → Similarity Profile</i> scores the
      result against the gestational-age references.</li>
  <li><b>Export</b> — <i>File → Export Metrics to Excel…</i> or
      <i>Save Data As…</i>.</li>
</ol>

<h3 style="color:{ACCENT};">Key concepts</h3>
<ul>
  <li><b>Scale &amp; units.</b> Areas, perimeters, depths and volumes are only
      meaningful once FetoMorph knows the physical size of a pixel/voxel. Always
      calibrate the scale before measuring; results follow the chosen unit
      (mm / cm).</li>
  <li><b>Slice kind.</b> A small deep-learning model labels each image as
      <i>axial</i>, <i>coronal</i>, <i>sagittal</i>, or <i>cropped</i>
      (not-full-slice). This decides how the sulcus filter and references are
      applied. Use <i>Adjustments → Slice Kind Override…</i> if the automatic
      guess is wrong.</li>
  <li><b>Sulcus classes.</b> Detected sulci are grouped as <i>primary</i>,
      <i>secondary</i>, and <i>tertiary</i>. The class is decided by comparing
      each sulcus's depth against predetermined thresholds expressed as a
      <i>ratio of the total height (extent)</i> of the section — not an absolute
      depth — so it stays consistent across sizes. (Cropped sections report a
      single <i>unclassified</i> pool.) Each measurement reports per-class counts
      and depths.</li>
  <li><b>Gyrification (LGI / GI).</b> A perimeter ratio of the folded cortex to
      its smooth outer envelope — higher means more folded. How smooth that outer
      envelope is (and therefore the LGI value) is controlled by the
      <i>kernel size</i>, set in <i>Adjustments → Set Kernel Size…</i>: a larger
      kernel smooths over more folds and raises the LGI.</li>
  <li><b>Normalized (scale-free) metrics.</b> For cropped sections and simulated
      geometries with arbitrary units, FetoMorph compares LGI, compactness,
      sulcal counts and depth <i>normalized</i> to the shape itself, so absolute
      size does not matter.</li>
</ul>

<h3 style="color:{ACCENT};">File</h3>
<ul>
  <li><b>Import → Image… / .vtk file… / .stl file… / NIfTI…</b> — load a 2D
      section (PNG/JPG/TIFF…), a VTK or STL surface mesh, or a 3D NIfTI
      segmentation volume (<code>.nii</code>/<code>.nii.gz</code>).</li>
  <li><b>Recent</b> — reopen a recently used file.</li>
  <li><b>Open Home Folder… / Open Current Temp Folder…</b> — jump to the data
      home, or the scratch folder holding the current run's outputs.</li>
  <li><b>Show Results…</b> — reopen the results of the current run.</li>
  <li><b>Save View As…</b> — save a snapshot image of the current view.</li>
  <li><b>Save Data As…</b> — save the whole output folder (charts, annotated
      images, source copies, Excel) to a chosen location.</li>
  <li><b>Export Metrics to Excel…</b> — export just the measurements table (one
      sheet per measured file).</li>
  <li><b>Reset view… / Close / Quit</b> — reset the display, close the file, or
      exit.</li>
</ul>

<h3 style="color:{ACCENT};">Process → Measure</h3>
<ul>
  <li><b>All hallmarks</b> — computes every metric at once and writes the full
      results (recommended start).</li>
  <li><b>Volumes</b> — 3D volume by Simpson integration of the cross-section
      areas (meshes / NIfTI).</li>
  <li><b>Area</b> — surface area (meshes) or cross-section area (2D).</li>
  <li><b>Perimeter</b> — contour perimeter. <li> <b>Curve Length</b> follows the folded
      outline; <b>Straight</b> uses a simplified contour. The estimation method
      (Arc length or Crofton) is set in <i>Adjustments → Perimeter Method…</i>.</li>
  <li><b>Sulci Depth</b> — detects sulci as convexity defects, filters them by the
      depth threshold, and classifies primary / secondary / tertiary with
      per-class counts and depths; labels are drawn on the image.</li>
</ul>

<h3 style="color:{ACCENT};">Process → Analysis</h3>
<ul>
  <li><b>LGI</b> — local/global gyrification index (folded perimeter ÷ outer
      envelope perimeter).</li>
  <li><b>Curvature</b> — contour curvature analysis.</li>
  <li><b>Compactness</b> — how compact the shape is (2D and 3D forms).</li>
  <li><b>Hausdorff distance</b> — the maximum shape difference between two
      contours; useful for comparing a simulation to a target.</li>
  <li><b>Similarity Profile (GASP)</b> — the validation tool. It scores the
      measured brain against per-week reference statistics (weeks 24–36) and
      reports the best-matching gestational age plus a per-metric breakdown and a
      profile chart. It runs from an imported image, hand-entered values, or the
      current mesh section. Cropped/simulated inputs are scored with the
      scale-free normalized metrics. Method, penalties and weights are set in
      <i>Adjustments → GASP Options</i> and <i>Settings → Preferences</i>.</li>
</ul>

<h3 style="color:{ACCENT};">Process (other)</h3>
<ul>
  <li><b>Process images batch</b> — measure a whole folder of images at once.
      Point it at a folder; it detects the pixel spacing, measures every image,
      and writes a structured Excel report — per-image rows, every individual
      sulcus value per class, and an automatic statistical <i>Analysis</i>
      sheet.</li>
  <li><b>Optimization</b> — multi-objective optimisation (NSGA-II/III) that finds
      the Pareto-optimal slices across one or more measurement Excel files.
      Select the files, then build the run in the dialog: add an
      <i>objective</i> row per metric to optimise, each set to <i>Maximize</i>
      or <i>Minimize</i> — at least two are needed, since the solver trades them
      off against one another. The dropdowns list whatever metrics your files
      actually contain (LGI, area, perimeter, compactness, sulci depths and
      counts, cell density…), so anything the measurement tools export can be
      optimised. Add <i>constraint</i> rows to restrict which slices qualify —
      each bounds a column from either side, e.g. <i>Number of sulci ≥ 2</i> or
      <i>cell density ≤ 2500</i>. Hover any row for an explanation of the
      metric and its range in your data. NSGA-II suits up to 3 objectives and is
      disabled beyond that; NSGA-III handles any number. Results are written to
      the output folder as <i>Pareto-optimal solutions.xlsx</i>, a scatter plot
      per objective pair, and a parameters file recording the run.</li>
  <li><b>Nifti masking…</b> — convert a NIfTI volume into slice images.</li>
  <li><b>Nifti extract regions…</b> — extract selected label regions from a
      segmentation.</li>
</ul>

<h3 style="color:{ACCENT};">Adjustments</h3>
<ul>
  <li><b>Custom label…</b> — set the text label drawn on exported images.</li>
  <li><b>Set Image Scale… / Set Scale From Scalebar…</b> — calibrate mm-per-pixel
      directly, or by measuring a scale bar drawn in the image.</li>
  <li><b>Set Kernel Size…</b> — morphological kernel (mm) used to build the outer
      closing surface for GI / LGI.</li>
  <li><b>Perimeter Method…</b> — <i>Arc length contour perimeter</i> or
      <i>Crofton perimeter</i>, with optional contour simplification (epsilon).</li>
  <li><b>Set Slice Thickness…</b> — spacing between slices for 3D reconstruction
      (in the length unit, not necessarily mm).</li>
  <li><b>Set filtered Threshold…</b> <span style="color:{_MUTED};">(Ctrl+T)</span>
      — minimum contour area kept, to drop specks/noise.</li>
  <li><b>Sulcus Depth Threshold…</b> — minimum depth (mm) counted as a sulcus,
      applied across <i>every</i> measurement (default 1&nbsp;mm).</li>
  <li><b>Slice Kind Override…</b> — force axial / coronal / sagittal / cropped
      instead of the automatic classifier.</li>
  <li><b>GASP Options</b> — scoring method, out-of-range penalties, and per-metric
      weights for the Similarity Profile.</li>
  <li><b>Contour Accounting</b> — Outer contours only / Subtract internal contours
      / Internal contours only (how holes are treated in area/perimeter).</li>
  <li><b>Surface-Connected Cavities…</b> — enable/tune cavity correction: cavities
      that open to the surface are removed from the volume and their walls added
      to the surface area; fully-enclosed voids are kept solid.</li>
  <li><b>Annotation…</b> <span style="color:{_MUTED};">(Ctrl+Shift+A)</span> — drag
      a square on the image and save the crop.</li>
  <li><b>Upscale Image…</b> <span style="color:{_MUTED};">(Ctrl+Shift+U)</span> —
      LANCZOS upscale + sharpen the current image and reload it (only when an
      image is loaded).</li>
  <li><b>ROI selection…</b> — pick which label IDs to include for NIfTI
      hallmarks.</li>
  <li><b>Mesh dimensions…</b> — define the physical size of a VTK mesh.</li>
</ul>

<h3 style="color:{ACCENT};">Freesurfer Viewer</h3>
<ul>
  <li><b>Surfaces…</b> — display reconstructed pial / white surfaces.</li>
  <li><b>Morph maps…</b> — display a morph map (sulc, thickness, curvature).</li>
  <li><b>Pial → STL… / Combined STL…</b> — convert a pial surface to STL, or merge
      surfaces into one mesh.</li>
</ul>

<h3 style="color:{ACCENT};">Examples</h3>
<ul>
  <li><b>Fetal brain 2D sections → Filled / Cropped 2D sections</b>
      <span style="color:{_MUTED};">(Ctrl+Alt+F / Ctrl+Alt+C)</span> — bundled 2D
      sections by gestational week and axis.</li>
  <li><b>Fetal brain 3D → Fetal surface MRI / Fetal brain STL</b> — bundled 3D
      NIfTI volumes and STL meshes by gestational week (24–38). Each opens a week
      picker.</li>
</ul>
<p>Great for trying the whole pipeline before using your own data.</p>

<h3 style="color:{ACCENT};">Settings &amp; About</h3>
<ul>
  <li><b>Settings → Preferences…</b> — visualization options, colours, and the
      GASP metric weights.</li>
  <li><b>About →</b> About FetoMorph, <b>User Guide</b> (this window, F1),
      Contributors, Acknowledgements, Copyright and License, Citing FetoMorph,
      Icon Credits, and ReadMe.</li>
</ul>

<h3 style="color:{ACCENT};">Keyboard shortcuts</h3>
<table cellspacing="0" cellpadding="4" style="border-collapse:collapse;">
  <tr><td><b>Ctrl+O</b></td><td>Import image</td>
      <td style="padding-left:18px;"><b>Ctrl+E</b></td><td>Export metrics to Excel</td></tr>
  <tr><td><b>Ctrl+Shift+V</b></td><td>Import .vtk</td>
      <td style="padding-left:18px;"><b>Ctrl+S</b></td><td>Save view as</td></tr>
  <tr><td><b>Ctrl+Shift+L</b></td><td>Import .stl</td>
      <td style="padding-left:18px;"><b>Ctrl+Shift+S</b></td><td>Save data as</td></tr>
  <tr><td><b>Ctrl+Shift+N</b></td><td>Import NIfTI</td>
      <td style="padding-left:18px;"><b>Ctrl+R</b></td><td>Reset view</td></tr>
  <tr><td><b>Ctrl+Shift+R</b></td><td>Show results / ROI</td>
      <td style="padding-left:18px;"><b>Ctrl+T</b></td><td>Filtered threshold</td></tr>
  <tr><td><b>Ctrl+Shift+A</b></td><td>Annotation</td>
      <td style="padding-left:18px;"><b>Ctrl+Shift+U</b></td><td>Upscale image</td></tr>
  <tr><td><b>Ctrl+Alt+F / Ctrl+Alt+C</b></td><td>Example 2D sections</td>
      <td style="padding-left:18px;"><b>F1</b></td><td>This guide</td></tr>
</table>

<h3 style="color:{ACCENT};">Tips &amp; troubleshooting</h3>
<ul>
  <li><b>Numbers look wrong / too large or small?</b> Re-check the scale
      (mm-per-pixel or mesh dimensions) — it is the most common cause.</li>
  <li><b>Sulci over- or under-counted?</b> Adjust <i>Sulcus Depth Threshold…</i>;
      raise it to ignore shallow folds, lower it to catch fine ones.</li>
  <li><b>Wrong reference / axis?</b> Set <i>Slice Kind Override…</i> so the correct
      (full vs cropped, and axis) reference is used for GASP.</li>
  <li><b>A tool is greyed out?</b> It needs data of a matching kind loaded first
      (e.g. mesh tools need an STL/VTK).</li>
  <li><b>Want a clean run folder?</b> Everything for the current run lands in the
      temp folder (<i>File → Open Current Temp Folder…</i>); use
      <i>Save Data As…</i> to keep it.</li>
</ul>

<p style="color:{_MUTED};"><i>Tip: start from an Example, run <b>All hallmarks</b>,
then a <b>Similarity Profile</b> to see the whole pipeline end to end.</i></p>
</body></html>
"""


class GuideDialog(QDialog):
    """A resizable, scrollable window showing the FetoMorph user guide."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("FetoMorph — User Guide")
        self.resize(760, 640)

        layout = QVBoxLayout(self)

        self.browser = QTextBrowser(self)
        self.browser.setOpenExternalLinks(True)
        self.browser.setStyleSheet(
            f"QTextBrowser {{ background: {BASE}; color: {TEXT}; "
            f"border: 1px solid {BORDER}; }}"
        )
        self.browser.setHtml(_GUIDE_HTML)
        layout.addWidget(self.browser)

        buttons = QDialogButtonBox(QDialogButtonBox.Close, self)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
