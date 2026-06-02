"""Metrics storage, table display, and Excel export for FetoMorph."""

from __future__ import annotations

from deps import *
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from FetoMorph import MainWindow

logger = logging.getLogger(__name__)


class MetricsStore:
    """Owns the in-memory metrics dict, the Metrics dock widget, and Excel export."""

    def __init__(self, mw: MainWindow) -> None:
        self.mw = mw

        # Data storage
        self.metrics: dict[str, list[dict]] = {}
        self.annotation_records: list[dict] = []
        self.annotations_by_source: dict[str, list[dict]] = {}
        self._roi_counter_by_source: dict[str, int] = {}
        self.annotation_labels_by_path: dict[str, str] = {}

        # Dock widgets (created in init_dock)
        self.metricsDock: QDockWidget | None = None
        self.metricsView: QTableView | None = None
        self._metrics_model: QStandardItemModel | None = None

        self.init_dock()

    # ------------------------------------------------------------------
    # Dock setup
    # ------------------------------------------------------------------

    def init_dock(self) -> None:
        """Create the Metrics dock widget, its table model, and wire toolbar actions."""
        mw = self.mw

        self.metricsDock = QDockWidget("Metrics", mw)
        self.metricsDock.setObjectName("MetricsDock")
        self.metricsDock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        host = QWidget()
        v = QVBoxLayout(host)
        v.setContentsMargins(0, 0, 0, 0)

        tb = QToolBar()
        act_copy = QAction("Copy", mw)
        act_copy.setShortcut(QKeySequence.Copy)
        act_export = QAction("Export Excel…", mw)
        act_clear = QAction("Clear (this file)", mw)
        tb.addAction(act_copy)
        tb.addAction(act_export)
        tb.addSeparator()
        tb.addAction(act_clear)
        v.addWidget(tb)

        self.metricsView = QTableView()
        self.metricsView.setSortingEnabled(True)
        self.metricsView.horizontalHeader().setStretchLastSection(True)
        self.metricsView.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        v.addWidget(self.metricsView)

        self.metricsDock.setWidget(host)
        mw.addDockWidget(Qt.RightDockWidgetArea, self.metricsDock)
        self.metricsDock.hide()

        self._metrics_model = QStandardItemModel(0, 0, mw)
        self.metricsView.setModel(self._metrics_model)

        act_copy.triggered.connect(self.copy_selection)
        act_export.triggered.connect(self.export_metrics_excel)
        act_clear.triggered.connect(self.clear_current_file)

    # ------------------------------------------------------------------
    # Row management
    # ------------------------------------------------------------------

    def ensure_metric_row(
        self,
        path: str | None,
        kind: str | None,
        label: str | None = None,
        annotation: str | None = None,
        source: str | None = None,
        direction: str | None = None,
        *,
        pixel_size: float | None = None,
        pixel_size_units: str | None = None,
        kernel_size: float | None = None,
        unit: str | None = None,
        slice_thickness: float | None = None,
        contour_mode: str | None = None,
        new_on_param_change: bool = False,
    ):
        """Ensure a metrics row exists for a given (path, annotation) pair.

        Returns the existing or newly created row dict, or None if *path* is falsy.
        """
        if not path:
            return None

        rows = self.metrics.get(path)
        if isinstance(rows, dict):
            rows = [rows]
            self.metrics[path] = rows
        elif rows is None:
            rows = []
            self.metrics[path] = rows

        last = next((r for r in reversed(rows) if r.get("Annotation") == annotation), None)

        def differs(key: str, new_val):
            if new_val is None:
                return False
            if last is None:
                return True
            return new_val != last.get(key)

        make_new = (
            last is None
            or (new_on_param_change and any([
                differs("PixelSize", pixel_size),
                differs("PixelSizeUnits", pixel_size_units),
                differs("KernelSize", kernel_size),
                differs("SliceThickness", slice_thickness),
                differs("LengthUnit", unit),
                differs("SliceDirection", direction),
                differs("ContourMode", contour_mode),
            ]))
        )

        if make_new:
            row = {
                "File": os.path.basename(path),
                "Kind": kind,
                "Label": label,
                "Annotation": annotation,
                "Source": source,
                "SliceDirection": direction,
                "Area": None,
                "PixelSize": pixel_size if pixel_size is not None else (last.get("PixelSize") if last else None),
                "PixelSizeUnits": pixel_size_units if pixel_size_units is not None else (last.get("PixelSizeUnits") if last else None),
                "KernelSize": kernel_size if kernel_size is not None else (last.get("KernelSize") if last else None),
                "LengthUnit": unit if unit is not None else (last.get("LengthUnit") if last else None),
                "SliceThickness": slice_thickness if slice_thickness is not None else (last.get("SliceThickness") if last else None),
                "ContourMode": (
                    str(contour_mode)
                    if contour_mode is not None
                    else (last.get("ContourMode") if last else None)
                ),
                "Length(PA)": None,
                "Width(LR)": None,
                "Height(IS)": None,
                "Volume": None,
                "Perimeter": None,
                "Perimeter_convex": None,
                "SliceKind": None,
                "SulciCount": None,
                "PrimarySulciCount": None,
                "SecondarySulciCount": None,
                "TertiarySulciCount": None,
                "UnclassifiedSulciCount": None,
                "MinDepth": None,
                "MaxDepth": None,
                "MeanDepth": None,
                "LGI": None,
                "Compactness": None,
            }
            rows.append(row)
            return row

        last["File"] = os.path.basename(path)
        if kind is not None:
            last["Kind"] = kind
        if label is not None:
            last["Label"] = label
        if pixel_size is not None:
            last["PixelSize"] = pixel_size
        if pixel_size_units is not None:
            last["PixelSizeUnits"] = pixel_size_units
        if kernel_size is not None:
            last["KernelSize"] = kernel_size
        if unit is not None:
            last["LengthUnit"] = unit
        if slice_thickness is not None:
            last["SliceThickness"] = slice_thickness
        if direction is not None:
            last["SliceDirection"] = direction
        if contour_mode is not None:
            last["ContourMode"] = str(contour_mode)
        return last

    def record_metric_for(self, path: str, annotation: str | None = None, source: str | None = None, **vals):
        """Record one or more metric values for a given file path."""
        if not path:
            return
        kind = getattr(self.mw, "current_kind", None)
        label = getattr(self.mw, "custom_label", None)

        psize = vals.pop("pixel_size", None)
        punit = vals.pop("pixel_size_units", None)
        ksize = vals.pop("kernel_size", None)
        thicsl = vals.pop("slice_thickness", None)
        direction = vals.pop("direction", None)
        uni = vals.pop("unit", None)
        cmode = vals.pop("contour_mode", None)
        row = self.ensure_metric_row(
            path, kind, label, annotation,
            source, direction,
            pixel_size=psize,
            pixel_size_units=punit,
            kernel_size=ksize,
            unit=uni,
            slice_thickness=thicsl,
            contour_mode=cmode,
            new_on_param_change=True,
        )

        sk = vals.pop("slice_kind", None)
        if sk is not None:
            row["SliceKind"] = sk

        sd = vals.pop("sulci_depth", None)
        if sd is not None:
            if isinstance(sd, (list, tuple)):
                n = len(sd)
                if n == 0:
                    row["SulciCount"] = row["MinDepth"] = row["MaxDepth"] = row["MeanDepth"] = None
                else:
                    row["SulciCount"] = n
                    row["MinDepth"] = min(sd)
                    row["MaxDepth"] = max(sd)
                    row["MeanDepth"] = sum(sd) / n
            else:
                raise ValueError("sulci_depth must be an iterable")

        sds = vals.pop("sulci_depth_sets", None)
        if sds is not None and isinstance(sds, dict):
            row["PrimarySulciCount"] = len(sds.get("primary", []))
            row["SecondarySulciCount"] = len(sds.get("secondary", []))
            row["TertiarySulciCount"] = len(sds.get("tertiary", []))
            row["UnclassifiedSulciCount"] = len(sds.get("unclassified", []))

        ld = vals.pop("dimensions", None)
        if ld is not None:
            if isinstance(ld, (list, tuple)):
                nl = len(ld)
                if nl == 3:
                    row["Length(PA)"] = ld[0]
                    row["Width(LR)"] = ld[1]
                    row["Height(IS)"] = ld[2]
                else:
                    raise ValueError("one or more dimensions are missing")
            else:
                raise ValueError("dimensions must be an iterable")

        keymap = {
            "area": "Area",
            "volume": "Volume",
            "perimeter": "Perimeter",
            "perimeter_convex": "Perimeter_convex",
            "lgi": "LGI",
            "compactness": "Compactness",
        }
        for k, v in vals.items():
            col = keymap.get(k.lower(), k)
            row[col] = v

        self.rebuild_for_current()

    # ------------------------------------------------------------------
    # Table display
    # ------------------------------------------------------------------

    def headers(self) -> list[str]:
        """Return the ordered list of column header strings."""
        return [
            "File", "Kind", "Label", "Annotation", "Source", "SliceDirection",
            "PixelSize", "PixelSizeUnits", "KernelSize", "LengthUnit", "SliceThickness",
            "ContourMode", "Length(PA)", "Width(LR)", "Height(IS)", "SliceKind",
            "Area", "Volume", "Perimeter", "Perimeter_convex",
            "SulciCount", "PrimarySulciCount", "SecondarySulciCount",
            "TertiarySulciCount", "UnclassifiedSulciCount",
            "MinDepth", "MaxDepth", "MeanDepth",
            "LGI", "Compactness", 
        ]

    def show_results_dock(self) -> None:
        """Refresh and display the Metrics dock for the currently loaded file."""
        self.rebuild_for_current()
        self.metricsDock.show()
        self.metricsDock.raise_()

    def rebuild_for_current(self) -> None:
        """Rebuild the table for the currently open file."""
        hdrs = self.headers()
        m = self._metrics_model
        m.clear()
        m.setHorizontalHeaderLabels(hdrs)

        cur_path = getattr(self.mw, "current_path", None)
        rows = []
        if cur_path and cur_path in self.metrics:
            for rec in (self.metrics.get(cur_path) or []):
                if not isinstance(rec, dict):
                    continue
                rows.append([rec.get(h, "") for h in hdrs])

        for row in rows:
            items = []
            for val in row:
                txt = "" if val is None else str(val)
                it = QStandardItem(txt)
                try:
                    f = float(txt)
                except ValueError:
                    pass
                else:
                    it.setText(f"{f:.3f}")
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                items.append(it)
            m.appendRow(items)

    def append_record(self) -> None:
        """Append the newest metrics row for the current file into the table model."""
        cur_path = getattr(self.mw, "current_path", None)
        if not cur_path or cur_path not in self.metrics:
            return
        seq = self.metrics[cur_path]
        if not seq:
            return
        rec = seq[-1]
        if not isinstance(rec, dict):
            return

        hdrs = self.headers()
        if self._metrics_model.columnCount() != len(hdrs):
            self.rebuild_for_current()
            return

        row_vals = [rec.get(h, "") for h in hdrs]
        items = []
        for val in row_vals:
            txt = "" if val is None else str(val)
            it = QStandardItem(txt)
            try:
                f = float(txt)
            except ValueError:
                pass
            else:
                it.setText(f"{f:.3f}")
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            items.append(it)
        self._metrics_model.appendRow(items)

    def copy_selection(self) -> None:
        """Copy selected table cells as CSV (with header)."""
        sel = self.metricsView.selectionModel()
        if not sel or not sel.hasSelection():
            return
        idxs = sorted(sel.selectedIndexes(), key=lambda i: (i.row(), i.column()))
        rows = {}
        model = self._metrics_model
        for i in idxs:
            rows.setdefault(i.row(), {})[i.column()] = model.item(i.row(), i.column()).text()

        header = ",".join(self.headers())
        lines = [header]
        for r in sorted(rows):
            cols = []
            for c in range(model.columnCount()):
                cols.append(rows[r].get(c, ""))
            lines.append(",".join(cols))
        QApplication.clipboard().setText("\n".join(lines))

    def clear_current_file(self) -> None:
        """Clear in-memory metrics for the current file and refresh the table."""
        cur_path = getattr(self.mw, "current_path", None)
        if not cur_path:
            return
        self.metrics[cur_path] = []
        self.rebuild_for_current()
        try:
            print(f"[Metrics] Cleared metrics for {cur_path}")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Excel export
    # ------------------------------------------------------------------

    def export_metrics_excel(self) -> None:
        """Export collected metrics to an Excel workbook with one
        spec-layout sheet per source file. Each per-measurement row
        carries the adjustment parameters that were active when that
        row was recorded, so re-measuring the same file with different
        kernel size / pixel size / threshold / contour mode produces
        rows that reflect those changes."""
        mw = self.mw
        if not self.metrics:
            QMessageBox.information(mw, "Export Metrics", "No metrics to export yet.")
            return

        from helpers.results_excel_format import (
            ResultsSheet, write_results_workbook,
        )

        metric_keys = (
            "Area", "Perimeter", "LGI", "Compactness",
            "PrimarySulciCount", "SecondarySulciCount",
            "TertiarySulciCount", "UnclassifiedSulciCount",
            "PrimaryMeanDepth", "SecondaryMeanDepth",
            "TertiaryMeanDepth", "UnclassifiedMeanDepth",
        )
        # Adjustment parameters that may differ across measurement runs
        # of the same file. Each becomes a per-row column so the values
        # used for each measurement are visible inline.
        extra_columns = (
            "Kernel size",
            "Pixel spacing",
            "Slice thickness",
            "Contour mode",
            "Slice direction",
            "Length unit",
        )

        def _pixel_spacing(r: dict) -> str | None:
            v = r.get("PixelSize")
            if v in (None, ""):
                return None
            u = r.get("PixelSizeUnits") or r.get("LengthUnit") or ""
            return f"{v} {u}/pixel".strip()

        sheets: list[ResultsSheet] = []
        for path, rows in self.metrics.items():
            if not rows:
                continue
            if isinstance(rows, dict):
                rows = [rows]
            non_empty = [r for r in rows if isinstance(r, dict)
                         and any(r.get(k) is not None for k in metric_keys)]
            if not non_empty:
                continue

            results_rows = []
            for i, r in enumerate(non_empty, start=1):
                section = (r.get("Annotation") or r.get("Source")
                           or r.get("Label") or f"Row {i}")
                row_dict = {
                    "Section": section,
                    "Kernel size": r.get("KernelSize"),
                    "Pixel spacing": _pixel_spacing(r),
                    "Slice thickness": r.get("SliceThickness"),
                    "Contour mode": r.get("ContourMode"),
                    "Slice direction": r.get("SliceDirection"),
                    "Length unit": r.get("LengthUnit"),
                }
                for k in metric_keys:
                    row_dict[k] = r.get(k)
                results_rows.append(row_dict)

            sheet_name = os.path.basename(path) if path else "Results"
            # The top Parameters block is left intentionally empty for the
            # cross-measurement dock export: the per-row columns below
            # carry the authoritative per-run values, and a single summary
            # block at the top would silently hide rows whose parameters
            # differ from the first one.
            sheets.append(ResultsSheet(
                sheet_name=sheet_name,
                file_name=os.path.basename(path) if path else None,
                folder=(os.path.dirname(path) if path else None) or None,
                parameters={},
                rows=results_rows,
                extra_columns=extra_columns,
                drop_empty_columns=True,
            ))

        if not sheets:
            QMessageBox.information(
                mw, "Export Metrics",
                "No non-empty metrics to export yet.")
            return

        last_dir = getattr(self.mw, "last_dir", os.getcwd())
        default_name = os.path.join(last_dir, "metrics.xlsx")
        path, _ = QFileDialog.getSaveFileName(
            mw, "Export Metrics to Excel…", default_name,
            "Excel Workbook (*.xlsx)",
        )
        if not path:
            return
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"

        folder = os.path.dirname(path)
        if folder and not os.path.exists(folder):
            reply = QMessageBox.question(
                mw, "Create Folder?",
                f"The folder\n\n{folder}\n\ndoes not exist. Create it?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                try:
                    os.makedirs(folder, exist_ok=True)
                    print(f"Created folder: {folder}")
                except Exception as ex:
                    logger.error("Error creating folder: %s", ex)
                    QMessageBox.critical(
                        mw, "Export Failed",
                        f"Could not create folder:\n{ex}")
                    return
            else:
                return

        try:
            write_results_workbook(path, sheets)
            print(f"Exported metrics to: {path}")
            if folder:
                self.mw.last_dir = folder
        except Exception as ex:
            logger.error("Error exporting metrics: %s", ex)
            QMessageBox.critical(
                mw, "Export Failed",
                f"{type(ex).__name__}: {ex}")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def depth_summary(vals, unit: str) -> str:
        if not isinstance(vals, (list, tuple)) or len(vals) == 0:
            return "No sulci are identified; the profile appears less lissencephalic."
        return ", ".join(f"{float(v):.2f}" for v in vals[:3]) + f" {unit}"
