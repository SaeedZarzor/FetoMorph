"""Slice thickness dialog widget.

A single slider drives the slicing of a 3-D mesh, and two read-outs update
together as it moves:

* **Slice thickness** – the inter-slice distance (mm), a floating-point value.
* **Number of slices** – an integer count.

They are inverse views of the same quantity over the mesh's shortest dimension
``L``: ``number_of_slices = L / slice_thickness``. Either read-out can be typed
into; the slider and the other read-out follow. The slice-thickness maximum is
``L / 2`` (passed in as ``maximum``), so the geometry always yields at least two
slices.
"""

from deps import *

class SliceThicknessDialog(QDialog):
    """Dialog for selecting slice thickness with one slider and two read-outs.

    An integer QSlider is internally mapped to the float thickness range
    [minimum, maximum]; a thickness spin box and a slice-count spin box show the
    two equivalent values, linked via ``L = 2 * maximum`` (the mesh's shortest
    dimension).
    """

    def __init__(self, parent=None, initial: float = 0.5, minimum: float = 0.001,
                 maximum: float = 10.0, step: float = 0.005,
                 reference_length: float | None = None, unit: str = "mm"):
        """Initialise the slice thickness dialog.

        Args:
            parent: Parent widget.
            initial: Starting thickness value (in ``unit``).
            minimum: Lower bound of the thickness range (in ``unit``).
            maximum: Upper bound of the thickness range (in ``unit``) — half the
                shortest mesh dimension, so there are always at least two slices.
            step: Step size for the slider and thickness spin box.
            reference_length: Shortest mesh dimension ``L`` (in ``unit``) used to
                map thickness <-> slice count. Defaults to ``2 * maximum``.
            unit: Length unit label for thickness/dimension read-outs.
        """
        super().__init__(parent)
        self.setWindowTitle("Set Slice Thickness")
        self.setModal(True)
        self._unit = str(unit or "mm")

        # clamp + map float range -> integer slider steps
        initial = max(minimum, min(maximum, initial))
        self._min = float(minimum)
        self._max = float(maximum)
        self._step = float(step)
        self._imax = max(1, int(round((self._max - self._min) / self._step)))

        # Reference length: the shortest dimension. maximum is L/2, so L = 2*max.
        self._L = float(reference_length) if reference_length else (2.0 * self._max)
        # Slice-count bounds: thickness=max -> fewest slices; thickness=min -> most.
        self._n_min = max(1, int(round(self._L / self._max)))
        self._n_max = max(self._n_min, int(round(self._L / self._min)))

        self._syncing = False
        form = QFormLayout(self)

        # ---- single slider (drives thickness) ----
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setRange(0, self._imax)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(max(1, int(round(0.1 / self._step))))
        self.slider.setValue(self._to_int(initial))
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setMinimumWidth(280)
        self.slider.setTickInterval(max(1, int(round(0.05 / self._step))))
        self.slider.setStyleSheet("""
        QSlider::groove:horizontal {height: 6px; background: #444; border-radius: 3px;}
        QSlider::sub-page:horizontal {background: #2aa1ff; height: 6px; border-radius: 3px;}
        QSlider::add-page:horizontal {background: #333; height: 6px; border-radius: 3px;}
        QSlider::handle:horizontal {width: 16px; background: #2aa1ff; border-radius: 8px; margin: -5px 0;}
        """)

        # ---- thickness read-out (float, editable) ----
        self.spin = QDoubleSpinBox(self)
        self.spin.setRange(self._min, self._max)
        self.spin.setSingleStep(self._step)
        self.spin.setDecimals(max(0, min(6, self._decimals(self._step))))
        self.spin.setValue(initial)
        self.spin.setSuffix(f" {self._unit}")
        font = self.spin.font(); font.setPointSize(14); self.spin.setFont(font)
        self.spin.setMinimumWidth(100)

        # ---- slice-count read-out (int, editable, inverse of thickness) ----
        self.n_spin = QSpinBox(self)
        self.n_spin.setRange(self._n_min, self._n_max)
        self.n_spin.setSingleStep(1)
        nfont = self.n_spin.font(); nfont.setPointSize(14); self.n_spin.setFont(nfont)
        self.n_spin.setMinimumWidth(100)
        self.n_spin.setValue(self._n_from_thickness(initial))

        # ---- wiring: one slider, two linked read-outs ----
        self.slider.valueChanged.connect(lambda i: self._apply_thickness(self._to_float(i)))
        self.spin.valueChanged.connect(self._apply_thickness)
        self.n_spin.valueChanged.connect(self._apply_n)

        form.addRow("Slices  ↔  Thickness:", self.slider)
        form.addRow(f"Slice Thickness ({self._unit}):", self.spin)
        form.addRow("Number of slices:", self.n_spin)

        info = QLabel(f"Shortest dimension: {self._L:.3g} {self._unit} — max thickness = "
                      f"{self._max:.3g} {self._unit} (≥ 2 slices).", self)
        info.setStyleSheet("color:#888;")
        form.addRow(info)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    # ---- sync handlers (single slider + two read-outs) ----
    def _apply_thickness(self, f: float) -> None:
        """Slider/thickness changed → sync slider, thickness spin and slice count."""
        if self._syncing:
            return
        self._syncing = True
        try:
            f = max(self._min, min(self._max, float(f)))
            self.slider.setValue(self._to_int(f))
            self.spin.setValue(f)
            self.n_spin.setValue(self._n_from_thickness(f))
        finally:
            self._syncing = False

    def _apply_n(self, n: int) -> None:
        """Slice count changed → derive thickness and sync slider + thickness spin."""
        if self._syncing:
            return
        self._syncing = True
        try:
            n = max(self._n_min, min(self._n_max, int(n)))
            self.n_spin.setValue(n)
            f = max(self._min, min(self._max, self._L / n))
            self.slider.setValue(self._to_int(f))
            self.spin.setValue(f)
        finally:
            self._syncing = False

    def _n_from_thickness(self, f: float) -> int:
        n = int(round(self._L / f)) if f > 0 else self._n_max
        return max(self._n_min, min(self._n_max, n))

    # ---- float<->int slider mapping ----
    def _to_int(self, val: float) -> int:
        """Convert a float thickness to its integer slider position."""
        return max(0, min(self._imax, int(round((float(val) - self._min) / self._step))))

    def _to_float(self, i: int) -> float:
        """Convert an integer slider position back to its float thickness."""
        return self._min + i * self._step

    @staticmethod
    def _decimals(step: float) -> int:
        s = f"{step:.10f}".rstrip("0").split(".")
        return len(s[1]) if len(s) == 2 else 0

    def value(self) -> float:
        """Return the selected slice thickness (mm) as a float."""
        return float(self.spin.value())

    def number_of_slices(self) -> int:
        """Return the selected number of slices."""
        return int(self.n_spin.value())
