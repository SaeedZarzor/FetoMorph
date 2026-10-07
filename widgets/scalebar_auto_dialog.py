"""Confirmation dialog for the automatically read scale bar.

Shows what :func:`helpers.scalebar_reader.read_scalebar` found -- the bar it
measured and the label it read, outlined on a crop of the image -- and lets
the user accept it, correct the length or unit, or switch to drawing the
line by hand.
"""

from deps import *

from PySide6.QtWidgets import QCheckBox

from helpers.scalebar_reader import (
    ScalebarContours, ScalebarReading, annotate_reading,
)


class ScalebarAutoDialog(QDialog):
    """Review and accept an automatically detected scale bar.

    The length and unit stay editable: the reader can be confident about the
    bar it measured while misreading a digit, and correcting the number is
    quicker than re-measuring the bar by hand.
    """

    #: Returned by :meth:`outcome` when the user asked to draw the line instead.
    MANUAL = "manual"
    ACCEPT = "accept"

    def __init__(self, reading: ScalebarReading, image_bgr, parent=None,
                 contours: ScalebarContours | None = None,
                 current_threshold: float | None = None):
        """Build the dialog.

        Args:
            reading: The detected bar and label.
            image_bgr: The image it was read from, in BGR.
            parent: Parent widget.
            contours: Contour area the drawn bar adds to the image, used to
                offer excluding it from measurements. ``None`` hides the offer.
            current_threshold: The contour-area threshold in force (mm²),
                shown for comparison.
        """
        super().__init__(parent)
        self._contours = contours
        self._current_threshold = current_threshold
        self.setWindowTitle("Scale From Scalebar (automatic)")
        self.setModal(True)
        self._reading = reading
        self._outcome = self.ACCEPT

        lay = QVBoxLayout(self)

        preview = self._build_preview(image_bgr, reading)
        if preview is not None:
            lay.addWidget(preview)

        form = QFormLayout()
        form.addRow("Bar length:", QLabel(f"{reading.bar_px:.0f} px"))
        form.addRow("Label read:", QLabel(f"“{reading.text}”"))

        self.len_spin = QDoubleSpinBox(self)
        self.len_spin.setRange(1e-9, 1e12)
        self.len_spin.setDecimals(4)
        self.len_spin.setValue(float(reading.value))
        self.len_spin.setMinimumWidth(140)

        self.unit_box = QComboBox(self)
        self.unit_box.setEditable(True)
        self.unit_box.addItems(["mm", "µm", "cm", "m"])
        idx = self.unit_box.findText(reading.unit)
        self.unit_box.setCurrentIndex(idx if idx >= 0 else 0)
        if idx < 0:
            self.unit_box.setEditText(reading.unit)

        form.addRow("Real-world length:", self.len_spin)
        form.addRow("Unit:", self.unit_box)

        self.lbl_scale = QLabel()
        form.addRow("Resulting scale:", self.lbl_scale)
        lay.addLayout(form)

        self.lbl_conf = QLabel()
        self.lbl_conf.setWordWrap(True)
        lay.addWidget(self.lbl_conf)

        # The bar and its label are ink on the slice, so the segmentation sees
        # them as contours and every area/perimeter measurement includes them.
        # Offer to raise the contour-area threshold just past the bar.
        self.chk_exclude = QCheckBox("Exclude the scalebar from measurements", self)
        self.chk_exclude.setToolTip(
            "Raise the contour-area threshold just above the drawn bar so it is "
            "filtered out of area, perimeter and sulci measurements.")
        self.lbl_threshold = QLabel()
        self.lbl_threshold.setWordWrap(True)
        if contours is not None:
            self.chk_exclude.setChecked(contours.is_safe())
            if not contours.is_safe():
                self.chk_exclude.setEnabled(False)
            lay.addWidget(self.chk_exclude)
            lay.addWidget(self.lbl_threshold)
            self.chk_exclude.toggled.connect(self._refresh)
        else:
            self.chk_exclude.setChecked(False)
            self.chk_exclude.hide()
            self.lbl_threshold.hide()

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self._manual_btn = btns.addButton("Draw Line Instead…",
                                          QDialogButtonBox.ActionRole)
        self._manual_btn.setToolTip(
            "Ignore the detected bar and measure it by hand instead.")
        self._manual_btn.clicked.connect(self._choose_manual)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

        self.len_spin.valueChanged.connect(self._refresh)
        self.unit_box.currentTextChanged.connect(self._refresh)
        self._refresh()

    # -- construction helpers ------------------------------------------------

    def _build_preview(self, image_bgr, reading: ScalebarReading):
        """Return a label showing the outlined bar, or ``None`` if unavailable."""
        try:
            import numpy as np

            marked = annotate_reading(image_bgr, reading)
            # Crop generously around the bar and its label so the user sees the
            # measurement in context without the whole slice shrinking it away.
            bx, by, bw, bh = reading.bar_box
            boxes = [reading.bar_box] + ([reading.text_box] if reading.text_box else [])
            x0 = min(b[0] for b in boxes)
            y0 = min(b[1] for b in boxes)
            x1 = max(b[0] + b[2] for b in boxes)
            y1 = max(b[1] + b[3] for b in boxes)
            padx = max(20, int(0.35 * (x1 - x0)))
            pady = max(16, int(0.9 * (y1 - y0)))
            h, w = marked.shape[:2]
            crop = marked[max(0, y0 - pady):min(h, y1 + pady),
                          max(0, x0 - padx):min(w, x1 + padx)]
            if crop.size == 0:
                return None
            crop = np.ascontiguousarray(crop[:, :, ::-1])  # BGR → RGB
            ch, cw = crop.shape[:2]
            qimg = QImage(crop.data, cw, ch, 3 * cw, QImage.Format_RGB888).copy()
            pix = QPixmap.fromImage(qimg)
            target = 460
            if cw < target:
                pix = pix.scaledToWidth(target, Qt.FastTransformation)
            elif cw > 2 * target:
                pix = pix.scaledToWidth(2 * target, Qt.SmoothTransformation)

            label = QLabel(self)
            label.setPixmap(pix)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("background:#111; padding:6px;")
            return label
        except Exception:
            # A preview is a convenience; never block the calibration on it.
            return None

    # -- behaviour -----------------------------------------------------------

    def _choose_manual(self) -> None:
        self._outcome = self.MANUAL
        self.reject()

    def _refresh(self) -> None:
        """Recompute the derived scale shown under the inputs."""
        unit = (self.unit_box.currentText() or "mm").strip()
        value = float(self.len_spin.value())
        if value > 0 and self._reading.bar_px > 0:
            per_px = value / self._reading.bar_px
            self.lbl_scale.setText(f"{per_px:.6f} {unit}/pixel "
                                   f"({self._reading.bar_px / value:.3f} px/{unit})")
        else:
            self.lbl_scale.setText("—")

        conf = self._reading.confidence
        if conf >= 0.6:
            note = "Label read clearly."
        elif conf >= 0.5:
            note = "Label read with moderate confidence — check the number above."
        else:
            note = "Label was hard to read — check the number above before accepting."
        self.lbl_conf.setText(f"Match confidence {conf:.0%}. {note}")
        self._refresh_threshold()

    def _refresh_threshold(self) -> None:
        """Describe the contour-area threshold the exclusion would apply."""
        if self._contours is None:
            return
        new_mm2 = self.contour_threshold_mm2()
        mm_px = self._mm_per_px()
        bar_mm2 = self._contours.scalebar_max_px2 * mm_px * mm_px
        brain_mm2 = self._contours.largest_other_px2 * mm_px * mm_px
        current = self._current_threshold

        if not self._contours.is_safe():
            self.lbl_threshold.setText(
                f"The scalebar covers {bar_mm2:.1f} mm², too close to the largest "
                f"other contour ({brain_mm2:.1f} mm²) to filter out safely. "
                "Leaving the threshold unchanged.")
            return
        if new_mm2 is None:
            was = f" (currently {current:.2f} mm²)" if current is not None else ""
            self.lbl_threshold.setText(
                f"Scalebar covers {bar_mm2:.1f} mm² across "
                f"{self._contours.count} contour(s) and will be measured as "
                f"part of the slice{was}.")
            return
        was = f"{current:.2f}" if current is not None else "—"
        self.lbl_threshold.setText(
            f"Contour-area threshold {was} → {new_mm2:.2f} mm², which drops the "
            f"{bar_mm2:.1f} mm² bar and keeps the {brain_mm2:.1f} mm² brain.")

    def _mm_per_px(self) -> float:
        """Pixel size in mm implied by the values currently in the dialog."""
        from helpers.scalebar_reader import _UNIT_MM
        unit = (self.unit_box.currentText() or "mm").strip()
        value = float(self.len_spin.value())
        if value <= 0 or self._reading.bar_px <= 0:
            return 0.0
        return (value * _UNIT_MM.get(unit, 1.0)) / self._reading.bar_px

    # -- results -------------------------------------------------------------

    def contour_threshold_mm2(self) -> float | None:
        """The contour-area threshold to apply, or ``None`` to leave it alone.

        Tracks the length and unit currently entered, so correcting the label
        also corrects the threshold.
        """
        if self._contours is None or not self.chk_exclude.isChecked():
            return None
        if not self._contours.is_safe():
            return None
        mm_px = self._mm_per_px()
        if mm_px <= 0:
            return None
        return self._contours.threshold_px2() * mm_px * mm_px

    def outcome(self) -> str:
        """``ACCEPT`` or ``MANUAL`` — what the user chose on rejection."""
        return self._outcome

    def values(self) -> tuple[float, str]:
        """Return ``(px_per_unit, unit)`` for the confirmed reading."""
        value = float(self.len_spin.value())
        unit = (self.unit_box.currentText() or "mm").strip()
        if value <= 0:
            raise ValueError("Real length must be > 0.")
        return self._reading.bar_px / value, unit
