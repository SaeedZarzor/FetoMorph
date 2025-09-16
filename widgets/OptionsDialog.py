from deps import *

class ProcessingOptionsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Processing Options")
        self.setModal(True)

        # --- Unify color ---
        self.chk_unify = QtWidgets.QCheckBox("Unify colored regions")
        self.btn_color = QtWidgets.QPushButton("Choose color…")
        self.lbl_preview = QtWidgets.QLabel()
        self.lbl_preview.setFixedSize(48, 24)
        self.lbl_preview.setFrameShape(QtWidgets.QFrame.Shape.Box)

        row_unify = QtWidgets.QHBoxLayout()
        row_unify.addWidget(self.btn_color)
        row_unify.addWidget(QtWidgets.QLabel("Preview:"))
        row_unify.addWidget(self.lbl_preview)
        row_unify.addStretch(1)

        # --- Scale bar ---
        self.chk_scalebar = QtWidgets.QCheckBox("Add scale bar")

        # --- Smooth options ---
        self.cmb_smooth = QtWidgets.QComboBox()
        self.cmb_smooth.addItems(["None", "Gaussian", "Median", "Bilateral"])
        self.slider_strength = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_strength.setRange(1, 25)
        self.slider_strength.setValue(5)

        row_smooth = QtWidgets.QHBoxLayout()
        row_smooth.addWidget(QtWidgets.QLabel("Smooth:"))
        row_smooth.addWidget(self.cmb_smooth)
        row_smooth.addWidget(QtWidgets.QLabel("Strength:"))
        row_smooth.addWidget(self.slider_strength)

        # --- Buttons ---
        btn_defaults = QtWidgets.QPushButton("Defaults")
        btn_ok = QtWidgets.QPushButton("OK")
        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_defaults.clicked.connect(self.set_defaults)
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)

        row_btn = QtWidgets.QHBoxLayout()
        row_btn.addWidget(btn_defaults)
        row_btn.addStretch(1)
        row_btn.addWidget(btn_cancel)
        row_btn.addWidget(btn_ok)

        # --- Layout ---
        vbox = QtWidgets.QVBoxLayout(self)
        vbox.addWidget(self.chk_unify)
        vbox.addLayout(row_unify)
        vbox.addSpacing(8)
        vbox.addWidget(self.chk_scalebar)
        vbox.addSpacing(8)
        vbox.addLayout(row_smooth)
        vbox.addSpacing(12)
        vbox.addLayout(row_btn)

        # default color
        self._bgr = (0, 0, 255)  # red
        self._update_preview()

        self.chk_unify.toggled.connect(self._toggle_unify)
        self.btn_color.clicked.connect(self._choose_color)

        # Apply defaults on startup
        self.set_defaults()

    # --- Defaults handler ---
    def set_defaults(self):
        self._bgr = (255, 0, 0)             # Blue in BGR
        self._update_preview()
        self.chk_unify.setChecked(True)
        self.chk_scalebar.setChecked(True)
        self.cmb_smooth.setCurrentText("Median")
        self.slider_strength.setValue(5)

    # --- Internals ---
    def _choose_color(self):
        r, g, b = self._bgr[2], self._bgr[1], self._bgr[0]
        start = QtGui.QColor(r, g, b)
        color = QtWidgets.QColorDialog.getColor(start, self, "Select unify color")
        if color.isValid():
            self._bgr = (color.blue(), color.green(), color.red())
            self._update_preview()

    def _update_preview(self):
        pix = QtGui.QPixmap(self.lbl_preview.size())
        pix.fill(QtGui.QColor(self._bgr[2], self._bgr[1], self._bgr[0]))
        self.lbl_preview.setPixmap(pix)

    def _toggle_unify(self, enabled: bool):
        self.btn_color.setEnabled(enabled)
        self.lbl_preview.setEnabled(enabled)

    # --- Public getters ---
    def unify_color(self) -> Optional[Tuple[int,int,int]]:
        return self._bgr if self.chk_unify.isChecked() else None

    def add_scale_bar(self) -> bool:
        return self.chk_scalebar.isChecked()

    def smooth_kind(self) -> str:
        return self.cmb_smooth.currentText().lower()

    def smooth_strength(self) -> int:
        return self.slider_strength.value()

