"""Dockable region-of-interest selector widget.

Presents a checkable list of integer region labels, bulk-action buttons
(select all, clear, invert, defaults), and a quick-entry text field so
the user can type comma-separated label IDs directly.
"""

import re
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
                                QLabel, QListWidget, QListWidgetItem,
                                QLineEdit, QPushButton)

class RegionsDock(QDockWidget):
    """Dock widget for selecting which segmentation region labels to include.

    Signals:
        applied: Emitted with the set of selected label ints on Apply.
        closed: Emitted when the dock is closed.
    """

    applied = Signal(set)   # emits the selected set on Apply
    closed = Signal()       # emits when dock is closed

    def __init__(self, parent=None, title="Regions of Interest"):
        """Initialise the regions dock.

        Args:
            parent: Parent widget.
            title: Window title for the dock.
        """
        super().__init__(title, parent)
        self.setObjectName("RegionsDock")
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        self._w = QWidget()
        self.setWidget(self._w)
        self._w.setLayout(QVBoxLayout())
        lay = self._w.layout()

        lay.addWidget(QLabel("Tick labels to include (double-click toggles).\n"
                             "You can also type a comma-separated list below."))

        # list of labels (user-checkable)
        self.lst = QListWidget()
        self.lst.setSelectionMode(QListWidget.NoSelection)
        self.lst.itemChanged.connect(self._on_item_changed)
        lay.addWidget(self.lst)

        # quick entry
        quick = QHBoxLayout()
        self.quick_edit = QLineEdit()
        self.quick_edit.setPlaceholderText("e.g. 2,3,4,5,6,11,12,13,14,15,17")
        btn_apply_typed = QPushButton("Apply typed list")
        btn_apply_typed.clicked.connect(self._apply_typed)
        quick.addWidget(self.quick_edit, 1)
        quick.addWidget(btn_apply_typed)
        lay.addLayout(quick)

        # bulk actions
        bulk = QHBoxLayout()
        self.btn_all = QPushButton("Select All")
        self.btn_none = QPushButton("Clear All")
        self.btn_inv = QPushButton("Invert")
        self.btn_def = QPushButton("Defaults")
        self.btn_all.clicked.connect(self._select_all)
        self.btn_none.clicked.connect(self._clear_all)
        self.btn_inv.clicked.connect(self._invert)
        self.btn_def.clicked.connect(self._apply_defaults)
        for b in (self.btn_all, self.btn_none, self.btn_inv, self.btn_def):
            bulk.addWidget(b)
        lay.addLayout(bulk)

        # commit row
        commit = QHBoxLayout()
        self.btn_apply = QPushButton("Apply")
        self.btn_close = QPushButton("Close")
        self.btn_apply.clicked.connect(self._emit_apply)
        self.btn_close.clicked.connect(self.close)
        commit.addStretch(1)
        commit.addWidget(self.btn_apply)
        commit.addWidget(self.btn_close)
        lay.addLayout(commit)

        # state
        self._defaults = set()
        self._label_to_color = {}  # lab -> QColor
        self._block_item_changed = False
        self._live_selected = set()  # mirrors current check states
        
        
         # ---------- public API ----------
    def populate(self, labels_available: list[int], selected: set[int], label_to_color: dict[int, 'QColor']|None=None):
        """Fill the list with checkable items and initial selection."""
        self._block_item_changed = True
        self.lst.clear()
        self._label_to_color = label_to_color or {}
        self._live_selected = set(selected)
        for lab in sorted(set(int(x) for x in labels_available)):
            it = QListWidgetItem(str(lab))
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if lab in selected else Qt.Unchecked)
            # optional icon (color square)
            if lab in self._label_to_color:
                it.setIcon(self.parent()._color_square_icon(self._label_to_color[lab]))
            self.lst.addItem(it)
        self._block_item_changed = False

    def set_defaults(self, defaults: set[int]):
        """Store the default label set used by the Defaults button."""
        self._defaults = set(int(x) for x in defaults or set())

    def current_selection(self) -> set[int]:
        """Return the set of currently checked label integers."""
        return {int(self.lst.item(i).text())
                for i in range(self.lst.count())
                if self.lst.item(i).checkState() == Qt.Checked}

    # ---------- internals ----------
    def _parse_labels(self, text: str) -> set[int]:
        return {int(t) for t in re.findall(r"\d+", text or "")}

    def _set_checks_from_set(self, s: set[int]):
        self._block_item_changed = True
        have = {int(self.lst.item(i).text()) for i in range(self.lst.count())}
        # update existing rows
        for i in range(self.lst.count()):
            lab = int(self.lst.item(i).text())
            self.lst.item(i).setCheckState(Qt.Checked if lab in s else Qt.Unchecked)
        # optionally add extras not yet present
        extras = sorted(s - have)
        for lab in extras:
            it = QListWidgetItem(str(lab), self.lst)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked)
        self._block_item_changed = False
        self._live_selected = set(self.current_selection())

    def _on_item_changed(self, item: QListWidgetItem):
        if self._block_item_changed:
            return
        lab = int(item.text())
        if item.checkState() == Qt.Checked:
            self._live_selected.add(lab)
        else:
            self._live_selected.discard(lab)
        # live preview hook to parent (if desired)
        parent = self.parent()
        if hasattr(parent, "show_nifti_slice") and getattr(parent, "current_kind", None) == "nifti":
            idx = int(getattr(parent, "slice_slider", None).value()) if hasattr(parent, "slice_slider") else 0
            parent.nifti_selected_regions = set(self._live_selected)
            parent.show_nifti_slice(idx)

    def _apply_typed(self):
        self._set_checks_from_set(self._parse_labels(self.quick_edit.text()))

    def _select_all(self):
        self._block_item_changed = True
        for i in range(self.lst.count()):
            self.lst.item(i).setCheckState(Qt.Checked)
        self._block_item_changed = False
        self._live_selected = set(self.current_selection())

    def _clear_all(self):
        self._block_item_changed = True
        for i in range(self.lst.count()):
            self.lst.item(i).setCheckState(Qt.Unchecked)
        self._block_item_changed = False
        self._live_selected.clear()

    def _invert(self):
        self._block_item_changed = True
        for i in range(self.lst.count()):
            it = self.lst.item(i)
            it.setCheckState(Qt.Unchecked if it.checkState() == Qt.Checked else Qt.Checked)
        self._block_item_changed = False
        self._live_selected = set(self.current_selection())

    def _apply_defaults(self):
        avail = {int(self.lst.item(i).text()) for i in range(self.lst.count())}
        self._set_checks_from_set(self._defaults & avail)

    def _emit_apply(self):
        self.applied.emit(self.current_selection())

    def closeEvent(self, e):
        super().closeEvent(e)
        self.closed.emit()

