"""Image browser dialog widget.

Displays thumbnail previews of all images in a folder and lets the
user select one.  Consistent with the dark-themed FetoMorph dialogs.
"""

from deps import *


class ImageBrowserDialog(QDialog):
    """Dialog that shows a thumbnail grid of images from a directory.

    The user clicks a thumbnail to select it, then presses Ok (or
    double-clicks) to confirm.
    """

    EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif")
    THUMB_SIZE = 160

    def __init__(self, parent=None, folder: str = "", title: str = "Select Image"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(700, 500)
        self._selected_path: str | None = None

        layout = QVBoxLayout(self)

        # thumbnail grid using QListWidget in icon mode
        self.list = QListWidget(self)
        self.list.setViewMode(QListWidget.IconMode)
        self.list.setIconSize(QSize(self.THUMB_SIZE, self.THUMB_SIZE))
        self.list.setResizeMode(QListWidget.Adjust)
        self.list.setSpacing(10)
        self.list.setMovement(QListWidget.Static)
        self.list.setSelectionMode(QListWidget.SingleSelection)
        self.list.setWordWrap(True)
        self.list.setStyleSheet("""
        QListWidget {background: #222; border: none;}
        QListWidget::item {color: #ccc; padding: 4px; border-radius: 6px;}
        QListWidget::item:selected {background: #2aa1ff; color: #fff;}
        QListWidget::item:hover {background: #333;}
        """)
        self.list.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self.list)

        # ok / cancel
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        # populate thumbnails
        self._load_images(folder)

    def _load_images(self, folder: str):
        if not os.path.isdir(folder):
            return
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(self.EXTS))
        for fname in files:
            path = os.path.join(folder, fname)
            pm = QPixmap(path)
            if pm.isNull():
                continue
            thumb = pm.scaled(self.THUMB_SIZE, self.THUMB_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            item = QListWidgetItem(QIcon(thumb), fname)
            item.setData(Qt.UserRole, path)
            item.setSizeHint(QSize(self.THUMB_SIZE + 20, self.THUMB_SIZE + 30))
            self.list.addItem(item)

    def _on_double_click(self, item):
        self._selected_path = item.data(Qt.UserRole)
        self.accept()

    def accept(self):
        sel = self.list.currentItem()
        if sel:
            self._selected_path = sel.data(Qt.UserRole)
        super().accept()

    def selected_path(self) -> str | None:
        """Return the full path of the selected image, or None."""
        return self._selected_path
