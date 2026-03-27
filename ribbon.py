"""Office-style ribbon bar widget for FetoMorph.

Provides a tabbed toolbar where each tab is a horizontal row of large
``QToolButton`` widgets driven by ``QAction`` objects.
"""

from deps import *


class RibbonBar(QWidget):
    """
    Simple ribbon without groups:
      - Tabs (QTabWidget)
      - Each tab is just a horizontal row of big QToolButtons bound to QActions.
    Usage:
        ribbon.add_action("Import", some_qaction)
    """
    def __init__(self, parent=None, icon_size: QSize = QSize(36, 36)):
        super().__init__(parent)
        self._tabs = QTabWidget(self)
        self._tabs.setDocumentMode(True)
        self._tabs.setTabPosition(QTabWidget.North)
        self._tabs.setMovable(False)

        self._icon_size = icon_size

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self._tabs)

        self.setStyleSheet("""
        QToolButton { padding: 6px 10px; }
        """)

    # ---------- internal ----------
    def _find_tab_index(self, title: str) -> int:
        for i in range(self._tabs.count()):
            if self._tabs.tabText(i) == title:
                return i
        return -1

    def _get_tab_widget(self, title: str) -> QWidget:
        idx = self._find_tab_index(title)
        if idx != -1:
            return self._tabs.widget(idx)
        page = QWidget(self)
        page.setObjectName(f"tab::{title}")
        lay = QHBoxLayout(page)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(8)
        # keep items left-justified with a stretch on the right
        lay.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        self._tabs.addTab(page, title)
        return page

    # ---------- public API ----------
    def add_tab(self, title: str):
        self._get_tab_widget(title)

    def add_action(self, tab_title: str, action):
        """Add a QAction as a big tool button on a tab."""
        page = self._get_tab_widget(tab_title)
        lay: QHBoxLayout = page.layout()  # type: ignore

        # insert before the trailing stretch so buttons stay left-aligned
        # remove the stretch, add button, then re-add stretch
        stretch_item = lay.takeAt(lay.count() - 1)

        btn = QToolButton(page)
        btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        btn.setIconSize(self._icon_size)
        btn.setDefaultAction(action)  # keeps text, icon, tooltip, enabled state in sync
        lay.addWidget(btn)

        if stretch_item is not None:
            lay.addItem(stretch_item)

    def set_icon_size(self, size: QSize):
        self._icon_size = size
        # update existing buttons
        for i in range(self._tabs.count()):
            page = self._tabs.widget(i)
            for child in page.findChildren(QToolButton):
                child.setIconSize(size)


    def set_current_tab(self, title: str):
        """Switch to a tab by title (no-op if it doesn't exist)."""
        for i in range(self._tabs.count()):
            if self._tabs.tabText(i) == title:
                self._tabs.setCurrentIndex(i)
                break

    def remove_tab(self, title: str):
        """(Optional) Remove a tab by title."""
        for i in range(self._tabs.count()):
            if self._tabs.tabText(i) == title:
                self._tabs.removeTab(i)
                break
