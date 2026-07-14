"""Office-style ribbon bar widget for FetoMorph.

Provides a tabbed toolbar where each tab is a horizontal row of large
``QToolButton`` widgets driven by ``QAction`` objects.

Each tab's button row lives inside a horizontal scroller flanked by small
chevron arrows. When the buttons don't all fit, the arrows appear: pressing the
right chevron slides the icons to the left to reveal the hidden tools (and the
left chevron slides them back). This keeps the ribbon a bounded width so a
crowded tab (e.g. Adjustments) can't force the main window wider.
"""

from deps import *
# QScrollArea / QFrame / QPalette are not re-exported by deps.py's __all__.
from PySide6.QtWidgets import QScrollArea, QFrame
from PySide6.QtGui import QPalette

_ARROW_FG = QColor("#e6e6e6")  # light chevron glyph on the slate ribbon


class RibbonBar(QWidget):
    """
    Simple ribbon without groups:
      - Tabs (QTabWidget)
      - Each tab is a horizontally-scrollable row of big QToolButtons bound to
        QActions, with chevron arrows to scroll when the row overflows.
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

        # Slate-gray ribbon with light text and a teal accent on the active tab.
        self.setObjectName("ribbonBar")
        self.setStyleSheet("""
        QWidget#ribbonBar { background: #2b2f36; }

        /* Tab pages / scroll area sit on the same slate background */
        #ribbonBar QWidget[ribbonPage="true"] { background: #2b2f36; }
        #ribbonBar QScrollArea { background: #2b2f36; }
        #ribbonBar QScrollArea > QWidget > QWidget { background: #2b2f36; }

        /* Tab bar */
        #ribbonBar QTabWidget::pane { border: 0; background: #2b2f36; }
        #ribbonBar QTabBar { background: #2b2f36; }
        #ribbonBar QTabBar::tab {
            background: #2b2f36;
            color: #b6bcc6;
            padding: 6px 14px;
            border: 0;
            border-bottom: 2px solid transparent;
        }
        #ribbonBar QTabBar::tab:hover { color: #e6e6e6; background: #3a3f48; }
        #ribbonBar QTabBar::tab:selected {
            color: #ffffff;
            border-bottom: 2px solid #2dd4bf;   /* teal accent */
        }

        /* Tool buttons */
        #ribbonBar QToolButton {
            padding: 6px 10px;
            color: #e6e6e6;
            background: transparent;
            border: 1px solid transparent;
            border-radius: 4px;
        }
        #ribbonBar QToolButton:hover {
            background: #3a3f48;
            border: 1px solid #454b55;
        }
        #ribbonBar QToolButton:pressed { background: #22262c; }
        #ribbonBar QToolButton:disabled { color: #6b7280; }

        /* Chevron scroll arrows */
        #ribbonBar QToolButton#ribbonScrollArrow {
            padding: 0px;
            color: #e6e6e6;
            background: #2b2f36;
            border: 0;
        }
        #ribbonBar QToolButton#ribbonScrollArrow:hover { background: #3a3f48; }
        """)

    # ---------- internal ----------
    def _find_tab_index(self, title: str) -> int:
        for i in range(self._tabs.count()):
            if self._tabs.tabText(i) == title:
                return i
        return -1

    def _make_arrow(self, page: QWidget, arrow, tooltip: str, direction: int) -> QToolButton:
        btn = QToolButton(page)
        btn.setObjectName("ribbonScrollArrow")
        btn.setArrowType(arrow)
        btn.setAutoRaise(True)
        btn.setAutoRepeat(True)
        btn.setToolTip(tooltip)
        btn.setFixedWidth(22)
        # The arrow glyph is painted from the palette, not the QSS `color`, so set
        # a light foreground explicitly to keep it visible on the slate ribbon.
        pal = btn.palette()
        pal.setColor(QPalette.ButtonText, _ARROW_FG)
        pal.setColor(QPalette.WindowText, _ARROW_FG)
        btn.setPalette(pal)
        btn.clicked.connect(lambda _=False, p=page, d=direction: self._scroll_page(p, d))
        return btn

    def _get_tab_widget(self, title: str) -> QWidget:
        idx = self._find_tab_index(title)
        if idx != -1:
            return self._tabs.widget(idx)

        page = QWidget(self)
        page.setObjectName(f"tab::{title}")
        page.setProperty("ribbonPage", True)  # for the slate-background QSS rule
        outer = QHBoxLayout(page)
        outer.setContentsMargins(4, 6, 4, 6)
        outer.setSpacing(2)

        left_btn = self._make_arrow(page, Qt.LeftArrow, "Scroll left", -1)

        area = QScrollArea(page)
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.NoFrame)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        inner = QWidget(area)
        row = QHBoxLayout(inner)
        row.setContentsMargins(4, 0, 4, 0)
        row.setSpacing(8)
        # trailing stretch left-aligns the buttons and has zero minimum width, so
        # it collapses (letting the row overflow + scroll) when space is tight.
        row.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        area.setWidget(inner)

        right_btn = self._make_arrow(page, Qt.RightArrow, "Scroll right", +1)

        outer.addWidget(left_btn)
        outer.addWidget(area, 1)
        outer.addWidget(right_btn)

        page._ribbon_area = area
        page._ribbon_row = row
        page._ribbon_inner = inner
        page._ribbon_left = left_btn
        page._ribbon_right = right_btn

        sb = area.horizontalScrollBar()
        sb.rangeChanged.connect(lambda _mn, _mx, p=page: self._update_arrows(p))
        sb.valueChanged.connect(lambda _v, p=page: self._update_arrows(p))

        self._tabs.addTab(page, title)
        self._update_arrows(page)
        return page

    def _scroll_page(self, page: QWidget, direction: int):
        """Scroll a tab's row by ~80% of its width. +1 slides icons left."""
        area = page._ribbon_area
        sb = area.horizontalScrollBar()
        step = max(1, int(area.viewport().width() * 0.8))
        sb.setValue(sb.value() + direction * step)

    def _update_arrows(self, page: QWidget):
        """Show the chevrons only on overflow; enable each by scroll position."""
        sb = page._ribbon_area.horizontalScrollBar()
        overflow = sb.maximum() > 0
        page._ribbon_left.setVisible(overflow)
        page._ribbon_right.setVisible(overflow)
        if overflow:
            page._ribbon_left.setEnabled(sb.value() > sb.minimum())
            page._ribbon_right.setEnabled(sb.value() < sb.maximum())

    def resizeEvent(self, event):  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        # Overflow state depends on viewport width; refresh after the tabs resize.
        for i in range(self._tabs.count()):
            self._update_arrows(self._tabs.widget(i))

    # ---------- public API ----------
    def add_tab(self, title: str):
        self._get_tab_widget(title)

    def add_action(self, tab_title: str, action):
        """Add a QAction as a big tool button on a tab's scrollable row."""
        page = self._get_tab_widget(tab_title)
        row: QHBoxLayout = page._ribbon_row

        btn = QToolButton(page._ribbon_inner)
        btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        btn.setIconSize(self._icon_size)
        btn.setDefaultAction(action)  # keeps text, icon, tooltip, enabled state in sync

        # insert before the trailing stretch so buttons stay left-aligned
        row.insertWidget(row.count() - 1, btn)
        self._update_arrows(page)

    def set_icon_size(self, size: QSize):
        self._icon_size = size
        # update existing tool buttons (not the chevron arrows, which live on the page)
        for i in range(self._tabs.count()):
            page = self._tabs.widget(i)
            for child in page._ribbon_inner.findChildren(QToolButton):
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
