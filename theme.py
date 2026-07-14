"""Application-wide slate/teal theme for FetoMorph.

Applies one consistent look to every window and dialog by combining:
  1. the Fusion style (so the palette is actually honoured — the native macOS
     style ignores custom palettes for most widgets), plus
  2. a dark "slate" QPalette (handles backgrounds/text everywhere, including
     QMessageBox / QInputDialog / menus that carry no explicit stylesheet), plus
  3. a thin global QSS layer for accents (teal selection/underline, hover
     states, borders, slim scrollbars).

Call :func:`apply_theme(app)` once, right after constructing the QApplication.
The colours here intentionally match the ribbon's own scoped stylesheet in
``ribbon.py`` so the two never drift.
"""

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QColor, QPalette
from PySide6.QtCore import Qt


# --- Palette (kept in sync with ribbon.py) ---------------------------------
WINDOW = "#2b2f36"        # slate window background (matches the ribbon)
BASE = "#23272e"          # darker fields: inputs, lists, tables, console
ALT_BASE = "#2b2f36"
TEXT = "#e6e6e6"          # light foreground
DISABLED_TEXT = "#6b7280"
ACCENT = "#2dd4bf"        # teal accent
ACCENT_TEXT = "#10141a"   # text drawn on top of the teal accent
BORDER = "#454b55"
HOVER = "#3a3f48"
PRESSED = "#22262c"
TOOLTIP_BG = "#23272e"


def build_palette() -> QPalette:
    """A dark slate palette used app-wide."""
    pal = QPalette()
    win = QColor(WINDOW)
    base = QColor(BASE)
    text = QColor(TEXT)
    disabled = QColor(DISABLED_TEXT)

    pal.setColor(QPalette.Window, win)
    pal.setColor(QPalette.WindowText, text)
    pal.setColor(QPalette.Base, base)
    pal.setColor(QPalette.AlternateBase, QColor(ALT_BASE))
    pal.setColor(QPalette.Text, text)
    pal.setColor(QPalette.Button, win)
    pal.setColor(QPalette.ButtonText, text)
    pal.setColor(QPalette.BrightText, QColor("#ff5555"))
    pal.setColor(QPalette.ToolTipBase, QColor(TOOLTIP_BG))
    pal.setColor(QPalette.ToolTipText, text)
    pal.setColor(QPalette.PlaceholderText, disabled)
    pal.setColor(QPalette.Link, QColor(ACCENT))
    pal.setColor(QPalette.LinkVisited, QColor(ACCENT))
    pal.setColor(QPalette.Highlight, QColor(ACCENT))
    pal.setColor(QPalette.HighlightedText, QColor(ACCENT_TEXT))

    # Disabled-state roles so greyed widgets stay legible.
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        pal.setColor(QPalette.Disabled, role, disabled)
    pal.setColor(QPalette.Disabled, QPalette.Highlight, QColor(HOVER))
    pal.setColor(QPalette.Disabled, QPalette.HighlightedText, disabled)
    return pal


# --- Accent QSS layer (backgrounds/text come from the palette) --------------
def _qss() -> str:
    return f"""
    QToolTip {{
        background: {TOOLTIP_BG};
        color: {TEXT};
        border: 1px solid {BORDER};
    }}

    QGroupBox {{
        border: 1px solid {BORDER};
        border-radius: 4px;
        margin-top: 8px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 8px;
        padding: 0 4px;
        color: {TEXT};
    }}

    QPushButton {{
        background: {WINDOW};
        color: {TEXT};
        border: 1px solid {BORDER};
        border-radius: 4px;
        padding: 4px 12px;
    }}
    QPushButton:hover {{ background: {HOVER}; }}
    QPushButton:pressed {{ background: {PRESSED}; }}
    QPushButton:default {{ border: 1px solid {ACCENT}; }}
    QPushButton:disabled {{ color: {DISABLED_TEXT}; border-color: {PRESSED}; }}

    QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background: {BASE};
        color: {TEXT};
        border: 1px solid {BORDER};
        border-radius: 4px;
        padding: 3px 6px;
        selection-background-color: {ACCENT};
        selection-color: {ACCENT_TEXT};
    }}
    QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus,
    QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
        border: 1px solid {ACCENT};
    }}
    QComboBox QAbstractItemView {{
        background: {BASE};
        color: {TEXT};
        selection-background-color: {ACCENT};
        selection-color: {ACCENT_TEXT};
        border: 1px solid {BORDER};
    }}

    QCheckBox, QRadioButton, QLabel {{ color: {TEXT}; background: transparent; }}

    QTabWidget::pane {{ border: 1px solid {BORDER}; }}
    QTabBar::tab {{
        background: {WINDOW};
        color: #b6bcc6;
        padding: 6px 14px;
        border: 0;
        border-bottom: 2px solid transparent;
    }}
    QTabBar::tab:hover {{ color: {TEXT}; background: {HOVER}; }}
    QTabBar::tab:selected {{ color: #ffffff; border-bottom: 2px solid {ACCENT}; }}

    QMenuBar {{ background: {WINDOW}; color: {TEXT}; }}
    QMenuBar::item:selected {{ background: {HOVER}; }}
    QMenu {{ background: {BASE}; color: {TEXT}; border: 1px solid {BORDER}; }}
    QMenu::item:selected {{ background: {ACCENT}; color: {ACCENT_TEXT}; }}
    QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 8px; }}

    QToolBar {{ background: {WINDOW}; border: 0; spacing: 4px; }}
    QStatusBar {{ background: {WINDOW}; color: {TEXT}; }}
    QStatusBar::item {{ border: 0; }}

    QHeaderView::section {{
        background: {WINDOW};
        color: {TEXT};
        border: 0;
        border-right: 1px solid {BORDER};
        border-bottom: 1px solid {BORDER};
        padding: 4px 6px;
    }}
    QTableView, QTreeView, QListView {{
        background: {BASE};
        color: {TEXT};
        alternate-background-color: {ALT_BASE};
        gridline-color: {BORDER};
        selection-background-color: {ACCENT};
        selection-color: {ACCENT_TEXT};
        border: 1px solid {BORDER};
    }}

    QProgressBar {{
        background: {BASE};
        color: {TEXT};
        border: 1px solid {BORDER};
        border-radius: 4px;
        text-align: center;
    }}
    QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}

    QScrollBar:vertical {{ background: {WINDOW}; width: 12px; margin: 0; }}
    QScrollBar:horizontal {{ background: {WINDOW}; height: 12px; margin: 0; }}
    QScrollBar::handle {{ background: {BORDER}; border-radius: 5px; min-height: 24px; min-width: 24px; }}
    QScrollBar::handle:hover {{ background: #55606c; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

    QSplitter::handle {{ background: {BORDER}; }}
    """


def apply_theme(app: QApplication) -> None:
    """Apply the slate/teal theme to the whole application."""
    app.setStyle("Fusion")  # honours the palette on every platform (esp. macOS)
    app.setPalette(build_palette())
    app.setStyleSheet(_qss())


# --- Light override (for individual windows that should stay white) ----------
LIGHT_WINDOW = "#f4f5f7"
LIGHT_BASE = "#ffffff"
LIGHT_TEXT = "#1a1a1a"
LIGHT_DISABLED = "#9aa0a6"
LIGHT_BORDER = "#c4c8ce"
LIGHT_HOVER = "#e7eaee"
LIGHT_PRESSED = "#dcdfe4"


def build_light_palette() -> QPalette:
    """A light palette for windows that opt out of the dark theme."""
    pal = QPalette()
    win = QColor(LIGHT_WINDOW)
    base = QColor(LIGHT_BASE)
    text = QColor(LIGHT_TEXT)
    disabled = QColor(LIGHT_DISABLED)

    pal.setColor(QPalette.Window, win)
    pal.setColor(QPalette.WindowText, text)
    pal.setColor(QPalette.Base, base)
    pal.setColor(QPalette.AlternateBase, QColor(LIGHT_WINDOW))
    pal.setColor(QPalette.Text, text)
    pal.setColor(QPalette.Button, win)
    pal.setColor(QPalette.ButtonText, text)
    pal.setColor(QPalette.BrightText, QColor("#c81e1e"))
    pal.setColor(QPalette.ToolTipBase, base)
    pal.setColor(QPalette.ToolTipText, text)
    pal.setColor(QPalette.PlaceholderText, disabled)
    pal.setColor(QPalette.Link, QColor(ACCENT))
    pal.setColor(QPalette.Highlight, QColor(ACCENT))
    pal.setColor(QPalette.HighlightedText, QColor(ACCENT_TEXT))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        pal.setColor(QPalette.Disabled, role, disabled)
    return pal


def _light_qss() -> str:
    # Unscoped rules set on the target widget itself; they cascade only to that
    # widget's children and take precedence over the app-wide dark stylesheet.
    return f"""
    QDialog, QWidget {{ background: {LIGHT_WINDOW}; color: {LIGHT_TEXT}; }}
    QLabel, QCheckBox, QRadioButton, QGroupBox {{
        color: {LIGHT_TEXT}; background: transparent;
    }}
    QGroupBox {{ border: 1px solid {LIGHT_BORDER}; border-radius: 4px; margin-top: 8px; }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}

    QPushButton {{
        background: {LIGHT_BASE};
        color: {LIGHT_TEXT};
        border: 1px solid {LIGHT_BORDER};
        border-radius: 4px;
        padding: 4px 12px;
    }}
    QPushButton:hover {{ background: {LIGHT_HOVER}; }}
    QPushButton:pressed {{ background: {LIGHT_PRESSED}; }}
    QPushButton:default {{ border: 1px solid {ACCENT}; }}
    QPushButton:disabled {{ color: {LIGHT_DISABLED}; }}

    QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background: {LIGHT_BASE};
        color: {LIGHT_TEXT};
        border: 1px solid {LIGHT_BORDER};
        border-radius: 4px;
        padding: 3px 6px;
        selection-background-color: {ACCENT};
        selection-color: {ACCENT_TEXT};
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
        border: 1px solid {ACCENT};
    }}
    QComboBox QAbstractItemView {{
        background: {LIGHT_BASE}; color: {LIGHT_TEXT};
        selection-background-color: {ACCENT}; selection-color: {ACCENT_TEXT};
        border: 1px solid {LIGHT_BORDER};
    }}
    QSlider::groove:horizontal {{ height: 4px; background: {LIGHT_BORDER}; border-radius: 2px; }}
    QSlider::handle:horizontal {{
        background: {ACCENT}; width: 14px; margin: -6px 0; border-radius: 7px;
    }}
    """


def apply_light_theme(widget) -> None:
    """Make a single window/dialog light (white) despite the dark app theme.

    Sets a light palette (for palette-driven bits) and a light stylesheet on the
    widget itself (which cascades to its children and overrides the app-wide
    dark QSS for this subtree).
    """
    widget.setPalette(build_light_palette())
    widget.setStyleSheet(_light_qss())
