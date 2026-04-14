"""Central dependency imports for FetoMorph.

Every module in the project does ``from deps import *`` so that Qt, VTK,
NumPy, OpenCV and other heavy libraries are imported once and made
available everywhere under short, consistent names.
"""

# ======================= Standard Library =======================
import logging
import math
import os
import pathlib
import re
import shutil
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Sequence, Tuple, Union

# ======================= Qt - Core =======================
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import (
    QEventLoop, QObject, QPoint, QRect, QRectF,
    QSettings, QSize, QTimer, QUrl, Qt, Signal,
)

# ======================= Qt - GUI =======================
from PySide6.QtGui import (
    QAction, QColor, QDesktopServices, QFont, QIcon, QImage,
    QKeySequence, QPainter, QPen, QPixmap, QShortcut,
    QStandardItem, QStandardItemModel, QTextCursor,
)

# ======================= Qt - Widgets =======================
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QDialogButtonBox,
    QDockWidget, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QInputDialog,
    QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QMenu, QMessageBox, QPlainTextEdit,
    QPushButton, QRubberBand, QSizePolicy, QSlider,
    QSpinBox, QSplitter, QStyle, QTabWidget, QTableView,
    QToolBar, QToolButton, QVBoxLayout, QWidget, QSpacerItem,
)

# ======================= VTK - Qt Bridge =======================
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

# ======================= VTK - Rendering =======================
from vtkmodules.vtkRenderingCore import (
    vtkActor, vtkImageSlice, vtkImageSliceMapper,
    vtkPolyDataMapper, vtkRenderer, vtkVolume,
    vtkVolumeProperty, vtkWindowToImageFilter,
)
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor, vtkScalarBarActor
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
import vtkmodules.vtkInteractionStyle          # noqa: F401
import vtkmodules.vtkRenderingOpenGL2          # noqa: F401

# ======================= VTK - IO =======================
from vtkmodules.vtkIOImage import vtkJPEGWriter, vtkNIFTIImageReader, vtkPNGWriter
from vtkmodules.vtkIOXML import vtkXMLImageDataReader, vtkXMLPolyDataReader
from vtkmodules.vtkIOLegacy import (
    vtkDataSetReader, vtkGenericDataObjectReader, vtkPolyDataReader,
)
from vtkmodules.vtkIOGeometry import vtkSTLReader

# ======================= VTK - Data Models =======================
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkImageData, vtkPolyData

# ======================= VTK - Filters & Core =======================
from vtkmodules.vtkFiltersCore import vtkAppendPolyData
try:
    from vtkmodules.vtkFiltersGeometry import vtkDataSetSurfaceFilter
except ImportError:
    from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter as vtkDataSetSurfaceFilter
from vtkmodules.vtkCommonCore import vtkFloatArray, vtkOutputWindow, vtkPoints

# ======================= Scientific / IO =======================
import numpy as np
import cv2

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import nibabel as nib
except ImportError:
    nib = None

import trimesh
import pyvista as pv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ======================= Helper(s) =======================
def qt_icon(style: QStyle, rel_path: str | None = None) -> QIcon:
    """
    If 'rel_path' (under ./icons/) exists, load it; else return a standard icon from style.
    Usage: qt_icon(self.style(), "ruler.png") or qt_icon(self.style(), QStyle.SP_DriveHDIcon)
    """
    if isinstance(rel_path, str):
        try:
            base = os.path.dirname(os.path.abspath(__file__))
            p = os.path.join(base, "icons", rel_path)
            if os.path.isfile(p):
                return QIcon(p)
        except (OSError, TypeError):
            pass
    sp = rel_path if isinstance(rel_path, QStyle.StandardPixmap) else QStyle.SP_FileIcon
    return style.standardIcon(sp)


# ======================= __all__ =======================
__all__ = [
    # Qt modules
    "QtCore", "QtGui", "QtWidgets",
    # Qt Core
    "QEventLoop", "QObject", "QPoint", "QRect", "QRectF",
    "QSettings", "QSize", "QTimer", "QUrl", "Qt", "Signal",
    # Qt GUI
    "QAction", "QColor", "QDesktopServices", "QFont", "QIcon", "QImage",
    "QKeySequence", "QPainter", "QPen", "QPixmap", "QShortcut",
    "QStandardItem", "QStandardItemModel", "QTextCursor",
    # Qt Widgets
    "QApplication", "QComboBox", "QDialog", "QDialogButtonBox",
    "QDockWidget", "QDoubleSpinBox", "QFileDialog", "QFormLayout",
    "QGroupBox", "QHBoxLayout", "QHeaderView", "QInputDialog",
    "QLabel", "QLineEdit", "QListWidget", "QListWidgetItem",
    "QMainWindow", "QMenu", "QMessageBox", "QPlainTextEdit",
    "QPushButton", "QRubberBand", "QSizePolicy", "QSlider",
    "QSpinBox", "QSplitter", "QStyle", "QTabWidget", "QTableView",
    "QSpacerItem", "QToolBar", "QToolButton", "QVBoxLayout", "QWidget",
    # VTK
    "QVTKRenderWindowInteractor",
    "vtkActor", "vtkAppendPolyData", "vtkAxesActor",
    "vtkCellArray", "vtkDataSetReader", "vtkDataSetSurfaceFilter",
    "vtkFloatArray", "vtkGenericDataObjectReader",
    "vtkImageData", "vtkImageSlice", "vtkImageSliceMapper",
    "vtkJPEGWriter", "vtkNIFTIImageReader", "vtkOrientationMarkerWidget",
    "vtkOutputWindow", "vtkPNGWriter", "vtkPoints",
    "vtkPolyData", "vtkPolyDataMapper", "vtkPolyDataReader",
    "vtkRenderer", "vtkSTLReader", "vtkScalarBarActor",
    "vtkSmartVolumeMapper", "vtkVolume", "vtkVolumeProperty",
    "vtkWindowToImageFilter",
    "vtkXMLImageDataReader", "vtkXMLPolyDataReader",
    # Scientific / IO
    "cv2", "matplotlib", "nib", "np", "pd", "plt", "pv", "trimesh",
    # Standard library
    "datetime", "logging", "math", "os", "pathlib", "re", "shutil", "sys",
    "tempfile", "time", "uuid",
    "Any", "List", "Literal", "Optional", "Path", "Sequence",
    "TYPE_CHECKING", "Tuple", "Union",
    # Helpers
    "qt_icon",
]
