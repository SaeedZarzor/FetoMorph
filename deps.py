# deps.py  — central imports for the app

# ---------------- Qt ----------------
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QSize, QRect, QObject, Signal, QPoint, QUrl, QTimer, QRectF
from PySide6.QtGui import QPixmap, QAction, QPainter, QTextCursor, QImage, QKeySequence, QIcon, QDesktopServices, QColor, QPen,QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QFileDialog,QRubberBand,
    QVBoxLayout, QHBoxLayout, QToolBar, QSlider, QComboBox,QDockWidget,
    QMessageBox, QSizePolicy, QGroupBox, QPlainTextEdit, QSplitter, QInputDialog, QDialog, QFormLayout, QSizePolicy,
    QDoubleSpinBox, QDialogButtonBox, QSpinBox, QStyle, QTabWidget, QGroupBox, QToolButton,
    QWidgetItem, QListWidget,QListWidgetItem, QLineEdit, QPushButton
)

# -------------- VTK + Qt bridge --------------
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

# -------------- VTK core rendering --------------
from vtkmodules.vtkRenderingCore import (
    vtkRenderer, vtkPolyDataMapper, vtkActor,
    vtkImageSliceMapper, vtkImageSlice, vtkVolume, vtkVolumeProperty,
    vtkWindowToImageFilter
)
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper

# -------------- VTK IO --------------
from vtkmodules.vtkIOImage import vtkNIFTIImageReader, vtkPNGWriter, vtkJPEGWriter
from vtkmodules.vtkIOXML import vtkXMLImageDataReader, vtkXMLPolyDataReader
from vtkmodules.vtkIOLegacy import vtkGenericDataObjectReader, vtkDataSetReader
from vtkmodules.vtkIOGeometry import vtkSTLReader

# -------------- VTK data models --------------
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPolyData

# -------------- VTK filters & logging --------------
try:
    from vtkmodules.vtkFiltersGeometry import vtkDataSetSurfaceFilter
except ImportError:
    from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter as vtkDataSetSurfaceFilter
from vtkmodules.vtkCommonCore import vtkOutputWindow

import vtkmodules.vtkInteractionStyle  # noqa: F401
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401

# ---------- Sci/IO ----------
import numpy as np
import cv2
try:
    import pandas as pd
except Exception:  # keep optional
    pd = None

try:
    import nibabel as nib
except Exception:
    nib = None
    
    
import os, sys, math, tempfile, shutil, pathlib, uuid, time, shutil, re
from datetime import datetime
from typing import Optional, Tuple
from pathlib import Path


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Small helper(s) you want globally ----------
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
        except Exception:
            pass
    # If rel_path is not a file, treat it as a StandardPixmap enum
    sp = rel_path if isinstance(rel_path, QStyle.StandardPixmap) else QStyle.SP_FileIcon
    return style.standardIcon(sp)

# Control what `from deps import *` exports (optional but nice)
__all__ = [
    # Qt modules & classes
    "QtCore","QtGui","QtWidgets","Qt","QSize","QRect","QPoint","QObject","Signal","QUrl","QRectF",
    "QAction","QKeySequence","QIcon","QPainter","QPixmap","QTextCursor","QImage","QDockWidget",
    "QApplication","QMainWindow","QWidget","QLabel","QFileDialog","QVBoxLayout","QHBoxLayout","QPushButton",
    "QToolBar","QSlider","QComboBox","QMessageBox","QSizePolicy","QGroupBox","QPlainTextEdit","QLineEdit",
    "QSplitter","QInputDialog","QDialog","QFormLayout","QDoubleSpinBox","QSpinBox", "QListWidget", "QListWidgetItem",
    "QDialogButtonBox","QStyle","QDesktopServices","QTabWidget","QToolButton","QRubberBand","QColor","QPen","QFont",
    # VTK
    "QVTKRenderWindowInteractor","vtkRenderer","vtkPolyDataMapper","vtkActor",
    "vtkImageSliceMapper","vtkImageSlice","vtkVolume","vtkVolumeProperty","vtkWindowToImageFilter",
    "vtkSmartVolumeMapper","vtkNIFTIImageReader","vtkPNGWriter","vtkJPEGWriter",
    "vtkXMLImageDataReader","vtkXMLPolyDataReader","vtkGenericDataObjectReader","vtkDataSetReader",
    "vtkSTLReader","vtkImageData","vtkPolyData","vtkDataSetSurfaceFilter","vtkOutputWindow",
    # Sci/IO
    "np","cv2","pd","os","sys","math","tempfile","shutil","pathlib","datetime","Optional","Tuple","Path","re",
    # helpers
    "qt_icon",
]

