"""User-tunable visualization settings (text, colors, sizes, view toggles).

Holds the live state for every knob exposed by the Preferences dialog and
persists it across launches via :class:`QSettings`. Defaults match the values
that were previously hardcoded throughout the drawing code, so first launch
looks identical to before.

Non-Qt drawing helpers (in ``helpers/`` and ``functions/``) reach the active
instance via :func:`get_active`; the main window wires itself in by calling
:func:`set_active` after constructing the singleton.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from deps import QObject, QSettings, Signal


@dataclass(frozen=True)
class VizDefaults:
    """Frozen container of factory defaults."""
    # --- Text on images ---
    text_scale_multiplier: float = 1.0
    hallmark_text_color_bgr: tuple = (255, 255, 255)
    sulcus_label_scale_multiplier: float = 0.75
    scalebar_text_color_bgr: tuple = (0, 0, 0)

    # --- Drawn-element colors (BGR for cv2) ---
    contour_inner_color_bgr: tuple = (0, 0, 255)
    contour_outer_color_bgr: tuple = (0, 255, 0)
    measurement_line_color_bgr: tuple = (255, 0, 0)
    sulcus_primary_color_bgr: tuple = (255, 0, 0)
    sulcus_secondary_color_bgr: tuple = (0, 215, 255)
    sulcus_tertiary_color_bgr: tuple = (255, 255, 0)
    sulcus_unclassified_color_bgr: tuple = (200, 200, 200)

    # --- Size multipliers (1.0 keeps current auto-scaling) ---
    contour_thickness_multiplier: float = 1.0
    marker_radius_multiplier: float = 1.0
    scalebar_thickness_multiplier: float = 1.0

    # --- View toggles ---
    show_label_overlay: bool = True
    show_zoom_controls: bool = True

    # --- VTK 3-D viewer colors (RGB float 0..1) ---
    vtk_background_rgbf: tuple = (0.07, 0.07, 0.07)
    vtk_surface_rgbf: tuple = (0.69, 0.77, 0.87)


_DEFAULTS = VizDefaults()
_DEFAULTS_DICT = asdict(_DEFAULTS)


def _is_color_field(name: str) -> bool:
    return name.endswith("_bgr") or name.endswith("_rgbf")


def _tuple_to_str(t: tuple) -> str:
    return ",".join(str(x) for x in t)


def _str_to_tuple(s: Any, default: tuple) -> tuple:
    try:
        parts = [float(x) for x in str(s).split(",")]
        if all(isinstance(d, int) for d in default):
            parts = [int(round(x)) for x in parts]
        return tuple(parts)
    except (ValueError, TypeError):
        return default


class VisualizationSettings(QObject):
    """Live state holder for visualization preferences; persists to QSettings."""

    settingsChanged = Signal()

    QSETTINGS_ORG = "FetoMorph"
    QSETTINGS_APP = "FetoMorph"
    QSETTINGS_GROUP = "visualization"

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        for name, value in _DEFAULTS_DICT.items():
            setattr(self, name, value)

    # ----- public API -----

    def load(self) -> None:
        """Populate every field from QSettings; missing keys keep defaults."""
        s = QSettings(self.QSETTINGS_ORG, self.QSETTINGS_APP)
        s.beginGroup(self.QSETTINGS_GROUP)
        try:
            for name, default in _DEFAULTS_DICT.items():
                if not s.contains(name):
                    continue
                setattr(self, name, self._coerce(name, s.value(name), default))
        finally:
            s.endGroup()

    def save(self) -> None:
        """Write the current state to QSettings."""
        s = QSettings(self.QSETTINGS_ORG, self.QSETTINGS_APP)
        s.beginGroup(self.QSETTINGS_GROUP)
        try:
            for name in _DEFAULTS_DICT:
                value = getattr(self, name)
                if _is_color_field(name):
                    s.setValue(name, _tuple_to_str(value))
                else:
                    s.setValue(name, value)
        finally:
            s.endGroup()
        s.sync()

    def reset_to_defaults(self) -> None:
        for name, value in _DEFAULTS_DICT.items():
            setattr(self, name, value)
        self.settingsChanged.emit()

    def snapshot(self) -> dict:
        """Return a shallow copy of every field, for the dialog's Cancel-revert."""
        return {name: getattr(self, name) for name in _DEFAULTS_DICT}

    def restore(self, snap: dict) -> None:
        for name, value in snap.items():
            setattr(self, name, value)
        self.settingsChanged.emit()

    def apply(self, updates: dict) -> None:
        """Update fields from a dict (used by the dialog's Apply / OK)."""
        for name, value in updates.items():
            if name in _DEFAULTS_DICT:
                setattr(self, name, value)
        self.settingsChanged.emit()

    # ----- internals -----

    @staticmethod
    def _coerce(name: str, raw: Any, default: Any) -> Any:
        if _is_color_field(name):
            return _str_to_tuple(raw, default)
        if isinstance(default, bool):
            if isinstance(raw, bool):
                return raw
            return str(raw).lower() in ("true", "1", "yes")
        if isinstance(default, int) and not isinstance(default, bool):
            try:
                return int(raw)
            except (TypeError, ValueError):
                return default
        if isinstance(default, float):
            try:
                return float(raw)
            except (TypeError, ValueError):
                return default
        return raw


# Module-level accessor used by non-Qt drawing helpers.
_active: VisualizationSettings | None = None


def get_active() -> VisualizationSettings:
    """Return the active :class:`VisualizationSettings`; lazily create one."""
    global _active
    if _active is None:
        _active = VisualizationSettings()
        _active.load()
    return _active


def set_active(vs: VisualizationSettings) -> None:
    """Register *vs* as the singleton that drawing helpers should read."""
    global _active
    _active = vs


def defaults() -> VizDefaults:
    """Return the immutable :class:`VizDefaults` instance."""
    return _DEFAULTS
