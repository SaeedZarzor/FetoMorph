"""Dataclass and factory for slice geometry state along an arbitrary axis.

Used by the STL/VTK measurement pipelines to precompute the normal, camera
position, origin, and translation for each coronal/sagittal/axial slice.
"""

from dataclasses import dataclass


@dataclass
class SliceState:
    """Immutable snapshot of a single slicing plane's geometry."""
    axis: str
    normal: tuple[int, int, int]
    cpos: str
    increment: float
    total_thickness: float
    origin: tuple[float, float, float]
    trns: tuple[float, float, float]
    length: float  # depth along the slicing axis

def make_state(snorml: str, k: int, s_n: int,
               Xstart: float, Xend: float,
               Ystart: float, Yend: float,
               Zstart: float, Zend: float,
               depth: float) -> SliceState:
    """Create a ``SliceState`` for the *k*-th of *s_n* evenly spaced slices.

    Args:
        snorml: Slice axis — ``"X"``, ``"Y"``, or ``"Z"``.
        k: Zero-based slice index.
        s_n: Total number of slices.
        Xstart, Xend: Mesh bounding-box range along X.
        Ystart, Yend: Mesh bounding-box range along Y.
        Zstart, Zend: Mesh bounding-box range along Z.
        depth: Depth along the slicing axis (used for camera translation).

    Returns:
        A populated ``SliceState``.
    """
    a = snorml.upper()
    if a == 'X':
        inc   = round(Xstart + k / s_n * (Xend - Xstart), 3)
        thick = (Xend - Xstart)
        return SliceState(
            axis='X',
            normal=(1, 0, 0),
            cpos="yz",
            increment=inc,
            total_thickness=thick,
            origin=(inc, 0.0, 0.0),
            trns=(-0.5 * depth, 2.0, 2.0),
            length=depth,
        )
    if a == 'Y':
        inc   = round(Ystart + k / s_n * (Yend - Ystart), 3)
        thick = (Yend - Ystart)
        return SliceState(
            axis='Y',
            normal=(0, 1, 0),
            cpos="xz",
            increment=inc,
            total_thickness=thick,
            origin=(0.0, inc, 0.0),
            trns=(2.0, -0.5 * depth, 2.0),
            length=depth,
        )
    if a == 'Z':
        inc   = round(Zstart + k / s_n * (Zend - Zstart), 3)
        thick = (Zend - Zstart)
        return SliceState(
            axis='Z',
            normal=(0, 0, 1),
            cpos="xy",
            increment=inc,
            total_thickness=thick,
            origin=(0.0, 0.0, inc),
            trns=(2.0, 2.0, -0.5 * depth),
            length=depth,
        )
    raise ValueError("snorml must be 'X', 'Y', or 'Z'.")
