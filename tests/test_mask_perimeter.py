import math

import cv2
import numpy as np
import pytest

from helpers.helpers import mask_perimeter_mm


def _circle_mask(radius=100, pad=12):
    size = 2 * (radius + pad) + 1
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (size // 2, size // 2), radius, 1, thickness=cv2.FILLED)
    return mask


def _rect_mask(width, height, pad=20):
    mask = np.zeros((height + 2 * pad, width + 2 * pad), dtype=np.uint8)
    cv2.rectangle(mask, (pad, pad), (pad + width, pad + height), 1, thickness=cv2.FILLED)
    return mask


def _rotated_square_mask(side=160, angle=0, pad=80):
    size = side + 2 * pad
    mask = np.zeros((size, size), dtype=np.uint8)
    rect = ((size / 2, size / 2), (side, side), angle)
    pts = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def test_circle_crofton_is_closer_than_arc_length():
    mask = _circle_mask(radius=100)
    analytic = 2 * math.pi * 100
    arc = mask_perimeter_mm(mask, 1.0, 1.0, method="arc_length")
    crofton = mask_perimeter_mm(mask, 1.0, 1.0, method="crofton")

    assert abs(crofton - analytic) < abs(arc - analytic)


def test_axis_aligned_square_documents_crofton_straight_edge_bias():
    mask = _rect_mask(200, 200)
    analytic = 800.0
    crofton = mask_perimeter_mm(mask, 1.0, 1.0, method="crofton")

    assert crofton < analytic
    assert abs(crofton - analytic) / analytic < 0.08


def test_rotated_square_spread_is_bounded_for_both_methods():
    arc_values = []
    crofton_values = []
    for angle in (0, 15, 30, 45):
        mask = _rotated_square_mask(angle=angle)
        arc_values.append(mask_perimeter_mm(mask, 1.0, 1.0, method="arc_length"))
        crofton_values.append(mask_perimeter_mm(mask, 1.0, 1.0, method="crofton"))

    assert max(arc_values) - min(arc_values) < 0.20 * np.mean(arc_values)
    assert max(crofton_values) - min(crofton_values) < 0.20 * np.mean(crofton_values)


def test_anisotropic_spacing_is_handled_by_both_methods():
    mask = _rect_mask(100, 80)
    analytic = 2 * (100 * 0.5 + 80 * 2.0)

    arc = mask_perimeter_mm(mask, 0.5, 2.0, method="arc_length")
    crofton = mask_perimeter_mm(mask, 0.5, 2.0, method="crofton")

    assert abs(arc - analytic) / analytic < 0.03
    assert abs(crofton - analytic) / analytic < 0.10


def test_crofton_fills_holes_before_perimeter():
    outer = _rect_mask(200, 200)
    ring = outer.copy()
    cv2.rectangle(ring, (80, 80), (160, 160), 0, thickness=cv2.FILLED)

    outer_only = mask_perimeter_mm(outer, 1.0, 1.0, method="arc_length")
    crofton_ring = mask_perimeter_mm(ring, 1.0, 1.0, method="crofton")

    assert abs(crofton_ring - outer_only) / outer_only < 0.08


def test_arc_length_is_deterministic_and_simplification_is_opt_in():
    mask = _circle_mask(radius=40)

    a = mask_perimeter_mm(mask, 1.0, 1.0, method="arc_length")
    b = mask_perimeter_mm(mask, 1.0, 1.0, method="arc_length")
    simplified = mask_perimeter_mm(
        mask, 1.0, 1.0, method="arc_length", simplify=True, epsilon=2.0)

    assert a == pytest.approx(b)
    assert simplified != pytest.approx(a)
