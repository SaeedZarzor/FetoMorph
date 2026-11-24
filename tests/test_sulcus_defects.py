import cv2
import numpy as np

from helpers.helpers import defect_mm_per_px_and_fixed, iter_hull_edge_sulci


def _contour(points, size=(520, 520)):
    """Filled polygon -> its external OpenCV contour (N, 1, 2)."""
    mask = np.zeros(size, np.uint8)
    cv2.fillPoly(mask, [np.array(points, np.int32)], 255)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cnts, key=cv2.contourArea)


# One hull edge (0,0)->(400,0) spanning two sulci (60 and 70 px deep) separated
# by a low ridge that dips only 30 px below the chord.
TWO_SULCI = [(60, 60), (160, 120), (260, 90), (360, 130), (460, 60),
             (460, 460), (60, 460)]


def _sulci(cnt, prominence_px):
    return [s for _a, _b, sulci in iter_hull_edge_sulci(cnt, prominence_px) for s in sulci]


def _cv2_defect_depths(cnt):
    hull = cv2.convexHull(cnt, returnPoints=False, clockwise=True)
    d = cv2.convexityDefects(cnt, hull)
    return [] if d is None else [x[0][3] / 256.0 for x in d]


def test_two_sulci_on_one_hull_edge_are_both_found():
    cnt = _contour(TWO_SULCI)
    old = [d for d in _cv2_defect_depths(cnt) if d > 20]
    assert len(old) == 1                      # OpenCV reports one point for the edge
    found = _sulci(cnt, 20)
    assert len(found) == 2
    depths = sorted(d for _p, d in found)
    assert abs(depths[0] - 60) < 1.5 and abs(depths[1] - 70) < 1.5


def test_deepest_matches_opencv_depth():
    cnt = _contour(TWO_SULCI)
    deepest = max(d for _p, d in _sulci(cnt, 20))
    assert abs(deepest - max(_cv2_defect_depths(cnt))) < 0.05


def test_prominence_drops_shallow_neighbour_but_keeps_deepest():
    cnt = _contour(TWO_SULCI)
    # The 60-px sulcus rises only ~30 px above the ridge next to it.
    found = _sulci(cnt, 40)
    assert len(found) == 1 and abs(found[0][1] - 70) < 1.5


def test_prominence_larger_than_every_sulcus_gives_none():
    assert _sulci(_contour(TWO_SULCI), 200) == []


def test_independent_of_contour_start_including_wrapped_edge():
    cnt = _contour(TWO_SULCI)
    hull = cv2.convexHull(cnt, returnPoints=False, clockwise=True).ravel()
    counts = set()
    for start in sorted(hull):                # start the contour at each hull vertex
        rolled = np.roll(cnt, -int(start), axis=0)
        found = _sulci(rolled, 20)
        if found:                             # legacy guard may skip some starts
            counts.add(len(found))
    assert counts == {2}


def test_small_ripples_below_threshold_are_ignored():
    # Same two sulci, but the profile carries +-2 px ripples (pixel-level noise).
    xs = np.arange(60, 461, 4)
    profile = np.interp(xs, [60, 160, 260, 360, 460], [60, 120, 90, 130, 60])
    ys = profile + 2.0 * np.sin(xs / 3.0)
    ys[0], ys[-1] = 60, 60                      # keep the two hull corners exact
    top = [(int(x), int(round(y))) for x, y in zip(xs, ys)]
    cnt = _contour(top + [(460, 460), (60, 460)])
    found = _sulci(cnt, 20)                     # prominence 20 px >> ripple amplitude
    assert len(found) == 2
    # A threshold below the ripple amplitude lets noise through -- documents why
    # the Sulcus Depth Threshold is the noise control.
    assert len(_sulci(cnt, 1)) > 2


def test_degenerate_contours_yield_nothing():
    tri = np.array([[[0, 0]], [[10, 0]], [[0, 10]]], np.int32)
    assert list(iter_hull_edge_sulci(tri, 5)) == []


def test_anisotropic_spacing_matches_legacy_projection():
    """Deepest sulcus per edge equals the old d/256 * defect_mm_per_px_and_fixed value."""
    sx, sz = 0.5, 1.5
    cnt = _contour(TWO_SULCI)
    hull = cv2.convexHull(cnt, returnPoints=False, clockwise=True)
    legacy = []
    for s, e, f, d in cv2.convexityDefects(cnt, hull)[:, 0]:
        _mm_px, mm_fixed = defect_mm_per_px_and_fixed(
            tuple(cnt[s][0]), tuple(cnt[e][0]), tuple(cnt[f][0]), sx, sz)
        legacy.append(d * mm_fixed)
    mine = [d for _a, _b, sul in iter_hull_edge_sulci(cnt, 5.0, spacing=(sx, sz)) for _p, d in sul]
    assert abs(max(mine) - max(legacy)) < 0.05


def test_isotropic_spacing_scales_depth_and_prominence():
    cnt = _contour(TWO_SULCI)
    px = [d for _a, _b, sul in iter_hull_edge_sulci(cnt, 20) for _p, d in sul]
    mm = [d for _a, _b, sul in iter_hull_edge_sulci(cnt, 20 * 0.1, spacing=(0.1, 0.1)) for _p, d in sul]
    assert len(px) == len(mm) == 2
    assert np.allclose(sorted(mm), np.array(sorted(px)) * 0.1, atol=1e-6)
