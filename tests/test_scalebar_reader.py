import cv2
import numpy as np
import pytest

from functions.nifti_to_image import draw_new_scale_bar
from helpers.scalebar_reader import read_scalebar


def _slice_with_bar(size=768, bar_px=180, text="20 mm"):
    """A white export with a coloured blob and FetoMorph's own scale bar."""
    img = np.full((size, size, 3), 255, np.uint8)
    cv2.circle(img, (size // 2, size // 2), size // 5, (255, 0, 255), -1)
    return draw_new_scale_bar(img, bar_px, text=text)


def _white_on_dark(size=900, bar_px=200, text="20 mm"):
    """The light-on-dark layout that ``helpers.helpers.add_scalebar`` paints."""
    img = np.full((size, size, 3), 25, np.uint8)
    cv2.circle(img, (size // 2, size // 2), size // 4, (120, 60, 160), -1)
    y = size - 90
    x = size - 70 - bar_px
    cv2.rectangle(img, (x, y), (x + bar_px, y + 9), (255, 255, 255), -1)
    cv2.putText(img, text, (x + bar_px // 6, y + 45), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, (255, 255, 255), 2, cv2.LINE_AA)
    return img


@pytest.mark.parametrize("bar_px,text,value,unit", [
    (180, "20 mm", 20.0, "mm"),
    (120, "5 mm", 5.0, "mm"),
    (200, "10 mm", 10.0, "mm"),
    (300, "100 mm", 100.0, "mm"),
    (250, "50 mm", 50.0, "mm"),
    (150, "2.5 mm", 2.5, "mm"),
    (90, "1 cm", 1.0, "cm"),
])
def test_reads_label_and_bar(bar_px, text, value, unit):
    reading = read_scalebar(_slice_with_bar(bar_px=bar_px, text=text))
    assert reading is not None, f"no scalebar found for {text!r}"
    assert reading.value == pytest.approx(value)
    assert reading.unit == unit
    # The drawn rectangle spans bar_px + 1 columns inclusive.
    assert reading.bar_px == pytest.approx(bar_px, abs=2)


def test_pixel_size_matches_the_label():
    reading = read_scalebar(_slice_with_bar(bar_px=200, text="10 mm"))
    assert reading is not None
    assert reading.mm_per_px == pytest.approx(10.0 / reading.bar_px)
    assert reading.unit_per_px == pytest.approx(reading.mm_per_px)


def test_reads_light_bar_on_dark_background():
    reading = read_scalebar(_white_on_dark())
    assert reading is not None
    assert reading.polarity == "light"
    assert reading.value == pytest.approx(20.0)
    assert reading.unit == "mm"


@pytest.mark.parametrize("size", [512, 768, 1024, 1280])
def test_scale_invariance(size):
    """The same physical bar must give the same answer at any export size."""
    reading = read_scalebar(_slice_with_bar(size=size, bar_px=size // 5,
                                            text="20 mm"))
    assert reading is not None
    assert reading.value == pytest.approx(20.0)
    assert reading.mm_per_px == pytest.approx(20.0 / (size // 5), rel=0.02)


def test_no_scalebar_returns_none():
    img = np.full((700, 700, 3), 255, np.uint8)
    cv2.circle(img, (350, 350), 180, (255, 0, 255), -1)
    cv2.ellipse(img, (350, 350), (190, 150), 0, 0, 360, (0, 165, 255), 6)
    assert read_scalebar(img) is None


def test_bar_without_a_label_returns_none():
    """A lone horizontal stroke is not a calibrated scale bar."""
    img = np.full((700, 700, 3), 255, np.uint8)
    cv2.line(img, (380, 600), (600, 600), (0, 0, 0), 4)
    assert read_scalebar(img) is None


def test_label_without_a_bar_returns_none():
    img = np.full((700, 700, 3), 255, np.uint8)
    cv2.putText(img, "20 mm", (400, 650), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (0, 0, 0), 2, cv2.LINE_AA)
    assert read_scalebar(img) is None


def test_implausible_reading_is_rejected():
    """A label implying a 20 m bar must be refused, not applied."""
    img = _slice_with_bar(bar_px=200, text="20 m")
    reading = read_scalebar(img)
    assert reading is None or reading.unit != "m"


def test_handles_grayscale_and_alpha_inputs():
    bgr = _slice_with_bar()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    for img in (gray, bgra):
        reading = read_scalebar(img)
        assert reading is not None
        assert reading.value == pytest.approx(20.0)


def test_empty_input_returns_none():
    assert read_scalebar(None) is None
    assert read_scalebar(np.zeros((0, 0, 3), np.uint8)) is None


# ---------------------------------------------------------------------------
# Excluding the drawn bar from measurements
# ---------------------------------------------------------------------------

def _measured(img, threshold, pixel_size):
    """Inner/internal contours the measurement pipeline keeps at *threshold*."""
    from functions.measurements_image import _binarise_brain, _split_contours_for_mode
    bw = _binarise_brain(img, cv2.COLOR_RGB2GRAY)
    return _split_contours_for_mode(bw, threshold, pixel_size)


def test_measures_the_contour_area_the_bar_adds():
    from helpers.scalebar_reader import measure_scalebar_contours

    img = _slice_with_bar()
    reading = read_scalebar(img)
    contours = measure_scalebar_contours(img, reading)
    assert contours is not None
    assert contours.count >= 1
    assert contours.scalebar_max_px2 > 0
    # The brain blob must dwarf the bar, leaving room to cut between them.
    assert contours.largest_other_px2 > contours.scalebar_max_px2
    assert contours.is_safe()
    assert contours.threshold_px2() > contours.scalebar_max_px2


def test_suggested_threshold_drops_the_bar_and_keeps_the_brain():
    from helpers.scalebar_reader import measure_scalebar_contours

    img = _slice_with_bar()
    reading = read_scalebar(img)
    contours = measure_scalebar_contours(img, reading)
    px = reading.mm_per_px
    threshold = contours.threshold_px2() * px * px

    before, _ = _measured(img, 1.0, px)
    after, _ = _measured(img, threshold, px)
    assert len(before) > len(after), "threshold did not filter anything out"
    assert len(after) == 1, "only the brain contour should survive"
    # The surviving contour is the brain, not a piece of the scale bar.
    assert cv2.contourArea(after[0]) == pytest.approx(contours.largest_other_px2,
                                                      rel=1e-6)


def test_excluding_the_bar_shrinks_the_measured_area():
    from helpers.scalebar_reader import measure_scalebar_contours

    img = _slice_with_bar()
    reading = read_scalebar(img)
    contours = measure_scalebar_contours(img, reading)
    px = reading.mm_per_px

    def total_area(threshold):
        inner, _ = _measured(img, threshold, px)
        return sum(cv2.contourArea(c) * px * px for c in inner)

    inflated = total_area(1.0)
    corrected = total_area(contours.threshold_px2() * px * px)
    assert corrected < inflated
    assert corrected == pytest.approx(contours.largest_other_px2 * px * px,
                                      rel=1e-6)


def test_unsafe_when_the_bar_rivals_the_anatomy():
    """A bar as big as the subject must not trigger a threshold change."""
    from helpers.scalebar_reader import ScalebarContours

    rivals = ScalebarContours(scalebar_max_px2=1000.0, scalebar_total_px2=1200.0,
                              largest_other_px2=1500.0, count=2)
    assert not rivals.is_safe()
    roomy = ScalebarContours(scalebar_max_px2=100.0, scalebar_total_px2=150.0,
                             largest_other_px2=10000.0, count=3)
    assert roomy.is_safe()
    assert not ScalebarContours(1.0, 1.0, 0.0, 1).is_safe()
