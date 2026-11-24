"""classify_slice_kind() override modes used by run_master_measurement_reports --slice-kind."""
import glob

import cv2
import pytest

from helpers.slice_kind_classifier import classify_slice_kind
from managers.visualization_settings import get_active

FULL = sorted(glob.glob("Examples/full_slices/30/axial/*.png"))[:1]
CROPPED = sorted(glob.glob("Examples/cropped_slices/30/coronal/*.png"))[:1]


class _Override:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        self.old = get_active().slice_kind_override
        get_active().slice_kind_override = self.value

    def __exit__(self, *exc):
        get_active().slice_kind_override = self.old   # in-process only; never saved


@pytest.mark.skipif(not (FULL and CROPPED), reason="example images not available")
def test_modes_on_full_and_cropped_images():
    full, cropped = cv2.imread(FULL[0]), cv2.imread(CROPPED[0])
    with _Override("auto"):                                   # mix: classifier decides
        assert classify_slice_kind(full)[0] != "not_full_slice"
        assert classify_slice_kind(cropped)[0] == "not_full_slice"
    with _Override("not_full_slice"):                         # cropped: never a full slice
        assert classify_slice_kind(full) == ("not_full_slice", 1.0)
    with _Override("full"):                                   # full: "not a full slice" ignored
        for img in (full, cropped):
            kind, conf = classify_slice_kind(img)
            assert kind in ("sagittal", "coronal", "axial") and conf == 1.0
