"""Shared constants for the FetoMorph application.

Centralises magic numbers and thresholds so that every measurement module
references the same values and they can be tuned in one place.
"""

# ---------------------------------------------------------------------------
# Image processing thresholds
# ---------------------------------------------------------------------------
BINARY_THRESHOLD_VTK = 150       # cv2.threshold value for VTK screenshots (lighter background)
BINARY_THRESHOLD_DEFAULT = 200   # cv2.threshold value for STL / image / NIfTI renders

# ---------------------------------------------------------------------------
# Red reference-cube colour detection (RGB space)
# ---------------------------------------------------------------------------
# A rendered red calibration cube is detected by requiring R above a minimum
# and G below a maximum; B is ignored.
RED_CHANNEL_MIN = 150   # R channel must exceed this to be considered "red"
GREEN_CHANNEL_MAX = 50  # G channel must stay below this

# ---------------------------------------------------------------------------
# Convexity-defect fixed-point divisor
# ---------------------------------------------------------------------------
# OpenCV stores convexity-defect depths in 8.8 fixed-point format:
#   depth_in_pixels = raw_d / 256
DEFECT_FIXED_POINT = 256

# ---------------------------------------------------------------------------
# Default NIfTI segmentation region labels
# ---------------------------------------------------------------------------
# FreeSurfer cortical / subcortical label IDs commonly used for fetal brain
# segmentation masks.  The set is passed to region-filtering routines so that
# only these labels contribute to area / volume / GI calculations.
DEFAULT_NIFTI_REGIONS = {2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 17}
