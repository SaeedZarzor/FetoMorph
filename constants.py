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

# Minimum pixel_size (mm/px) at which sulci-depth numeric labels are drawn on
# the annotated image. Below this, the image is so dense that text labels
# overlap and clutter the result, so the markers are still drawn but the
# numeric value text is suppressed.
MIN_PIXEL_SIZE_FOR_DEPTH_LABELS = 0.4

# ---------------------------------------------------------------------------
# Sulcus depth classification (full MRI slices only). Each defect that
# survives the SULCI_DEPTH_*_FRACTION filter is binned into one of these
# categories using its depth as a fraction of the brain's slice length.
# Defects that fall outside every range stay in an "unclassified" bucket
# but still appear in the depth list.
SULCUS_PRIMARY_MIN_FRACTION = 0.15    # 15% of slice length
SULCUS_PRIMARY_MAX_FRACTION = 0.50    # 50%
SULCUS_SECONDARY_MIN_FRACTION = 0.05  # 5%
SULCUS_SECONDARY_MAX_FRACTION = 0.15  # 15%
SULCUS_TERTIARY_MIN_FRACTION = 0.015  # 1.5%
SULCUS_TERTIARY_MAX_FRACTION = 0.05  # 5%
# ---------------------------------------------------------------------------
# Default NIfTI segmentation region labels
# ---------------------------------------------------------------------------
# FreeSurfer cortical / subcortical label IDs commonly used for fetal brain
# segmentation masks.  The set is passed to region-filtering routines so that
# only these labels contribute to area / volume / GI calculations.
DEFAULT_NIFTI_REGIONS = {2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 17}

# ---------------------------------------------------------------------------
# Application defaults
# ---------------------------------------------------------------------------
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 900
DEFAULT_PIXEL_SIZE = 0.01        # mm/px fallback when user has not calibrated
DEFAULT_CNT_THRESHOLD = 100      # contour detection threshold (0-255)
DEFAULT_KERNEL_SIZE = 5          # morphological kernel size in pixels
DEFAULT_SLICE_THICKNESS = 0.5    # mm between slices
DEFAULT_SCALEBAR_MM = 25         # default scale-bar length in mm
CONSOLE_MAX_BLOCKS = 10000       # max lines in the output console
