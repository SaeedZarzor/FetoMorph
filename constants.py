# constants.py — shared constants for FetoMorph

# Image processing thresholds
BINARY_THRESHOLD_VTK = 150       # cv2.threshold value for VTK screenshots
BINARY_THRESHOLD_DEFAULT = 200   # cv2.threshold value for STL/image/NIfTI

# Red reference-cube color detection (RGB space)
RED_CHANNEL_MIN = 150   # R channel must be above this
GREEN_CHANNEL_MAX = 50  # G channel must be below this

# Convexity defect minimum (OpenCV fixed-point: 256 = 1 pixel)
DEFECT_FIXED_POINT = 256

# Default NIfTI segmentation region labels
# These correspond to FreeSurfer cortical/subcortical labels commonly used
# for fetal brain analysis
DEFAULT_NIFTI_REGIONS = {2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 17}
