"""Shared constants and configuration for the card-centering pipeline.

A standard Pokemon (and most modern TCG) card measures 63 mm x 88 mm.
All geometry in the pipeline is derived from this physical aspect ratio so that
the rectified image has no built-in horizontal/vertical distortion that could
bias the centering measurement.
"""
from __future__ import annotations

# Physical card size (mm). Used only as a ratio, absolute units cancel out.
CARD_WIDTH_MM = 63.0
CARD_HEIGHT_MM = 88.0
CARD_ASPECT = CARD_WIDTH_MM / CARD_HEIGHT_MM  # ~0.7159 (w/h, portrait)

# Target long-edge resolution (px) of the rectified card. Higher = more
# sub-pixel headroom for the border scan. Kept large because 1% of a margin
# can change the grade.
WARP_LONG_EDGE = 2000

# Rectified card size (portrait), derived from the aspect ratio.
WARP_H = WARP_LONG_EDGE
WARP_W = int(round(WARP_LONG_EDGE * CARD_ASPECT))

# Small background margin kept around the warped card so the full colored border
# is captured even when Stage-1 corners are slightly off (the card is never cut).
MARGIN_FRAC = 0.05
WARP_MARGIN = int(round(MARGIN_FRAC * WARP_W))

# Downscale long edge used for the coarse card-locating pass (speed only; the
# precise line fit always runs on full-resolution pixels).
DETECT_LONG_EDGE = 1200

# HSV saturation threshold (0-255) above which a pixel is considered part of the
# *printed* card body rather than the transparent sleeve / neutral background.
# The sleeve, wood, and skin are comparatively desaturated; the yellow front
# border and blue back border are strongly saturated.
SATURATION_MIN = 60

# Acceptable aspect-ratio tolerance when validating a candidate card contour.
ASPECT_TOL = 0.25

# Half-width (px, in warped space) of the band scanned inward from each edge
# when locating the inner border. 22% of the shorter side comfortably covers
# the colored border plus a margin of artwork.
INNER_SCAN_FRACTION = 0.22

# Sides, in a fixed order used throughout the code base.
SIDES = ("left", "right", "top", "bottom")
