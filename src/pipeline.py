"""End-to-end orchestration: image path -> centering result.

Kept deliberately thin and returning a structured result so a future app / web
backend can call ``analyze()`` and consume the dict without touching the stages.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from . import detect, borders, centering
from .centering import Centering
from .detect import Rectification
from .borders import Borders


@dataclass
class Result:
    name: str
    image_path: str
    rectification: Rectification
    borders: Borders
    centering: Centering

    def to_dict(self) -> dict:
        c = self.centering
        return {
            "name": self.name,
            "image_path": self.image_path,
            "left_right": {"left_pct": round(c.lr, 2), "right_pct": round(100 - c.lr, 2)},
            "top_bottom": {"top_pct": round(c.tb, 2), "bottom_pct": round(100 - c.tb, 2)},
            "border_widths_px": {k: round(v, 2) for k, v in c.widths.items()},
            "worst_pct": round(c.worst_pct, 2),
            "grade": c.grade,
            "grade_label": c.grade_label,
            "confidence": c.confidence,
            "spread_px": {k: (round(v, 2) if v == v else None) for k, v in c.spread.items()},
            "corners": self.rectification.corners.round(1).tolist(),
        }


def analyze(image_path: str) -> Result:
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"could not read image: {image_path}")
    name = os.path.splitext(os.path.basename(image_path))[0]
    rect = detect.rectify(bgr)
    bd = borders.measure(rect.warped)
    ce = centering.compute(bd)
    return Result(name=name, image_path=image_path,
                  rectification=rect, borders=bd, centering=ce)
