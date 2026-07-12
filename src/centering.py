"""Stage 3 - centering ratios, grade estimate and a confidence signal.

Centering compares the border width on opposing sides:
  * horizontal: left vs right   -> ideally 50 / 50
  * vertical:   top  vs bottom  -> ideally 50 / 50

The reported ratio for an axis is ``a / (a + b)``. The "off-centering" of a card
is usually summarised by its worst (most lopsided) axis, e.g. a 60/40 card. A
rough PSA-style grade is derived from that worst ratio (front tolerances; backs
are graded more leniently by PSA but we keep one scale for simplicity and expose
the raw ratios, which are what actually matter).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .borders import Borders

# worst allowed larger-side % for each PSA-style grade (front centering).
# e.g. Gem-Mint 10 tolerates up to ~55/45.
_GRADE_TABLE = [
    (55.0, 10.0, "Gem Mint"),
    (60.0, 9.0, "Mint"),
    (65.0, 8.0, "NM-Mint"),
    (70.0, 7.0, "Near Mint"),
    (75.0, 6.0, "Excellent"),
    (80.0, 5.0, "VG-EX"),
    (90.0, 4.0, "Good"),
    (100.0, 3.0, "Poor"),
]


@dataclass
class Centering:
    lr: float            # left percentage of the horizontal border (left / (left+right))
    tb: float            # top percentage of the vertical border (top / (top+bottom))
    widths: Dict[str, float]
    worst_pct: float     # larger side % of the more off-centre axis (>= 50)
    grade: float         # estimated PSA-style centering grade
    grade_label: str
    confidence: str      # "high" | "medium" | "low" from border-detection spread
    spread: Dict[str, float]

    @property
    def lr_text(self) -> str:
        return f"{self.lr:.1f} / {100 - self.lr:.1f}"

    @property
    def tb_text(self) -> str:
        return f"{self.tb:.1f} / {100 - self.tb:.1f}"


def _grade_for(worst_pct: float):
    for thresh, grade, label in _GRADE_TABLE:
        if worst_pct <= thresh:
            return grade, label
    return 1.0, "Poor"


def _confidence(spread: Dict[str, float]) -> str:
    vals = [v for v in spread.values()]
    if any(v != v for v in vals):        # NaN -> a side was unrecoverable
        return "low"
    worst = max(vals)
    if worst <= 6:
        return "high"
    if worst <= 15:
        return "medium"
    return "low"


def compute(borders: Borders) -> Centering:
    w = borders.widths
    lr = 100.0 * w["left"] / (w["left"] + w["right"])
    tb = 100.0 * w["top"] / (w["top"] + w["bottom"])
    worst = max(lr, 100 - lr, tb, 100 - tb)
    grade, label = _grade_for(worst)
    return Centering(
        lr=lr, tb=tb, widths=dict(w), worst_pct=worst,
        grade=grade, grade_label=label,
        confidence=_confidence(borders.spread), spread=dict(borders.spread),
    )
