"""Stage 2 - inner / outer border detection on the rectified card.

The card is warped into a canvas with a small background margin
(config.WARP_MARGIN) so the full colored border (yellow front / blue back) is
always present. Centering is the width of that border on each side, so for every
side we detect BOTH edges:

  * outer edge - the card edge, where the background/glare/sleeve gives way to the
    border colour. Detecting it (rather than trusting the canvas boundary) absorbs
    Stage-1 slack and skips whatever margin lies outside the card.
  * inner edge - where the border colour gives way to the inner artwork.

Per side we sample a local reference border colour, then along many parallel scan
lines find the outer entry and inner exit as sub-pixel colour-distance crossings,
and take the median across the central 70% of the side (rejecting rounded corners,
text and artwork touching the border). Because the border is the same colour on
all four sides, a side whose local reference was contaminated by background is
detected (its found colour disagrees with the cross-side consensus) and
re-measured against the consensus colour.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np

from . import config

THR = 18.0   # Lab colour distance separating border from non-border


@dataclass
class Borders:
    outer: Dict[str, float]      # side -> outer-edge position (px, canvas space)
    inner: Dict[str, float]      # side -> inner-edge position (px, canvas space)
    widths: Dict[str, float]     # side -> border width (px) = margin on that side
    ref_color: Dict[str, np.ndarray]   # side -> local Lab border colour
    spread: Dict[str, float]     # side -> robust std of border width (px); quality signal
    size: tuple                  # (W, H) of the warped canvas
    margin: int                  # background margin around the card (px)


def _oriented_strip(lab: np.ndarray, side: str, scan: int) -> np.ndarray:
    """Return a strip with axis0 = along-side, axis1 = inward offset (0 = boundary)."""
    H, W = lab.shape[:2]
    if side == "left":
        s = lab[int(0.15 * H):int(0.85 * H), :scan, :]
    elif side == "right":
        s = lab[int(0.15 * H):int(0.85 * H), W - scan:, :][:, ::-1, :]
    elif side == "top":
        s = lab[:scan, int(0.15 * W):int(0.85 * W), :]
        s = np.transpose(s, (1, 0, 2))
    else:  # bottom
        s = lab[H - scan:, int(0.15 * W):int(0.85 * W), :][::-1, :, :]
        s = np.transpose(s, (1, 0, 2))
    return s


def _first_sustained(mask: np.ndarray, start: int, confirm: int) -> int:
    n = len(mask)
    for i in range(start, n - confirm):
        if mask[i] and mask[i:i + confirm].mean() >= 0.8:
            return i
    return -1


def _subpix(dist: np.ndarray, idx: int, thr: float) -> float:
    a, b = dist[idx - 1], dist[idx]
    if b == a:
        return float(idx)
    return (idx - 1) + (thr - a) / (b - a)


def _edges_1d(dist: np.ndarray, confirm: int, outer_max: int):
    """(outer_off, inner_off) sub-pixel border edges of one colour-distance profile."""
    o = _first_sustained(dist < THR, 0, confirm)
    if o < 0 or o > outer_max:
        return np.nan, np.nan
    outer = _subpix(dist, o, THR) if o > 0 else 0.0
    inn = _first_sustained(dist >= THR, o + confirm, confirm)
    if inn < 0:
        return np.nan, np.nan
    return outer, _subpix(dist, inn, THR)


def _measure_side(strip: np.ndarray, ref: np.ndarray, short: int,
                  margin: int, confirm: int):
    """(outer_off, inner_off, spread, found_border_colour) for one side."""
    dist = np.sqrt(((strip - ref[None, None, :]) ** 2).sum(axis=2))
    outer_max = margin + int(0.10 * short)            # allow for Stage-1 edge slack
    med = np.median(strip, axis=0)

    outers, inners = [], []
    for k in range(dist.shape[0]):
        o, i = _edges_1d(dist[k], confirm, outer_max)
        if not np.isnan(o) and not np.isnan(i) and i > o:
            outers.append(o)
            inners.append(i)
    if len(inners) < 10:
        raise RuntimeError("border edges not found on a side")
    outers = np.array(outers)
    inners = np.array(inners)
    widths = inners - outers
    med_w = float(np.median(widths))
    spread = float(1.4826 * np.median(np.abs(widths - med_w)))
    inl = np.abs(widths - med_w) <= max(3 * spread, 2.0)
    outer_off = float(np.median(outers[inl]))
    inner_off = float(np.median(inners[inl]))
    m0, m1 = int(outer_off + 0.3 * med_w), int(outer_off + 0.7 * med_w)
    m1 = max(m1, m0 + 1)
    found = np.median(med[m0:m1], axis=0)
    return outer_off, inner_off, spread, found


def _side_ref(strip: np.ndarray, short: int, margin: int) -> np.ndarray:
    """Local border colour: median just inside the (expected) card edge."""
    lo = margin + int(0.02 * short)
    hi = margin + int(0.045 * short)
    return np.median(strip[:, lo:hi, :].reshape(-1, 3), axis=0)


def measure(warped: np.ndarray) -> Borders:
    H, W = warped.shape[:2]
    margin = config.WARP_MARGIN
    lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB).astype(np.float32)
    short = min(H, W) - 2 * margin
    scan = margin + int(config.INNER_SCAN_FRACTION * short)
    confirm = max(5, int(0.006 * short))
    strips = {s: _oriented_strip(lab, s, scan) for s in config.SIDES}

    # Pass 1: local per-side reference (handles lighting gradients across the card).
    cand = {s: _side_ref(strips[s], short, margin) for s in config.SIDES}
    outer_off, inner_off, spread, found = {}, {}, {}, {}
    for s in config.SIDES:
        o, i, sp, bc = _measure_side(strips[s], cand[s], short, margin, confirm)
        outer_off[s], inner_off[s], spread[s], found[s] = o, i, sp, bc

    # Pass 2: re-measure only sides whose found colour disagrees with the
    # cross-side consensus (their local reference was contaminated by background).
    consensus = np.median(np.stack(list(found.values())), axis=0)
    ref = dict(cand)
    for s in config.SIDES:
        if np.linalg.norm(found[s] - consensus) > 28.0:
            try:
                o, i, sp, _ = _measure_side(strips[s], consensus, short, margin, confirm)
                outer_off[s], inner_off[s], spread[s], ref[s] = o, i, sp, consensus
            except RuntimeError:
                spread[s] = float("nan")   # flag: contaminated and unrecoverable

    outer = {"left": outer_off["left"], "right": W - outer_off["right"],
             "top": outer_off["top"], "bottom": H - outer_off["bottom"]}
    inner = {"left": inner_off["left"], "right": W - inner_off["right"],
             "top": inner_off["top"], "bottom": H - inner_off["bottom"]}
    widths = {s: inner_off[s] - outer_off[s] for s in config.SIDES}
    return Borders(outer=outer, inner=inner, widths=widths, ref_color=ref,
                   spread=spread, size=(W, H), margin=margin)
