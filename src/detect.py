"""Stage 1 - card localization and perspective rectification.

Design goals (from the plan):
  * The card is always inside a transparent sleeve / toploader whose edge sits
    slightly *outside* the card. We must lock onto the printed card's colored
    border edge (saturated yellow front / dark-blue back), never the sleeve.
  * Fingers occlude parts of the edges, so we fit a straight LINE to each side
    (robust to occlusion via RANSAC) and intersect adjacent lines to get precise
    corners, rather than trusting any single corner point.
  * The card is tilted, so we use a full perspective (homography) warp.

Public entry point: ``rectify(image_bgr, debug_dir=None, name=None)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

from . import config


@dataclass
class Rectification:
    """Result of Stage 1."""
    warped: np.ndarray                 # rectified BGR card, shape (WARP_H, WARP_W, 3)
    corners: np.ndarray                # 4x2 float32 precise corners in source px (TL,TR,BR,BL)
    homography: np.ndarray             # 3x3 source->warped
    coarse_corners: np.ndarray         # 4x2 coarse corners (minAreaRect) for debugging
    side_lines: dict = field(default_factory=dict)   # side -> (point, direction)


# --------------------------------------------------------------------------- #
# GrabCut card silhouette (robust to glossy reflections, warm wood, and
# fingers, which a pure saturation/edge threshold cannot separate reliably).
# --------------------------------------------------------------------------- #
def _grabcut_mask(bgr: np.ndarray, long_edge: int = 600) -> np.ndarray:
    """Foreground (card) silhouette via GrabCut with a central-rectangle prior.

    The card is the central subject; the outer ~10% ring (background wood,
    edge-touching fingers, sleeve corners) is treated as definite background,
    so GrabCut's colour model cleanly separates the card body.
    """
    cv2.setRNGSeed(0)   # deterministic GrabCut (its GMM init uses OpenCV's RNG)
    h, w = bgr.shape[:2]
    scale = min(long_edge / max(h, w), 1.0)
    small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    hh, ww = small.shape[:2]
    gc = np.zeros((hh, ww), np.uint8)
    m = 0.10
    rect = (int(ww * m), int(hh * m), int(ww * (1 - 2 * m)), int(hh * (1 - 2 * m)))
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(small, gc, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    mask = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    # store the scale so callers can map back to full resolution
    _grabcut_mask.last_scale = scale
    return mask


def _largest_central_component(mask: np.ndarray) -> np.ndarray:
    """Keep the connected component that best represents the card.

    Prefers large area and proximity to the image centre (the card is the
    subject of the photo), so a saturated patch of background is ignored.
    """
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    h, w = mask.shape
    cx, cy = w / 2.0, h / 2.0
    best_label, best_score = -1, -1.0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 0.02 * h * w:
            continue
        dist = np.hypot(centroids[i, 0] - cx, centroids[i, 1] - cy)
        # score rewards area, penalises distance from centre
        score = area / (1.0 + dist)
        if score > best_score:
            best_score, best_label = score, i
    if best_label < 0:
        return mask
    return (labels == best_label).astype(np.uint8) * 255


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL using the sum/diff trick."""
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _coarse_quad(bgr: np.ndarray) -> np.ndarray:
    """Coarse 4 corners of the card via GrabCut + minAreaRect.

    Returns corners in full-res pixels, ordered TL, TR, BR, BL.
    """
    mask = _grabcut_mask(bgr)
    mask = _largest_central_component(mask)
    scale = _grabcut_mask.last_scale

    ys, xs = np.nonzero(mask)
    if len(xs) < 100:
        raise RuntimeError("card mask empty - could not locate the card body")
    pts = np.column_stack([xs, ys]).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect) / scale  # back to full res
    return _order_corners(box)


# --------------------------------------------------------------------------- #
# RANSAC line fit + intersection helpers.
# --------------------------------------------------------------------------- #
def _fit_line_ransac(points: np.ndarray, thresh: float = 2.0, iters: int = 400):
    """RANSAC line fit. Returns (point_on_line, unit_direction)."""
    pts = points[~np.isnan(points).any(axis=1)]
    if len(pts) < 2:
        raise RuntimeError("not enough edge points to fit a side line")
    best_inliers = None
    best_count = -1
    rng = np.random.default_rng(0)
    n = len(pts)
    for _ in range(iters):
        i, j = rng.choice(n, size=2, replace=False)
        a, b = pts[i], pts[j]
        d = b - a
        norm = np.hypot(*d)
        if norm < 1e-6:
            continue
        d = d / norm
        normal = np.array([-d[1], d[0]])
        dist = np.abs((pts - a) @ normal)
        inliers = dist < thresh
        c = int(inliers.sum())
        if c > best_count:
            best_count, best_inliers = c, inliers
    inl = pts[best_inliers]
    # least-squares refine on inliers via total least squares (PCA)
    mean = inl.mean(axis=0)
    u, _, vt = np.linalg.svd(inl - mean)
    direction = vt[0] / np.linalg.norm(vt[0])
    return mean.astype(np.float32), direction.astype(np.float32)


def _line_intersection(l1, l2) -> np.ndarray:
    """Intersection of two lines each given as (point, direction)."""
    p1, d1 = l1
    p2, d2 = l2
    # solve p1 + t d1 = p2 + u d2
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]], dtype=np.float64)
    b = (p2 - p1).astype(np.float64)
    t, _ = np.linalg.solve(A, b)
    return (p1 + t * d1).astype(np.float32)


# --------------------------------------------------------------------------- #
# Colour (chroma) based rectification - the precise path for saturated borders
# (yellow / blue). The border is a high-chroma band; per side we take the rising
# chroma crossing nearest the coarse edge and RANSAC-fit a line. This is very
# accurate and holo/text robust, but only applies to coloured borders.
# --------------------------------------------------------------------------- #
def _chroma_edge_points(lab_img, p0, p1, inward, band,
                        n_samples=200, step_px=1.0) -> np.ndarray:
    ts = np.linspace(0.06, 0.94, n_samples)
    bases = p0[None, :] + ts[:, None] * (p1 - p0)[None, :]
    n_half = int(round(band / step_px))
    steps = np.arange(-n_half, n_half + 1) * step_px
    coords = bases[:, None, :] + steps[None, :, None] * inward[None, None, :]
    xs, ys = coords[..., 0].ravel(), coords[..., 1].ravel()
    a = map_coordinates(lab_img[:, :, 1], [ys, xs], order=1, mode="nearest")
    b = map_coordinates(lab_img[:, :, 2], [ys, xs], order=1, mode="nearest")
    chroma = np.sqrt((a - 128.0) ** 2 + (b - 128.0) ** 2).reshape(n_samples, len(steps))
    kern = np.ones(5) / 5.0
    confirm, zero_idx = 8, n_half
    out = np.full((n_samples, 2), np.nan, dtype=np.float32)
    for i in range(n_samples):
        c = np.convolve(chroma[i], kern, mode="same")
        lo, hi = np.percentile(c, 15), np.percentile(c, 85)
        if hi - lo < 8.0:
            continue
        level = lo + 0.5 * (hi - lo)
        crossings = [k for k in range(1, len(c) - confirm)
                     if c[k - 1] < level <= c[k] and c[k:k + confirm].mean() >= level]
        if not crossings:
            continue
        k = min(crossings, key=lambda kk: abs(kk - zero_idx))
        aa, bb = c[k - 1], c[k]
        frac = (level - aa) / (bb - aa) if bb != aa else 0.0
        out[i] = bases[i] + (steps[k - 1] + frac * step_px) * inward
    return out


def _refine_corners_chroma(bgr: np.ndarray, coarse: np.ndarray):
    """Chroma-based corner refinement. Returns (corners, lines) or None.

    Returns None when the border is not clearly coloured (too few sides yield a
    chroma edge, or the chroma edge sits well inside the coarse edge - meaning it
    locked onto an inner boundary like gold-vs-blue rather than the card edge),
    so the caller can fall back to the colour-agnostic iterative path.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    tl, tr, br, bl = coarse
    centroid = coarse.mean(axis=0)
    side_pts = {"top": (tl, tr), "right": (tr, br), "bottom": (br, bl), "left": (bl, tl)}
    short = min(np.hypot(*(tr - tl)), np.hypot(*(bl - tl)))
    band = max(20.0, short * 0.06)

    lines, colored = {}, 0
    for side, (a, b) in side_pts.items():
        a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
        edge = (b - a) / (np.hypot(*(b - a)) + 1e-9)
        normal = np.array([-edge[1], edge[0]], dtype=np.float32)
        mid = 0.5 * (a + b)
        if np.dot(centroid - mid, normal) < 0:
            normal = -normal
        pts = _chroma_edge_points(lab, a, b, normal, band)
        valid = pts[~np.isnan(pts).any(axis=1)]
        # accept only if plenty of points AND they sit near the coarse outer edge
        off = np.median((valid - mid) @ normal) if len(valid) else 1e9
        if len(valid) >= 40 and off < 0.30 * band:
            lines[side] = _fit_line_ransac(pts)
            colored += 1
        else:
            lines[side] = None
    if colored < 2:
        return None
    # fill any missing side with the coarse edge line so all four are defined
    for side, (a, b) in side_pts.items():
        if lines[side] is None:
            a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
            lines[side] = (a, (b - a) / (np.hypot(*(b - a)) + 1e-9))
    return _corners_from_lines(lines), lines


# --------------------------------------------------------------------------- #
# Iterative rectification via colour-agnostic straight-edge detection.
#
# The card's outer edge is a long straight line whatever the border colour
# (yellow / blue / grey / gold). Rather than key on colour, we warp the card
# roughly straight, detect the outer edge of each side as the first strong
# straight Lab-gradient line coming in from the background, intersect the four
# lines for refined corners, and re-warp. Two or three iterations converge on a
# precise rectification. This bootstraps the "detect the border lines" idea past
# the chicken-and-egg of needing a good warp to detect clean lines.
# --------------------------------------------------------------------------- #
def _warp(bgr: np.ndarray, corners: np.ndarray):
    """Warp so ``corners`` (TL,TR,BR,BL) map to the card rect inside a margin."""
    m = config.WARP_MARGIN
    cw, ch = config.WARP_W + 2 * m, config.WARP_H + 2 * m
    dst = np.array([
        [m, m], [m + config.WARP_W - 1, m],
        [m + config.WARP_W - 1, m + config.WARP_H - 1], [m, m + config.WARP_H - 1],
    ], dtype=np.float32)
    H = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    warped = cv2.warpPerspective(bgr, H, (cw, ch), flags=cv2.INTER_CUBIC)
    return H, warped


def _outer_edge_points(lab: np.ndarray, side: str, margin: int, short: int) -> np.ndarray:
    """Warp-space (x,y) points on the outer card edge of one side.

    The outer edge sits near offset ``margin`` from the boundary (the previous
    corners were mapped there). Per scan line we take the first strong Lab-
    gradient peak coming inward from the background within a window around
    ``margin`` - the card cut edge. Border/artwork edges are further in and the
    transparent sleeve gives only a weak gradient, so this locks onto the true
    outer edge for any border colour.
    """
    H, W = lab.shape[:2]
    scan = margin + int(0.12 * short)
    lo = max(2, margin - int(0.06 * short))
    hi = margin + int(0.06 * short)
    if side in ("left", "right"):
        along = np.arange(int(0.10 * H), int(0.90 * H))
        sub = lab[along][:, :scan, :] if side == "left" else lab[along][:, W - scan:, :][:, ::-1, :]
    else:
        along = np.arange(int(0.10 * W), int(0.90 * W))
        sub = lab[:scan, along, :].transpose(1, 0, 2) if side == "top" \
            else lab[H - scan:, along, :][::-1].transpose(1, 0, 2)

    grad = np.sqrt(sum(np.gradient(sub[:, :, ch], axis=1) ** 2 for ch in range(3)))
    kern = np.ones(5) / 5.0
    grad = np.apply_along_axis(lambda r: np.convolve(r, kern, mode="same"), 1, grad)

    # Projection consensus: average gradient across the whole side. A real card
    # edge is a straight line spanning the side, so it dominates the projection;
    # blurry background structure and glare average out. The outer edge is the
    # first prominent projection peak coming inward from the background.
    prof = grad.mean(axis=0)
    pmax = prof[lo:hi].max()
    o_hat = -1
    for j in range(lo + 1, hi - 1):
        if prof[j] >= prof[j - 1] and prof[j] > prof[j + 1] and prof[j] > 0.45 * pmax:
            o_hat = j
            break
    if o_hat < 0:
        return np.empty((0, 2), dtype=np.float32)

    # Refine per scan line: peak nearest the consensus offset (ties all lines to
    # the same straight edge, rejecting lines that found background gradients).
    win = max(6, int(0.015 * short))
    pts = []
    for idx in range(len(along)):
        g = grad[idx]
        a0, a1 = max(1, o_hat - win), min(len(g) - 1, o_hat + win)
        j = a0 + int(np.argmax(g[a0:a1]))
        if g[j] < max(5.0, 0.30 * g.max()):
            continue
        a, b, c = g[j - 1], g[j], g[j + 1]
        den = a - 2 * b + c
        off = j + (0.5 * (a - c) / den if abs(den) > 1e-6 else 0.0)
        a_pos = float(along[idx])
        if side == "left":
            pts.append((off, a_pos))
        elif side == "right":
            pts.append((W - 1 - off, a_pos))
        elif side == "top":
            pts.append((a_pos, off))
        else:
            pts.append((a_pos, H - 1 - off))
    return np.array(pts, dtype=np.float32)


def _corners_from_lines(lines: dict) -> np.ndarray:
    return np.array([
        _line_intersection(lines["left"], lines["top"]),     # TL
        _line_intersection(lines["top"], lines["right"]),    # TR
        _line_intersection(lines["right"], lines["bottom"]),  # BR
        _line_intersection(lines["bottom"], lines["left"]),  # BL
    ], dtype=np.float32)


def _refine_iterative(bgr: np.ndarray, coarse: np.ndarray, iters: int = 3):
    """Iterate warp -> detect outer edges -> re-corner until convergence."""
    corners = coarse.astype(np.float32)
    m = config.WARP_MARGIN
    short = min(config.WARP_W, config.WARP_H)
    cw, ch = config.WARP_W + 2 * m, config.WARP_H + 2 * m
    # fallback line for each side = the current margin edge (keeps warp stable
    # if a side's outer edge cannot be detected, e.g. a very low-contrast border)
    fallback = {
        "left": (np.array([m, 0.]), np.array([0., 1.])),
        "right": (np.array([cw - 1 - m, 0.]), np.array([0., 1.])),
        "top": (np.array([0., m]), np.array([1., 0.])),
        "bottom": (np.array([0., ch - 1 - m]), np.array([1., 0.])),
    }
    H, warp = _warp(bgr, corners)
    lines = {}
    for _ in range(iters):
        lab = cv2.cvtColor(warp, cv2.COLOR_BGR2LAB).astype(np.float32)
        lines = {}
        for side in config.SIDES:
            pts = _outer_edge_points(lab, side, m, short)
            try:
                lines[side] = _fit_line_ransac(pts)
            except RuntimeError:
                lines[side] = fallback[side]
        corners_warp = _corners_from_lines(lines)
        # map refined corners from warp space back to the source image
        Hinv = np.linalg.inv(H)
        corners = cv2.perspectiveTransform(corners_warp[None, :, :], Hinv)[0]
        H, warp = _warp(bgr, corners)
    return corners, H, warp, lines


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def rectify(bgr: np.ndarray, debug_dir: Optional[str] = None,
            name: Optional[str] = None) -> Rectification:
    coarse = _coarse_quad(bgr)
    chroma = _refine_corners_chroma(bgr, coarse)
    if chroma is not None:
        # coloured border (yellow / blue): precise chroma rectification
        corners, lines = chroma
        H, warped = _warp(bgr, corners)
    else:
        # neutral border (grey / gold): colour-agnostic iterative rectification
        corners, H, warped, lines = _refine_iterative(bgr, coarse)

    result = Rectification(
        warped=warped, corners=corners, homography=H,
        coarse_corners=coarse, side_lines=lines,
    )
    if debug_dir and name:
        _save_debug(bgr, result, debug_dir, name)
    return result


def _save_debug(bgr, result: Rectification, debug_dir: str, name: str):
    import os
    os.makedirs(debug_dir, exist_ok=True)
    vis = bgr.copy()
    cv2.polylines(vis, [result.coarse_corners.astype(np.int32)], True, (0, 0, 255), 3)
    cv2.polylines(vis, [result.corners.astype(np.int32)], True, (0, 255, 0), 3)
    for c in result.corners:
        cv2.circle(vis, tuple(c.astype(int)), 10, (255, 0, 0), -1)
    cv2.imwrite(os.path.join(debug_dir, f"{name}_detect.jpg"), vis)
    cv2.imwrite(os.path.join(debug_dir, f"{name}_warp.jpg"), result.warped)
