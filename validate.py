"""Validation harness.

Repeatability: all front photos are of one card front, all back photos of one
back, so within each group the measured ratios should agree. We report the group
mean / robust-median / std, using the robust median as the card's true centering
(so a single bad photo does not dominate).

Lighting robustness: each image is re-analysed under brightness and gamma
perturbations; the ratio should barely move.

    python validate.py
"""
from __future__ import annotations

import glob
import os

import cv2
import numpy as np

from src import pipeline, detect, borders, centering


def _group(paths):
    rows = []
    for p in sorted(paths):
        try:
            r = pipeline.analyze(p)
            c = r.centering
            worst_sp = max([v for v in c.spread.values() if v == v] or [float("nan")])
            rows.append((r.name, c.lr, c.tb, c.confidence, worst_sp))
        except Exception as e:  # noqa
            rows.append((os.path.basename(p), None, None, "FAIL", float("nan")))
    return rows


def _summary(name, rows):
    lrs = np.array([r[1] for r in rows if r[1] is not None])
    tbs = np.array([r[2] for r in rows if r[2] is not None])
    print(f"\n=== {name} (n={len(rows)}) ===")
    print(f"{'image':10s} {'L:R':>12s} {'T:B':>12s}  {'conf':>6s} {'spread':>6s}")
    for nm, lr, tb, conf, sp in rows:
        if lr is None:
            print(f"{nm:10s} {'FAIL':>12s}")
            continue
        print(f"{nm:10s} {lr:6.1f}/{100-lr:4.1f} {tb:6.1f}/{100-tb:4.1f}  {conf:>6s} {sp:6.1f}")
    if len(lrs):
        print(f"  L:R  median={np.median(lrs):.1f}  mean={lrs.mean():.1f}  std={lrs.std():.2f}")
        print(f"  T:B  median={np.median(tbs):.1f}  mean={tbs.mean():.1f}  std={tbs.std():.2f}")


def _lighting_robustness(path):
    """Re-run under brightness/gamma changes; report ratio drift."""
    bgr0 = cv2.imread(path)
    variants = {
        "orig": bgr0,
        "dark": np.clip(bgr0 * 0.7, 0, 255).astype(np.uint8),
        "bright": np.clip(bgr0 * 1.3, 0, 255).astype(np.uint8),
        "gamma": (((bgr0 / 255.0) ** 1.4) * 255).astype(np.uint8),
    }
    out = {}
    for k, img in variants.items():
        try:
            rect = detect.rectify(img)
            c = centering.compute(borders.measure(rect.warped))
            out[k] = (c.lr, c.tb)
        except Exception:
            out[k] = (None, None)
    return out


def main():
    fronts = glob.glob("data/front*.jpg")
    backs = glob.glob("data/back*.jpg")
    front_rows = _group(fronts)
    back_rows = _group(backs)
    _summary("FRONT", front_rows)
    _summary("BACK", back_rows)

    print("\n=== Lighting robustness (L:R, T:B drift) ===")
    for path in [sorted(fronts)[0], sorted(backs)[0]]:
        name = os.path.splitext(os.path.basename(path))[0]
        res = _lighting_robustness(path)
        base = res["orig"]
        print(f"\n{name}:")
        for k, (lr, tb) in res.items():
            if lr is None:
                print(f"  {k:7s} FAIL")
                continue
            dlr = lr - base[0] if base[0] is not None else 0
            dtb = tb - base[1] if base[1] is not None else 0
            print(f"  {k:7s} L:R={lr:5.1f}  T:B={tb:5.1f}   (dLR={dlr:+.2f}, dTB={dtb:+.2f})")


if __name__ == "__main__":
    main()
