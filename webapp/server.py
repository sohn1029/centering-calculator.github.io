"""FastAPI backend for the semi-automatic card-centering web app.

Thin layer over the existing CV pipeline (src/): it provides a *rough*
recommendation for the outer card corners and does the perspective warp, then
the human refines everything in the browser. Two endpoints:

  POST /api/detect  (multipart image)      -> { id, w, h, corners }
  POST /api/warp    (json {id, corners})   -> { image(dataURL), w, h, outer, inner }

Run:  conda run -n card uvicorn webapp.server:app --reload --port 8000
Then open http://localhost:8000/
"""
from __future__ import annotations

import base64
import io
import uuid
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import detect, borders  # noqa: E402

app = FastAPI(title="Card Centering")

# the single-page app lives at the repo root (so GitHub Pages can serve it too)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_INDEX = os.path.join(_ROOT, "index.html")
# in-memory image cache: id -> BGR ndarray (fine for local single-user use)
_IMAGES: dict[str, np.ndarray] = {}


def _read_bgr(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "could not decode image")
    return bgr


def _png_data_url(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", bgr)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


@app.post("/api/detect")
async def detect_corners(file: UploadFile = File(...)):
    bgr = _read_bgr(await file.read())
    h, w = bgr.shape[:2]
    img_id = uuid.uuid4().hex
    _IMAGES[img_id] = bgr

    # Rough recommendation: coarse GrabCut quad, refined by the colour path when
    # it applies. Any failure just falls back to an inset default box; the user
    # drags the corners regardless, so this only needs to be roughly right.
    try:
        coarse = detect._coarse_quad(bgr)
        chroma = detect._refine_corners_chroma(bgr, coarse)
        corners = chroma[0] if chroma is not None else coarse
    except Exception:
        mx, my = w * 0.12, h * 0.12
        corners = np.array([[mx, my], [w - mx, my], [w - mx, h - my], [mx, h - my]],
                           dtype=np.float32)
    return {"id": img_id, "w": w, "h": h,
            "corners": np.asarray(corners, float).round(1).tolist()}


class WarpReq(BaseModel):
    id: str
    corners: List[List[float]]   # 4 points TL, TR, BR, BL in original-image px


@app.post("/api/warp")
async def warp(req: WarpReq):
    bgr = _IMAGES.get(req.id)
    if bgr is None:
        raise HTTPException(404, "image expired; please re-upload")
    corners = np.asarray(req.corners, dtype=np.float32)
    if corners.shape != (4, 2):
        raise HTTPException(400, "corners must be 4 points")

    H, warped = detect._warp(bgr, corners)
    ch, cw = warped.shape[:2]

    # Suggested inner / outer rectangles for the user to refine.
    try:
        b = borders.measure(warped)
        outer = {"left": b.outer["left"], "top": b.outer["top"],
                 "right": b.outer["right"], "bottom": b.outer["bottom"]}
        inner = {"left": b.inner["left"], "top": b.inner["top"],
                 "right": b.inner["right"], "bottom": b.inner["bottom"]}
    except Exception:
        m = detect.config.WARP_MARGIN
        outer = {"left": m, "top": m, "right": cw - m, "bottom": ch - m}
        gx, gy = 0.07 * (cw - 2 * m), 0.07 * (ch - 2 * m)
        inner = {"left": m + gx, "top": m + gy, "right": cw - m - gx, "bottom": ch - m - gy}

    return {"image": _png_data_url(warped), "w": cw, "h": ch,
            "outer": outer, "inner": inner}


@app.get("/")
async def index():
    return FileResponse(_INDEX)
