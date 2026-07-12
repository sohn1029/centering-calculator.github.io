"""Pokemon card centering analysis package.

Pipeline stages:
  detect     - Stage 1: locate the card and rectify it to a straight rectangle.
  borders    - Stage 2: find the inner/outer colored-border edges (sub-pixel).
  centering  - Stage 3: left/right & top/bottom ratios, grade, confidence.
  visualize  - Stage 4: render the report figure.
  pipeline   - end-to-end orchestration (``analyze(path) -> Result``).
"""
