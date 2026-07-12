"""CLI: analyse a single card image and write a visual report + JSON.

    python run.py data/front001.jpg
    python run.py data/front001.jpg --out output
"""
from __future__ import annotations

import argparse
import json
import os

from src import pipeline, visualize


def main():
    ap = argparse.ArgumentParser(description="Pokemon card centering analyzer")
    ap.add_argument("image", help="path to a card photo")
    ap.add_argument("--out", default="output", help="output directory")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    result = pipeline.analyze(args.image)
    c = result.centering

    report = os.path.join(args.out, f"{result.name}_report.png")
    visualize.render(result, report)
    with open(os.path.join(args.out, f"{result.name}.json"), "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"{result.name}")
    print(f"  Left/Right : {c.lr_text}")
    print(f"  Top/Bottom : {c.tb_text}")
    print(f"  Grade      : {c.grade:.0f} ({c.grade_label})   confidence={c.confidence}")
    print(f"  Report     : {report}")


if __name__ == "__main__":
    main()
