#!/usr/bin/env python3
"""Build the image_race spec JSONs from a dgx_official.py result.

dgx_official.py writes per-image {ours_s, up_greedy_s, boxes (ours, deduped),
target_w/h}. This turns that into the spec_<img>[_f16].json files image_race.py
renders: the displayed boxes are ours (clean, early-stopped), the two engine
bars are the real measured greedy times (ours vs the official bf16 greedy).

  python3 build_specs.py /tmp/dgx_official_q8.json ""     "locate-anything.cpp (q8)"
  python3 build_specs.py /tmp/dgx_official_f16.json "_f16" "locate-anything.cpp (f16)"
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MEDIA = HERE.parent / "media"
QUERIES = {"coco_skater": "person · skateboard", "bus": "person · bus",
           "coco_kitchen": "bottle · bowl"}
NOTE = "same detections · GB10 GPU"
LINK = "github.com/mudler/locate-anything.cpp"


def main(jsonpath, suffix, ours_label):
    d = json.load(open(jsonpath))
    for img, m in d.items():
        spec = {
            "image": str(MEDIA / f"{img}_in.png"),
            "img_w": m["target_w"], "img_h": m["target_h"],
            "query": QUERIES[img], "note": NOTE, "link": LINK,
            "boxes": m["boxes"],
            "engines": [
                {"label": ours_label, "device": "GB10 GPU", "proc_s": m["ours_s"],
                 "rate": f"{m['ours_boxes']} boxes", "accent": "teal"},
                {"label": "PyTorch (official, bf16)", "device": "GB10 GPU",
                 "proc_s": m["up_greedy_s"], "rate": "greedy", "accent": "slate"},
            ],
        }
        out = HERE / f"spec_{img}{suffix}.json"
        out.write_text(json.dumps(spec, indent=2))
        print(f"wrote {out.name}: {m['ours_boxes']} boxes, "
              f"ours {m['ours_s']}s vs official {m['up_greedy_s']}s "
              f"({round(m['up_greedy_s']/m['ours_s'],2)}x)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
