"""Full-dataset statistics for STSD v1.1 (LAS point clouds + projected PNG channels).

Usage:
    python -m stsd.tools.stats --root data/STSD_v1.1 --out reports/stsd_analysis --workers 12

Writes las_stats.csv, img_stats.csv and summary.json to --out.
"""
import argparse
import csv
import json
import os
import re
import struct
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

from stsd.utils.las import read_las

Image.MAX_IMAGE_PIXELS = None

CLASSES = ["tunnel_wall", "rail_track", "power_track", "walkway", "pipe", "cable", "attachment",
           "person", "big_bolt_hole", "small_bolt_hole", "signal_line", "joint"]
TYPES = {"horse": "horseshoe", "rec": "quasi_rectangular", "round": "circular"}
NAME_RE = re.compile(r"^(horse|rec|round)-(\d+)-(\d+)_(\d+)$")

IMG_DIRS = {"label": "label", "label show": "label_show", "intensity": "intensity", "H": "H", "z": "Z"}


def parse_name(stem):
    m = NAME_RE.match(stem)
    return TYPES[m.group(1)], int(m.group(3)), int(m.group(4))


def las_stats(path):
    stem = os.path.basename(path)[:-4]
    ttype, seq, part = parse_name(stem)
    a, scale, offset = read_las(path)
    xyz = [np.asarray(a[k]) * scale[i] + offset[i] for i, k in enumerate("XYZ")]
    uvw = [np.asarray(a[k]) * 1e-4 for k in ("inx", "iny", "inz")]
    inten = np.asarray(a["intensity"])
    cls = np.bincount(np.asarray(a["cls"]), minlength=12)
    xt, yt, zt = uvw
    length = float(xt.max() - xt.min())
    perim = float(yt.max() - yt.min())
    row = dict(name=stem, type=ttype, seq=seq, part=part, n_points=len(a), file_bytes=os.path.getsize(path),
               length_m=length, perimeter_m=perim, density_pts_per_m2=len(a) / max(length * perim, 1e-9),
               xt_min=float(xt.min()), xt_max=float(xt.max()),
               zt_min=float(zt.min()), zt_max=float(zt.max()),
               zt_p01=float(np.percentile(zt, 1)), zt_p99=float(np.percentile(zt, 99)),
               x_extent=float(np.ptp(xyz[0])), y_extent=float(np.ptp(xyz[1])), z_extent=float(np.ptp(xyz[2])),
               x_center=float(xyz[0].mean()), y_center=float(xyz[1].mean()), z_center=float(xyz[2].mean()),
               int_min=int(inten.min()), int_max=int(inten.max()), int_mean=float(inten.mean()),
               int_p99=float(np.percentile(inten, 99)),
               index2d_nonzero=int(np.count_nonzero(np.asarray(a["index2d"]))),
               gps_nonzero=int(np.count_nonzero(np.asarray(a["gps"]))),
               rgb_nonzero=int(np.count_nonzero(np.asarray(a["r"]))),
               n_cls_gt11=int(cls[12:].sum()))
    for i, c in enumerate(CLASSES):
        row[f"cls_{c}"] = int(cls[i])
    return row


def img_stats(args):
    root, stem = args
    ttype, seq, part = parse_name(stem)
    row = dict(name=stem, type=ttype)
    for d, key in IMG_DIRS.items():
        fn = os.path.join(root, "img", d, stem + ("_label_show" if d == "label show" else "") + ".png")
        im = Image.open(fn)
        arr = np.asarray(im)
        row[f"{key}_mode"] = im.mode
        row[f"{key}_shape"] = "x".join(map(str, arr.shape))
        row[f"{key}_dtype"] = str(arr.dtype)
        if key == "label":
            bc = np.bincount(arr.ravel(), minlength=12)
            row["label_max"] = int(arr.max())
            for i, c in enumerate(CLASSES):
                row[f"px_{c}"] = int(bc[i])
            row["px_gt11"] = int(bc[12:].sum())
            label = arr
        elif key == "label_show":
            row["label_show_n_colors"] = len(np.unique(arr.reshape(-1, arr.shape[-1]), axis=0))
        else:
            row[f"{key}_min"] = int(arr.min())
            row[f"{key}_max"] = int(arr.max())
            row[f"{key}_mean"] = float(arr.mean())
            row[f"{key}_zero_frac"] = float((arr == 0).mean())
    return row


def label_colors(root, stems):
    """Map label id -> RGB colour(s) used in the 'label show' images."""
    cmap = defaultdict(set)
    for stem in stems:
        lab = np.asarray(Image.open(os.path.join(root, "img", "label", stem + ".png")))
        rgb = np.asarray(Image.open(os.path.join(root, "img", "label show", stem + "_label_show.png")).convert("RGB"))
        for k in np.unique(lab):
            cols, cnt = np.unique(rgb[lab == k], axis=0, return_counts=True)
            cmap[int(k)].add(tuple(int(v) for v in cols[cnt.argmax()]))
    return {CLASSES[k] if k < 12 else str(k): sorted(v) for k, v in sorted(cmap.items())}


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def summarize(las_rows, img_rows):
    out = {}
    for group in ["all"] + list(TYPES.values()):
        L = [r for r in las_rows if group == "all" or r["type"] == group]
        I = [r for r in img_rows if group == "all" or r["type"] == group]
        cls_pts = np.array([[r[f"cls_{c}"] for c in CLASSES] for r in L]).sum(0)
        cls_px = np.array([[r[f"px_{c}"] for c in CLASSES] for r in I]).sum(0)
        pts = np.array([r["n_points"] for r in L])
        out[group] = dict(
            n_segments=len(L),
            total_points=int(pts.sum()),
            points_per_segment=dict(min=int(pts.min()), mean=float(pts.mean()), max=int(pts.max())),
            total_length_m=float(sum(r["length_m"] for r in L)),
            length_m=dict(min=min(r["length_m"] for r in L), max=max(r["length_m"] for r in L)),
            perimeter_m=dict(min=min(r["perimeter_m"] for r in L), max=max(r["perimeter_m"] for r in L)),
            density_pts_per_m2_mean=float(np.mean([r["density_pts_per_m2"] for r in L])),
            intensity_range=[min(r["int_min"] for r in L), max(r["int_max"] for r in L)],
            zt_range=[min(r["zt_min"] for r in L), max(r["zt_max"] for r in L)],
            las_bytes=int(sum(r["file_bytes"] for r in L)),
            class_points={c: int(v) for c, v in zip(CLASSES, cls_pts)},
            class_points_pct={c: round(float(v) / cls_pts.sum() * 100, 4) for c, v in zip(CLASSES, cls_pts)},
            class_pixels_pct={c: round(float(v) / cls_px.sum() * 100, 4) for c, v in zip(CLASSES, cls_px)},
            segments_with_class={c: int(sum(r[f"cls_{c}"] > 0 for r in L)) for c in CLASSES},
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/STSD_v1.1")
    ap.add_argument("--out", default="reports/stsd_analysis")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    las_dir = os.path.join(args.root, "las")
    stems = sorted(f[:-4] for f in os.listdir(las_dir) if f.endswith(".las"))
    with ProcessPoolExecutor(args.workers) as ex:
        las_rows = list(ex.map(las_stats, [os.path.join(las_dir, s + ".las") for s in stems]))
        img_rows = list(ex.map(img_stats, [(args.root, s) for s in stems]))
    write_csv(las_rows, os.path.join(args.out, "las_stats.csv"))
    write_csv(img_rows, os.path.join(args.out, "img_stats.csv"))
    summary = summarize(las_rows, img_rows)
    summary["label_show_colors"] = label_colors(args.root, [s for s in stems if s.endswith("_3")][::8])
    summary["image_formats"] = {k: sorted({r[f"{k}_mode"] + " " + r[f"{k}_shape"] + " " + r[f"{k}_dtype"]
                                           for r in img_rows}) for k in IMG_DIRS.values()}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if not kk.startswith("class_")}
                      for k, v in summary.items() if k in ("all",) + tuple(TYPES.values())}, indent=1))


if __name__ == "__main__":
    main()
