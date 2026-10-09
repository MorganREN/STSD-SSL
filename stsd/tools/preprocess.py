"""
Preprocess STSD v1.1 LAS files into the Pointcept scene-folder layout.

    <output_root>/
    ├── meta.json                   # per-scene metadata + per-type intensity scale
    └── <scene>/
        ├── coord.npy    float32 (n, 3)  original 3D coordinates minus scene centroid
        ├── unroll.npy   float32 (n, 3)  unrolled (s, c, r) in metres
        ├── strength.npy float32 (n, 1)  intensity / per-type p99.5, clipped to [0, 1.5]
        ├── normal.npy   float32 (n, 3)  PCA normals oriented towards the tunnel axis
        ├── segment.npy  int16   (n,)
        └── origin/
            ├── segment.npy  uint8 (N,)   labels of all original points
            └── inverse.npy  int32 (N,)   original point -> voxel index

Points are voxelised at `--grid_size` (default 1 cm), keeping the original point closest
to each voxel centre, so every stored point is a real measured point.

Usage:
    python -m stsd.tools.preprocess --las_root data/STSD_v1.1/las \
        --output_root data/processed/stsd_1cm --num_workers 8
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from stsd.utils import load_points, parse_scene_name, read_las


def voxelize(coord, grid_size):
    """Keep the point nearest to each voxel centre. Returns (kept index, inverse)."""
    origin = coord.min(0)
    grid = np.floor((coord - origin) / grid_size).astype(np.int64)
    dims = grid.max(0) + 1
    key = (grid[:, 0] * dims[1] + grid[:, 1]) * dims[2] + grid[:, 2]
    dist = np.square(coord - origin - (grid + 0.5) * grid_size).sum(1)
    order = np.lexsort((dist, key))
    uniq_key, first = np.unique(key[order], return_index=True)
    keep = order[first]
    inverse = np.searchsorted(uniq_key, key).astype(np.int32)
    return keep, inverse


def estimate_normals(coord, unroll, k=30, axis_bin=0.5):
    """PCA normals, flipped to point towards the local tunnel axis (centroid of an s-slice)."""
    _, nn = cKDTree(coord).query(coord, k=k, workers=1)
    nbr = coord[nn] - coord[nn].mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nbr, nbr)
    normal = np.linalg.eigh(cov)[1][:, :, 0]
    s_bin = np.floor((unroll[:, 0] - unroll[:, 0].min()) / axis_bin).astype(np.int64)
    count = np.bincount(s_bin)
    axis = np.stack([np.bincount(s_bin, coord[:, i]) for i in range(3)], 1) / np.maximum(count, 1)[:, None]
    flip = np.einsum("ni,ni->n", normal, axis[s_bin] - coord) < 0
    normal[flip] *= -1
    return normal.astype(np.float32)


def intensity_scales(las_paths, quantile=99.5, blocks=20, block_size=10_000, seed=0):
    """Per tunnel type intensity quantile from a few contiguous blocks of each scene (cheap I/O)."""
    rng = np.random.default_rng(seed)
    samples = {}
    for path in las_paths:
        rec, _, _ = read_las(path)
        starts = rng.integers(0, len(rec) - block_size, blocks)
        ttype = parse_scene_name(Path(path).stem)[0]
        samples.setdefault(ttype, []).extend(
            np.asarray(rec["intensity"][s : s + block_size], np.float32) for s in starts
        )
    return {t: float(np.percentile(np.concatenate(v), quantile)) for t, v in samples.items()}


def process_scene(las_path, output_root, scales, grid_size, normal_k):
    name = Path(las_path).stem
    ttype, seq, part = parse_scene_name(name)
    pts = load_points(las_path)
    keep, inverse = voxelize(pts["coord"], grid_size)
    centroid = pts["coord"].mean(0)
    coord = (pts["coord"][keep] - centroid).astype(np.float32)
    unroll = pts["unroll"][keep].astype(np.float32)
    strength = np.clip(pts["intensity"][keep] / scales[ttype], 0.0, 1.5)[:, None].astype(np.float32)

    scene_dir = Path(output_root) / name
    (scene_dir / "origin").mkdir(parents=True, exist_ok=True)
    np.save(scene_dir / "coord.npy", coord)
    np.save(scene_dir / "unroll.npy", unroll)
    np.save(scene_dir / "strength.npy", strength)
    np.save(scene_dir / "normal.npy", estimate_normals(coord, unroll, k=normal_k))
    np.save(scene_dir / "segment.npy", pts["segment"][keep].astype(np.int16))
    np.save(scene_dir / "origin" / "segment.npy", pts["segment"].astype(np.uint8))
    np.save(scene_dir / "origin" / "inverse.npy", inverse)
    return name, dict(
        type=ttype,
        seq=seq,
        part=part,
        num_points=int(len(inverse)),
        num_voxels=int(len(keep)),
        centroid=centroid.tolist(),
        s_range=[float(pts["unroll"][:, 0].min()), float(pts["unroll"][:, 0].max())],
        class_points=np.bincount(pts["segment"], minlength=12).tolist(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--las_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--grid_size", type=float, default=0.01)
    parser.add_argument("--normal_k", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    las_paths = sorted(str(p) for p in Path(args.las_root).glob("*.las"))
    os.makedirs(args.output_root, exist_ok=True)
    scales = intensity_scales(las_paths)
    print("intensity scales (p99.5):", scales)

    worker = partial(
        process_scene,
        output_root=args.output_root,
        scales=scales,
        grid_size=args.grid_size,
        normal_k=args.normal_k,
    )
    scenes = {}
    with ProcessPoolExecutor(args.num_workers) as pool:
        for i, (name, info) in enumerate(pool.map(worker, las_paths)):
            scenes[name] = info
            print(f"[{i + 1}/{len(las_paths)}] {name}: {info['num_points']} -> {info['num_voxels']}")

    meta = dict(grid_size=args.grid_size, intensity_scale=scales, scenes=scenes)
    with open(Path(args.output_root) / "meta.json", "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
