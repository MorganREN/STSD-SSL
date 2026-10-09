"""
Minimal reader for STSD v1.1 LAS files (LAS 1.2, point format 3 + 32 B extra bytes).

The extra-byte VLR describes inx/iny/inz as "original coordinates", but they actually
hold the unrolled 2.5D coordinates (s: along track, c: arc length, r: radial offset),
while X/Y/Z hold the original 3D coordinates.
"""

import struct

import numpy as np

LAS_DTYPE = np.dtype(
    [
        ("X", "<i4"), ("Y", "<i4"), ("Z", "<i4"),
        ("intensity", "<u2"), ("flags", "u1"), ("cls", "u1"),
        ("scan_angle", "i1"), ("user", "u1"), ("psid", "<u2"), ("gps", "<f8"),
        ("r", "<u2"), ("g", "<u2"), ("b", "<u2"),
        ("inx", "<i8"), ("iny", "<i8"), ("inz", "<i8"), ("index2d", "<u8"),
    ]
)
UNROLL_SCALE = 1e-4


def read_las(path):
    """Return (records memmap, scale (3,), offset (3,))."""
    with open(path, "rb") as f:
        hdr = f.read(227)
    assert hdr[:4] == b"LASF" and hdr[104] == 3, f"unexpected LAS format: {path}"
    (offset_to_points,) = struct.unpack("<I", hdr[96:100])
    record_len, num_points = struct.unpack("<HI", hdr[105:111])
    assert record_len == LAS_DTYPE.itemsize, f"unexpected record length: {path}"
    scale = np.array(struct.unpack("<3d", hdr[131:155]))
    offset = np.array(struct.unpack("<3d", hdr[155:179]))
    records = np.memmap(path, LAS_DTYPE, "r", offset=offset_to_points, shape=(num_points,))
    return records, scale, offset


def load_points(path):
    """Load the fields used by STSD-SSL as a dict of arrays (float64 coordinates)."""
    rec, scale, offset = read_las(path)
    coord = np.stack([rec[k] * scale[i] + offset[i] for i, k in enumerate("XYZ")], 1)
    unroll = np.stack([rec[k] * UNROLL_SCALE for k in ("inx", "iny", "inz")], 1)
    return dict(
        coord=coord,
        unroll=unroll,
        intensity=np.asarray(rec["intensity"], dtype=np.float32),
        segment=np.asarray(rec["cls"], dtype=np.int16),
    )
