"""
STSD-specific transforms, registered into Pointcept's TRANSFORMS.
"""

import numpy as np

from pointcept.datasets.transform import TRANSFORMS, index_operator

from stsd.utils import parse_scene_name


@TRANSFORMS.register_module()
class StrengthToColor(object):
    """Fill the 3 colour channels with normalised intensity, matching Sonata's 9-ch input layout."""

    def __init__(self, max_value=1.0):
        self.max_value = max_value

    def __call__(self, data_dict):
        strength = np.clip(np.asarray(data_dict["strength"], np.float32), 0, self.max_value)
        data_dict["color"] = np.repeat(strength / self.max_value * 255.0, 3, axis=1)
        return data_dict


@TRANSFORMS.register_module()
class LegacyIntensityToColor(object):
    """
    Brunklaus & Reiterer (2026) colour input: raw intensity / 4096 * 255 for every tunnel
    (so 8-bit circular-tunnel intensities end up in 0..16). Raw intensity is recovered
    from our per-type normalised `strength` via the per-type scales in meta.json.
    Only used to reproduce their numbers.
    """

    def __init__(self, scales, divisor=4096.0):
        self.scales = scales
        self.divisor = divisor

    def __call__(self, data_dict):
        ttype = parse_scene_name(data_dict["name"])[0]
        raw = np.asarray(data_dict["strength"], np.float32) * self.scales[ttype]
        data_dict["color"] = np.repeat(np.clip(raw / self.divisor, 0, 1) * 255.0, 3, axis=1)
        return data_dict


@TRANSFORMS.register_module()
class RingWindowCrop(object):
    """
    Crop a window of `length` metres along the tunnel (unrolled s) covering the full ring.

    mode="random": one random window (training).
    mode="center":   the central window.
    mode="fraction": window placed at `data_dict["window_fraction"]` in [0, 1] (validation).
    mode="all":      sliding windows with `stride`, returned as a list (test-time `crop`).
    """

    def __init__(self, length=2.5, mode="random", stride=1.25):
        assert mode in ["random", "center", "fraction", "all"]
        self.length = length
        self.mode = mode
        self.stride = stride

    def _window(self, data_dict, start):
        s = np.asarray(data_dict["unroll"][:, 0])
        end = start + self.length
        upper = s <= end if end >= s.max() else s < end  # last window keeps the far end point
        return np.flatnonzero((s >= start) & upper)

    def __call__(self, data_dict):
        s = np.asarray(data_dict["unroll"][:, 0])
        s_min, s_max = float(s.min()), float(s.max())
        span = max(s_max - s_min - self.length, 0.0)
        if self.mode == "random":
            return index_operator(data_dict, self._window(data_dict, s_min + np.random.rand() * span))
        if self.mode == "center":
            return index_operator(data_dict, self._window(data_dict, s_min + span / 2))
        if self.mode == "fraction":
            start = s_min + data_dict.pop("window_fraction", 0.5) * span
            return index_operator(data_dict, self._window(data_dict, start))

        if "index" in data_dict and "index" not in data_dict["index_valid_keys"]:
            # test-mode GridSample attaches `index` without registering it as indexable
            data_dict["index_valid_keys"] = list(data_dict["index_valid_keys"]) + ["index"]
        starts = np.arange(0.0, span + 1e-6, self.stride)
        if span - starts[-1] > 1e-3:
            starts = np.append(starts, span)
        parts = []
        for start in s_min + starts:
            index = self._window(data_dict, start)
            if len(index) > 0:
                parts.append(index_operator(data_dict, index, duplicate=True))
        return parts
