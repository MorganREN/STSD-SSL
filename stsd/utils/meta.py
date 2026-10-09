"""
Static STSD metadata.
"""

import re

CLASS_NAMES = (
    "tunnel_wall",
    "rail_track",
    "power_track",
    "walkway",
    "pipe",
    "cable",
    "attachment",
    "person",
    "big_bolt_hole",
    "small_bolt_hole",
    "signal_line",
    "joint",
)
RARE_CLASSES = (
    "attachment",
    "person",
    "big_bolt_hole",
    "small_bolt_hole",
    "signal_line",
    "joint",
)
TUNNEL_TYPES = dict(horse="horseshoe", rec="quasi_rectangular", round="circular")

_NAME_RE = re.compile(r"^(horse|rec|round)-(\d+)-(\d+)_(\d+)$")


def parse_scene_name(name):
    """'rec-1-13_4' -> ('quasi_rectangular', seq=13, part=4)."""
    m = _NAME_RE.match(name)
    assert m is not None, f"invalid STSD scene name: {name}"
    return TUNNEL_TYPES[m.group(1)], int(m.group(3)), int(m.group(4))
