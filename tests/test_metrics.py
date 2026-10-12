import numpy as np
import pandas as pd

from stsd.evaluation import scores
from stsd.evaluation.stats import segment_paired
from stsd.utils import CLASS_NAMES


def test_no_person_mean_ignores_person_class():
    conf = np.eye(12, dtype=np.int64) * 100
    person = CLASS_NAMES.index("person")
    conf[person, person] = 1
    conf[person, 0] = 99  # person almost always wrong
    s = scores(conf)
    assert s["miou"] < s["miou_no_person"]
    assert np.isclose(s["miou_no_person"], np.nanmean([v for c, v in s["iou"].items() if c != "person"]))


def test_segment_paired_averages_repeated_tests():
    scenes = pd.DataFrame(
        dict(scene=["a", "a", "b", "a", "b"], split=["random", "random", "random", "block", "block"],
             method="B1", miou=[0.9, 0.7, 0.6, 0.7, 0.6])
    )
    res = segment_paired(scenes, "miou", dict(split="random"), dict(split="block"))
    assert res["n"] == 2 and np.isclose(res["mean"], (0.8 - 0.7 + 0.0) / 2)
