"""
Paired comparisons between methods.

Runs are paired on (fold, subset, seed). Uncertainty comes from a hierarchical bootstrap:
resample folds, then paired runs within each resampled fold.
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PAIR_KEYS = ["fold", "subset", "seed"]


def paired_differences(df, method_a, method_b, metric="miou", by=()):
    """df rows: one run with columns method, fold, subset, seed, metric (+ `by` columns)."""
    keys = list(by) + PAIR_KEYS
    a = df[df.method == method_a].set_index(keys)[metric]
    b = df[df.method == method_b].set_index(keys)[metric]
    diff = (a - b).dropna().rename("diff").reset_index()
    assert len(diff), f"no paired runs between {method_a} and {method_b}"
    return diff


def hierarchical_bootstrap(diff, num_samples=10000, seed=0):
    rng = np.random.default_rng(seed)
    groups = [g["diff"].to_numpy() for _, g in diff.groupby("fold")]
    stats = np.empty(num_samples)
    for i in range(num_samples):
        picked = rng.integers(0, len(groups), len(groups))
        stats[i] = np.mean(
            np.concatenate([rng.choice(groups[k], len(groups[k])) for k in picked])
        )
    return stats


def segment_paired(scenes, metric, a, b, num_samples=10000, seed=0):
    """
    Same test segment scored under two settings (a, b: dicts of column filters, e.g.
    dict(method="B1", split="random")). Repeated tests of a segment are averaged first.
    Returns mean / median difference (a - b), bootstrap CI, share of segments where a > b.
    """
    select = lambda f: scenes.loc[np.logical_and.reduce([scenes[k] == v for k, v in f.items()])]
    pair = pd.concat(
        [select(a).groupby("scene")[metric].mean().rename("a"), select(b).groupby("scene")[metric].mean().rename("b")],
        axis=1,
    ).dropna()
    diff = (pair["a"] - pair["b"]).to_numpy()
    if len(diff) == 0:
        return None
    rng = np.random.default_rng(seed)
    boot = rng.choice(diff, (num_samples, len(diff))).mean(1)
    return dict(
        n=len(diff),
        mean=diff.mean(),
        median=np.median(diff),
        ci_low=np.percentile(boot, 2.5),
        ci_high=np.percentile(boot, 97.5),
        a_better=(diff > 0).mean(),
        wilcoxon_p=wilcoxon(diff).pvalue if np.any(diff != 0) else np.nan,
    )


def compare(df, method_a, method_b, metric="miou", by=(), num_samples=10000, seed=0):
    """Mean paired difference (a - b) with 95% CI, win rate and Wilcoxon p, per `by` group."""
    diff = paired_differences(df, method_a, method_b, metric, by)
    rows = []
    for key, g in (diff.groupby(list(by)) if by else [((), diff)]):
        boot = hierarchical_bootstrap(g, num_samples, seed)
        p = wilcoxon(g["diff"]).pvalue if len(g) > 1 and np.any(g["diff"] != 0) else np.nan
        rows.append(
            dict(
                **dict(zip(by, key if isinstance(key, tuple) else (key,))),
                a=method_a,
                b=method_b,
                n=len(g),
                mean=g["diff"].mean(),
                ci_low=np.percentile(boot, 2.5),
                ci_high=np.percentile(boot, 97.5),
                win_rate=(g["diff"] > 0).mean(),
                wilcoxon_p=p,
            )
        )
    return pd.DataFrame(rows)
