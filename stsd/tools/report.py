"""
Paper A analysis from evaluate.py outputs (markdown to stdout).

    python -m stsd.tools.report --runs exp/results/ft.csv --scenes exp/results/ft_scenes.csv

Sections (each skipped when its runs are missing):
  1. full-label results per split and method (mean / sd over folds)
  2. paired method comparisons (B1 vs B0, B3 vs B1) per split and budget
  3. leakage: random vs block, fold-level and segment-paired
  4. label-efficiency curve on the block protocol
  5. decision gate: B3 - B1 on fold 0 at b <= 12 (threshold 2 mIoU-11 points)
"""

import argparse

import pandas as pd

from stsd.evaluation import compare
from stsd.evaluation.stats import segment_paired

METRICS = ["miou_no_person", "miou", "miou_rare_no_person", "bf_joint"]
GATE_THRESHOLD = 2.0


def pct(df, cols):
    df = df.copy()
    df[cols] = df[cols] * 100
    return df


def md(df, digits=2):
    """Minimal markdown table (index + columns), no extra dependency."""
    df = df.round(digits).reset_index()
    cells = [[str(c) for c in df.columns]] + [["" if pd.isna(v) else str(v) for v in row] for row in df.itertuples(index=False)]
    lines = ["| " + " | ".join(r) + " |" for r in cells]
    return "\n".join([lines[0], "|" + "---|" * len(df.columns)] + lines[1:])


def full_label_table(runs):
    full = runs[(runs.protocol == "ft") & (runs.budget.astype(str) == "all")]
    g = full.groupby(["split", "method"])[METRICS]
    table = g.mean().join(g.std().add_suffix("_sd")[["miou_no_person_sd", "miou_sd"]]).join(g.size().rename("n"))
    return md(table)


def method_comparisons(runs):
    ft = runs[runs.protocol == "ft"]
    out = []
    for a, b in (("B1", "B0"), ("B3", "B1"), ("B3", "B0")):
        sub = ft[ft.method.isin([a, b])]
        if sub.method.nunique() < 2:
            continue
        for metric in ("miou_no_person", "miou_rare_no_person", "bf_joint"):
            try:
                res = compare(sub, a, b, metric, by=("split", "budget"))
            except AssertionError:
                continue
            out.append(res.assign(metric=metric))
    if not out:
        return None
    res = pd.concat(out)
    res[["mean", "ci_low", "ci_high"]] *= 100
    return md(res.set_index(["metric", "a", "b", "split", "budget"])[["n", "mean", "ci_low", "ci_high", "win_rate", "wilcoxon_p"]], 3)


def leakage(runs, scenes):
    full = runs[(runs.protocol == "ft") & (runs.budget.astype(str) == "all") & runs.split.isin(["random", "block"])]
    rows = []
    for method, g in full.groupby("method"):
        means = g.groupby("split")[["miou_no_person", "miou"]].mean()
        if len(means) < 2:
            continue
        row = dict(method=method, fold_level_11=means.loc["random", "miou_no_person"] - means.loc["block", "miou_no_person"],
                   fold_level_12=means.loc["random", "miou"] - means.loc["block", "miou"])
        if scenes is not None:
            seg = segment_paired(
                scenes[(scenes.protocol == "ft") & (scenes.budget.astype(str) == "all")],
                "miou_no_person",
                dict(method=method, split="random"),
                dict(method=method, split="block"),
            )
            if seg:
                row.update({f"segment_{k}": v for k, v in seg.items()})
        rows.append(row)
    if not rows:
        return None
    df = pd.DataFrame(rows).set_index("method")
    scale = [c for c in df.columns if c.startswith("fold_level") or c in ("segment_mean", "segment_median", "segment_ci_low", "segment_ci_high")]
    df[scale] *= 100
    return md(df, 3)


def budget_curve(runs):
    ft = runs[(runs.protocol == "ft") & (runs.split == "block")]
    if ft.budget.astype(str).nunique() < 2:
        return None
    order = {"3": 0, "12": 1, "48": 2, "all": 3}
    g = ft.assign(budget=ft.budget.astype(str)).groupby(["budget", "method"])["miou_no_person"]
    table = (g.mean() * 100).unstack("method").join((g.size()).unstack("method").add_suffix("_n"))
    return md(table.sort_index(key=lambda idx: idx.map(order)))


def gate(runs):
    sub = runs[(runs.protocol == "ft") & (runs.split == "block") & (runs.fold == 0) & runs.method.isin(["B1", "B3"])]
    sub = sub[sub.budget.astype(str).isin(["3", "12"])]
    if sub.method.nunique() < 2:
        return None
    res = compare(sub, "B3", "B1", "miou_no_person", by=("budget",))
    res[["mean", "ci_low", "ci_high"]] *= 100
    b12 = res[res.budget.astype(str) == "12"]
    verdict = "n/a"
    if len(b12):
        r = b12.iloc[0]
        verdict = "PASS" if r["mean"] >= GATE_THRESHOLD and (r["win_rate"] == 1.0) else "FAIL"
    return md(res.set_index("budget")[["n", "mean", "ci_low", "ci_high", "win_rate"]], 3) + f"\n\nGate (b=12, B3-B1 >= {GATE_THRESHOLD} and all pairs positive): **{verdict}**"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True, help="CSV from stsd.tools.evaluate --output")
    parser.add_argument("--scenes", default=None, help="CSV from stsd.tools.evaluate --scene-output")
    args = parser.parse_args()

    runs = pd.read_csv(args.runs)  # metrics stay fractions; each table scales to % / pp
    scenes = pd.read_csv(args.scenes) if args.scenes else None

    sections = [
        ("1. Full-label results (%, mean over folds)", lambda: full_label_table(pct(runs, METRICS))),
        ("2. Paired method comparisons (pp)", lambda: method_comparisons(runs)),
        ("3. Leakage: random minus block (pp)", lambda: leakage(runs, scenes)),
        ("4. Label efficiency, block protocol (mIoU-11 %)", lambda: budget_curve(runs)),
        ("5. Decision gate, fold 0 (mIoU-11 pp)", lambda: gate(runs)),
    ]
    for title, build in sections:
        body = build()
        if body is not None:
            print(f"## {title}\n\n{body}\n")


if __name__ == "__main__":
    main()
