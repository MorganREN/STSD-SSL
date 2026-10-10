"""
Weights & Biases run management.

One wandb run per experiment directory: the run id is stored in <save_path>/wandb_id on
first launch and reused on every restart, so Slurm requeues continue the same run.
Group / tags / config come from <save_path>/run_meta.json written by stsd.tools.sweep.
"""

import json
import uuid
from pathlib import Path

import wandb

META_TAGS = ("protocol", "method", "split", "fold", "budget")


def run_meta(save_path):
    path = Path(save_path) / "run_meta.json"
    return json.loads(path.read_text()) if path.exists() else {}


def run_id(save_path, create=True):
    path = Path(save_path) / "wandb_id"
    if path.exists():
        return path.read_text().strip()
    if not create:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(uuid.uuid4().hex[:12])
    return path.read_text().strip()


def init_wandb(cfg):
    save_path = Path(cfg.save_path)
    meta = run_meta(save_path)
    group = cfg.get("wandb_group") or "-".join(str(meta[k]) for k in ("protocol", "method", "split") if k in meta)
    return wandb.init(
        project=cfg.wandb_project,
        entity=cfg.get("wandb_entity"),
        id=run_id(save_path),
        resume="allow",
        name=meta.get("name", save_path.name),
        group=group or save_path.parent.name,
        job_type=meta.get("protocol"),
        tags=[f"{k}:{meta[k]}" for k in META_TAGS if k in meta],
        dir=str(save_path),
        config=dict(cfg=dict(cfg), run=meta),
        settings=wandb.Settings(api_key=cfg.wandb_key) if cfg.get("wandb_key") else None,
    )


def log_test_summary(save_path, project, metrics, entity=None):
    """Attach offline test metrics (stsd.tools.evaluate) to the run's summary."""
    rid = run_id(save_path, create=False)
    if rid is None:
        return False
    with wandb.init(project=project, entity=entity, id=rid, resume="allow", dir=str(save_path)) as run:
        run.summary.update({f"test/{k}": v for k, v in metrics.items()})
    return True
