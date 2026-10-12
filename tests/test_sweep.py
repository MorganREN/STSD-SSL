import json
import shlex
import sys

from stsd.tools import sweep


def test_launch_saves_metadata_beside_overridden_training_output(tmp_path, monkeypatch):
    run = sweep.matrix("a_leakage")[0]
    original = tmp_path / "original"
    output = tmp_path / "custom output"
    run["save_path"] = str(original)
    commands = []

    def call(command):
        commands.append(command)
        return 0

    monkeypatch.setattr(sweep.subprocess, "call", call)
    extra = shlex.quote(f"save_path={output}")
    assert sweep.launch(run, extra=extra) == 0
    assert json.loads((output / "run_meta.json").read_text()) == dict(
        name=run["name"], **run["meta"]
    )
    assert not original.exists()
    assert commands[0][-1] == f"save_path={output}"


def test_save_path_cli_reports_the_last_override(tmp_path, monkeypatch, capsys):
    output = tmp_path / "custom output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sweep",
            "--matrix", "a_leakage",
            "--save-path", "0",
            "--extra", f"save_path=earlier {shlex.quote(f'save_path={output}')}",
        ],
    )
    sweep.main()
    assert capsys.readouterr().out.strip() == str(output)


def test_matrices_have_unique_outputs_and_dapt_dependencies_are_isolated():
    names = {m: [r["name"] for r in sweep.matrix(m)] for m in sweep.MATRICES}
    for m, rows in names.items():
        assert len(rows) == len(set(rows)), m
    paths = [r["save_path"] for r in sweep.matrix("a_dapt")]
    assert len(paths) == len(set(paths))  # block and LOTO pretraining no longer collide
    # matrices that can run before DAPT never reference DAPT weights
    for m in ("a_leakage", "a_budget", "a_linear", "a_loto", "a_brunklaus_repro"):
        assert all("exp/stsd/pretrain" not in r["options"].get("weight", "") for r in sweep.matrix(m)), m
    # gate rows are a subset of the full label-efficiency matrices (shared runs, skipped when DONE)
    assert set(names["a_gate"]) <= set(names["a_budget"]) | set(names["a_budget_b3"])


def test_launch_skips_done_runs_and_marks_success(tmp_path, monkeypatch):
    run = dict(sweep.matrix("a_gate")[0], save_path=str(tmp_path / "run"))
    calls = []
    monkeypatch.setattr(sweep.subprocess, "call", lambda cmd: calls.append(cmd) or 0)
    assert sweep.launch(run) == 0 and (tmp_path / "run" / "DONE").exists()
    assert sweep.launch(run) == 0 and len(calls) == 1


def test_dapt_length_checkpoints_exist_under_save_freq():
    import re
    save_freq = int(re.search(r"CheckpointSaver\", save_freq=(\d+)", open("configs/stsd/pretrain-sonata-stsd-dapt.py").read()).group(1))
    assert all(n % save_freq == 0 for n in sweep.DAPT_LENGTHS)
