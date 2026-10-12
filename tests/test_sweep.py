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
