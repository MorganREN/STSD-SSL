import sys

import pytest
from pointcept.utils.config import Config

from tools import train


@pytest.mark.parametrize("smoke_test", [False, True])
def test_training_entry_preserves_checkpoint_hooks_and_records_test_mode(
    tmp_path, monkeypatch, smoke_test
):
    output = tmp_path / "output"
    config_path = tmp_path / "input.py"
    hooks = [
        dict(type="CheckpointLoader"),
        dict(type="CheckpointSaver", save_freq=1),
        dict(type="PreciseEvaluator", test_last=False),
    ]
    settings = dict(
        seed=0,
        epoch=1,
        eval_epoch=1,
        resume=False,
        save_path=str(output),
        smoke_test=smoke_test,
        data=dict(train=dict()),
        hooks=hooks,
    )
    config_path.write_text("\n".join(f"{key} = {value!r}" for key, value in settings.items()))
    launches = []
    monkeypatch.setattr(train, "launch", lambda worker, **kwargs: launches.append(kwargs))
    monkeypatch.setattr(sys, "argv", ["train.py", "--config-file", str(config_path)])

    train.main()

    expected = hooks[:2] if smoke_test else hooks
    assert len(launches) == 1
    assert launches[0]["cfg"][0].hooks == expected
    assert Config.fromfile(str(output / "config.py")).hooks == expected
