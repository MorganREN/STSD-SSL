from stsd.engines.tracking import run_id


def test_run_id_is_created_once_and_reused(tmp_path):
    assert run_id(tmp_path, create=False) is None
    first = run_id(tmp_path)
    assert run_id(tmp_path) == first == (tmp_path / "wandb_id").read_text()
