from unittest.mock import MagicMock

from botocore.exceptions import ClientError

from upload_experiment_results import Checkpoint, discover_checkpoints, upload_missing


def _make_run(tmp_path, category, label, layer, with_best=True, extra_files=()):
    layer_dir = tmp_path / category / label / f"layer-{layer}"
    layer_dir.mkdir(parents=True)
    if with_best:
        (layer_dir / "best.png").write_bytes(b"fake-png-bytes")
    for name in extra_files:
        (layer_dir / name).write_bytes(b"irrelevant")
    return layer_dir


def test_discover_checkpoints_finds_best_png_and_builds_correct_key(tmp_path):
    _make_run(
        tmp_path, "Animals", "giraffe", "8",
        extra_files=["final.png", "ema.png", "step_0000.png", "metrics.csv", "latents.pt"],
    )

    checkpoints = discover_checkpoints(tmp_path)

    assert len(checkpoints) == 1
    checkpoint = checkpoints[0]
    assert checkpoint.layer == "8"
    assert checkpoint.category == "Animals"
    assert checkpoint.label == "giraffe"
    assert checkpoint.s3_key == "data/layer_8/Animals/giraffe/0.png"
    assert checkpoint.local_path.name == "best.png"


def test_discover_checkpoints_skips_incomplete_run(tmp_path):
    _make_run(tmp_path, "Activities", "swimming", "8", with_best=False)

    checkpoints = discover_checkpoints(tmp_path)

    assert checkpoints == []


def test_discover_checkpoints_handles_multiple_layers_and_labels(tmp_path):
    _make_run(tmp_path, "Animals", "giraffe", "8")
    _make_run(tmp_path, "Animals", "giraffe", "16")
    _make_run(tmp_path, "People", "Marilyn_Monroe", "24")

    checkpoints = discover_checkpoints(tmp_path)
    keys = {c.s3_key for c in checkpoints}

    assert keys == {
        "data/layer_8/Animals/giraffe/0.png",
        "data/layer_16/Animals/giraffe/0.png",
        "data/layer_24/People/Marilyn_Monroe/0.png",
    }


def test_discover_checkpoints_returns_empty_for_missing_source_dir(tmp_path):
    assert discover_checkpoints(tmp_path / "does-not-exist") == []


def _not_found_error():
    return ClientError({"Error": {"Code": "404"}, "ResponseMetadata": {"HTTPStatusCode": 404}}, "HeadObject")


def test_upload_missing_skips_existing_and_uploads_new(tmp_path):
    existing = Checkpoint(layer="8", category="Animals", label="octopus", local_path=tmp_path / "best.png", s3_key="data/layer_8/Animals/octopus/0.png")
    missing = Checkpoint(layer="8", category="Animals", label="giraffe", local_path=tmp_path / "best.png", s3_key="data/layer_8/Animals/giraffe/0.png")
    s3 = MagicMock()

    def head_object(Bucket, Key):
        if Key == existing.s3_key:
            return {}
        raise _not_found_error()

    s3.head_object.side_effect = head_object

    uploaded, already_present = upload_missing(s3, "steering-visualization", [existing, missing])

    assert uploaded == [missing]
    assert already_present == [existing]
    s3.upload_file.assert_called_once_with(str(missing.local_path), "steering-visualization", missing.s3_key)


def test_upload_missing_dry_run_never_calls_upload_file(tmp_path):
    missing = Checkpoint(layer="8", category="Animals", label="giraffe", local_path=tmp_path / "best.png", s3_key="data/layer_8/Animals/giraffe/0.png")
    s3 = MagicMock()
    s3.head_object.side_effect = _not_found_error()

    uploaded, already_present = upload_missing(s3, "steering-visualization", [missing], dry_run=True)

    assert uploaded == [missing]
    assert already_present == []
    s3.upload_file.assert_not_called()
