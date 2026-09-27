import pytest

from common.s3_keys import (
    DataKey,
    ResultKey,
    display_name,
    eval_key,
    eval_partition_location,
    parse_data_key,
    parse_result_key,
    recognition_key,
    recognition_partition_location,
    results_delete_prefix,
)


def test_parse_data_key():
    assert parse_data_key("data/layer_0/Animals/Giraffe/0.png") == DataKey(
        layer="0", category="Animals", label="Giraffe", variant="0", ext="png"
    )


def test_parse_data_key_rejects_bad_format():
    with pytest.raises(ValueError):
        parse_data_key("data/Animals/Giraffe/0.png")


def test_parse_result_key_recognition():
    key = "results/layer_0/Animals/Giraffe/0/stringent/recognition/recognition_3.txt"
    assert parse_result_key(key) == ResultKey(
        layer="0", category="Animals", label="Giraffe", variant="0",
        prompt_type="stringent", kind="recognition", round=3,
    )


def test_parse_result_key_eval():
    key = "results/layer_0/Animals/Giraffe/1/lenient/eval/eval_10.txt"
    assert parse_result_key(key) == ResultKey(
        layer="0", category="Animals", label="Giraffe", variant="1",
        prompt_type="lenient", kind="eval", round=10,
    )


def test_recognition_key_builds_expected_path():
    assert recognition_key("0", "Animals", "Giraffe", "0", "stringent", 3) == (
        "results/layer_0/Animals/Giraffe/0/stringent/recognition/recognition_3.txt"
    )


def test_eval_key_builds_expected_path():
    assert eval_key("0", "Animals", "Giraffe", "1", "lenient", 10) == (
        "results/layer_0/Animals/Giraffe/1/lenient/eval/eval_10.txt"
    )


def test_recognition_partition_location():
    assert recognition_partition_location("steering-visualization", "0", "Animals", "Giraffe", "0", "stringent") == (
        "s3://steering-visualization/results/layer_0/Animals/Giraffe/0/stringent/recognition/"
    )


def test_eval_partition_location():
    assert eval_partition_location("steering-visualization", "0", "Animals", "Giraffe", "1", "lenient") == (
        "s3://steering-visualization/results/layer_0/Animals/Giraffe/1/lenient/eval/"
    )


def test_results_delete_prefix():
    assert results_delete_prefix("0", "Animals", "Giraffe", "0") == "results/layer_0/Animals/Giraffe/0/"


def test_display_name_replaces_underscores_with_spaces():
    assert display_name("Marilyn_Monroe") == "Marilyn Monroe"


def test_display_name_leaves_plain_words_unchanged():
    assert display_name("Giraffe") == "Giraffe"
