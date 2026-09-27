from unittest.mock import MagicMock

from cleanup.handler import process_deleted_image


def _glue_stub():
    glue = MagicMock()

    class _EntityNotFound(Exception):
        pass

    glue.exceptions.EntityNotFoundException = _EntityNotFound
    return glue


def _paginator_with_pages(*pages):
    paginator = MagicMock()
    paginator.paginate.return_value = pages
    return paginator


def test_process_deleted_image_deletes_only_this_variants_objects_and_partitions():
    s3 = MagicMock()
    s3.get_paginator.return_value = _paginator_with_pages(
        {
            "Contents": [
                {"Key": "results/layer_0/Animals/Giraffe/0/stringent/recognition/recognition_1.txt"},
                {"Key": "results/layer_0/Animals/Giraffe/0/stringent/eval/eval_1.txt"},
                {"Key": "results/layer_0/Animals/Giraffe/0/lenient/recognition/recognition_1.txt"},
                {"Key": "results/layer_0/Animals/Giraffe/0/lenient/eval/eval_1.txt"},
            ]
        }
    )
    glue = _glue_stub()

    process_deleted_image(
        s3, glue,
        bucket="steering-visualization",
        key="data/layer_0/Animals/Giraffe/0.png",
        database="steering_visualization",
        recognition_table="recognition_results",
        evaluation_table="evaluation_results",
    )

    # only variant "0" is listed/deleted -- other variants of the same label
    # (e.g. Giraffe/1) must be untouched
    s3.get_paginator.assert_called_once_with("list_objects_v2")
    list_call = s3.get_paginator.return_value.paginate.call_args
    assert list_call.kwargs["Prefix"] == "results/layer_0/Animals/Giraffe/0/"

    s3.delete_objects.assert_called_once_with(
        Bucket="steering-visualization",
        Delete={
            "Objects": [
                {"Key": "results/layer_0/Animals/Giraffe/0/stringent/recognition/recognition_1.txt"},
                {"Key": "results/layer_0/Animals/Giraffe/0/stringent/eval/eval_1.txt"},
                {"Key": "results/layer_0/Animals/Giraffe/0/lenient/recognition/recognition_1.txt"},
                {"Key": "results/layer_0/Animals/Giraffe/0/lenient/eval/eval_1.txt"},
            ]
        },
    )
    delete_calls = glue.delete_partition.call_args_list
    assert len(delete_calls) == 4  # 2 tables x 2 prompt_types
    called_pairs = {(c.kwargs["TableName"], tuple(c.kwargs["PartitionValues"])) for c in delete_calls}
    assert called_pairs == {
        ("recognition_results", ("0", "Animals", "Giraffe", "0", "stringent")),
        ("recognition_results", ("0", "Animals", "Giraffe", "0", "lenient")),
        ("evaluation_results", ("0", "Animals", "Giraffe", "0", "stringent")),
        ("evaluation_results", ("0", "Animals", "Giraffe", "0", "lenient")),
    }


def test_process_deleted_image_skips_delete_objects_when_nothing_to_remove():
    s3 = MagicMock()
    s3.get_paginator.return_value = _paginator_with_pages({})  # no "Contents" key
    glue = _glue_stub()

    process_deleted_image(
        s3, glue,
        bucket="steering-visualization",
        key="data/layer_0/Animals/Giraffe/0.png",
        database="steering_visualization",
        recognition_table="recognition_results",
        evaluation_table="evaluation_results",
    )

    s3.delete_objects.assert_not_called()
    # partitions are still (idempotently) removed even if the files were already gone
    assert glue.delete_partition.call_count == 4
