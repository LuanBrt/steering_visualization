from unittest.mock import MagicMock

from botocore.exceptions import ClientError

from recognition.handler import object_exists, process_image


def _not_found_error():
    return ClientError(
        {"Error": {"Code": "404"}, "ResponseMetadata": {"HTTPStatusCode": 404}},
        "HeadObject",
    )


def _forbidden_error():
    return ClientError(
        {"Error": {"Code": "403"}, "ResponseMetadata": {"HTTPStatusCode": 403}},
        "HeadObject",
    )


def test_object_exists_true_when_head_object_succeeds():
    s3 = MagicMock()
    assert object_exists(s3, "bucket", "key") is True


def test_object_exists_false_on_404():
    s3 = MagicMock()
    s3.head_object.side_effect = _not_found_error()
    assert object_exists(s3, "bucket", "key") is False


def test_object_exists_reraises_other_errors():
    s3 = MagicMock()
    s3.head_object.side_effect = _forbidden_error()
    try:
        object_exists(s3, "bucket", "key")
        assert False, "expected ClientError to propagate"
    except ClientError:
        pass


def _bedrock_stub(response_text="giraffe"):
    bedrock = MagicMock()
    bedrock.converse.return_value = {"output": {"message": {"content": [{"text": response_text}]}}}
    return bedrock


def _s3_stub_no_existing_results(image_bytes=b"fake-image-bytes"):
    s3 = MagicMock()
    s3.get_object.return_value = {"Body": MagicMock(read=MagicMock(return_value=image_bytes), __enter__=lambda self: self, __exit__=lambda *a: None)}
    s3.head_object.side_effect = _not_found_error()
    return s3


def _glue_stub():
    glue = MagicMock()

    class _AlreadyExists(Exception):
        pass

    glue.exceptions.AlreadyExistsException = _AlreadyExists
    return glue


def test_process_image_writes_all_rounds_for_both_prompt_types():
    s3 = _s3_stub_no_existing_results()
    bedrock = _bedrock_stub()
    glue = _glue_stub()

    process_image(
        s3, bedrock, glue,
        bucket="steering-visualization",
        key="data/layer_0/Animals/Giraffe/0.png",
        num_rounds=2,
        model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        max_tokens=20,
        database="steering_visualization",
        table="recognition_results",
        stringent_prompt="What is in the image? One word.",
        lenient_template="What {category} is in the image if you had to guess? One word.",
    )

    # 2 prompt types x 2 rounds = 4 Bedrock calls and 4 S3 writes
    assert bedrock.converse.call_count == 4
    assert s3.put_object.call_count == 4
    written_keys = {call.kwargs["Key"] for call in s3.put_object.call_args_list}
    assert written_keys == {
        "results/layer_0/Animals/Giraffe/0/stringent/recognition/recognition_1.txt",
        "results/layer_0/Animals/Giraffe/0/stringent/recognition/recognition_2.txt",
        "results/layer_0/Animals/Giraffe/0/lenient/recognition/recognition_1.txt",
        "results/layer_0/Animals/Giraffe/0/lenient/recognition/recognition_2.txt",
    }
    # the lenient prompt must have the category substituted in
    lenient_call = next(
        c for c in bedrock.converse.call_args_list
        if "Animals" in c.kwargs["messages"][0]["content"][1]["text"]
    )
    assert lenient_call.kwargs["messages"][0]["content"][1]["text"] == (
        "What Animals is in the image if you had to guess? One word."
    )
    assert glue.create_partition.call_count == 4
    partition_values = {tuple(c.kwargs["PartitionInput"]["Values"]) for c in glue.create_partition.call_args_list}
    assert partition_values == {
        ("0", "Animals", "Giraffe", "0", "stringent"),
        ("0", "Animals", "Giraffe", "0", "lenient"),
    }


def test_process_image_skips_rounds_that_already_have_a_result():
    s3 = MagicMock()
    s3.get_object.return_value = {"Body": MagicMock(read=MagicMock(return_value=b"bytes"), __enter__=lambda self: self, __exit__=lambda *a: None)}
    s3.head_object.return_value = {}  # every HeadObject succeeds -> already computed
    bedrock = _bedrock_stub()
    glue = _glue_stub()

    process_image(
        s3, bedrock, glue,
        bucket="steering-visualization",
        key="data/layer_0/Animals/Giraffe/0.png",
        num_rounds=3,
        model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        max_tokens=20,
        database="steering_visualization",
        table="recognition_results",
        stringent_prompt="What is in the image? One word.",
        lenient_template="What {category} is in the image if you had to guess? One word.",
    )

    bedrock.converse.assert_not_called()
    s3.put_object.assert_not_called()


def test_process_image_skips_unsupported_extension_without_raising():
    s3 = MagicMock()
    bedrock = _bedrock_stub()
    glue = _glue_stub()

    process_image(
        s3, bedrock, glue,
        bucket="steering-visualization",
        key="data/layer_0/Animals/Giraffe/0.bmp",
        num_rounds=2,
        model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        max_tokens=20,
        database="steering_visualization",
        table="recognition_results",
        stringent_prompt="What is in the image? One word.",
        lenient_template="What {category} is in the image if you had to guess? One word.",
    )

    s3.get_object.assert_not_called()
    bedrock.converse.assert_not_called()


def test_process_image_continues_after_one_bedrock_failure():
    s3 = _s3_stub_no_existing_results()
    bedrock = MagicMock()
    bedrock.converse.side_effect = [
        Exception("throttled"),
        {"output": {"message": {"content": [{"text": "giraffe"}]}}},
    ]
    glue = _glue_stub()

    process_image(
        s3, bedrock, glue,
        bucket="steering-visualization",
        key="data/layer_0/Animals/Giraffe/0.png",
        num_rounds=1,
        model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        max_tokens=20,
        database="steering_visualization",
        table="recognition_results",
        stringent_prompt="What is in the image? One word.",
        lenient_template="What {category} is in the image if you had to guess? One word.",
    )

    # first call raised, second succeeded -> exactly one write
    assert s3.put_object.call_count == 1
