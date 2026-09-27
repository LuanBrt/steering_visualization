from unittest.mock import MagicMock

from botocore.exceptions import ClientError

from evaluation.handler import process_record

_STRINGENT_PROMPT = "What is in the image? One word."
_LENIENT_TEMPLATE = "What {category} is in the image if you had to guess? One word."
_EVAL_TEMPLATE = (
    "REQUEST: {request}\nRESPONSE: {response}\nCONCEPT: {concept}\n"
    'End your answer with "Result: 0" or "Result: 1".'
)


def _not_found_error():
    return ClientError({"Error": {"Code": "404"}, "ResponseMetadata": {"HTTPStatusCode": 404}}, "HeadObject")


def _glue_stub():
    glue = MagicMock()

    class _AlreadyExists(Exception):
        pass

    glue.exceptions.AlreadyExistsException = _AlreadyExists
    return glue


def _common_kwargs():
    return dict(
        model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        max_tokens=200,
        database="steering_visualization",
        table="evaluation_results",
        stringent_prompt=_STRINGENT_PROMPT,
        lenient_template=_LENIENT_TEMPLATE,
        eval_prompt_template=_EVAL_TEMPLATE,
    )


def test_process_record_ignores_eval_files_to_avoid_self_trigger():
    s3 = MagicMock()
    bedrock = MagicMock()
    glue = _glue_stub()

    process_record(
        s3, bedrock, glue,
        bucket="steering-visualization",
        key="results/layer_0/Animals/Giraffe/0/stringent/eval/eval_1.txt",
        **_common_kwargs(),
    )

    s3.get_object.assert_not_called()
    bedrock.converse.assert_not_called()


def test_process_record_skips_when_eval_already_exists():
    s3 = MagicMock()
    s3.head_object.return_value = {}  # eval file already exists
    bedrock = MagicMock()
    glue = _glue_stub()

    process_record(
        s3, bedrock, glue,
        bucket="steering-visualization",
        key="results/layer_0/Animals/Giraffe/0/stringent/recognition/recognition_1.txt",
        **_common_kwargs(),
    )

    bedrock.converse.assert_not_called()
    s3.put_object.assert_not_called()


def test_process_record_writes_digit_and_registers_partition():
    s3 = MagicMock()
    s3.head_object.side_effect = _not_found_error()  # eval file does not exist yet
    s3.get_object.return_value = {
        "Body": MagicMock(read=MagicMock(return_value=b"giraffe"), __enter__=lambda self: self, __exit__=lambda *a: None)
    }
    bedrock = MagicMock()
    bedrock.converse.return_value = {"output": {"message": {"content": [{"text": "reasoning...\nResult: 1"}]}}}
    glue = _glue_stub()

    process_record(
        s3, bedrock, glue,
        bucket="steering-visualization",
        key="results/layer_0/Animals/Giraffe/2/lenient/recognition/recognition_4.txt",
        **_common_kwargs(),
    )

    s3.put_object.assert_called_once_with(
        Bucket="steering-visualization",
        Key="results/layer_0/Animals/Giraffe/2/lenient/eval/eval_4.txt",
        Body=b"1",
    )
    # the judge prompt must carry the reconstructed request, the recognition
    # output, and the pure label (never the variant) as the concept
    judge_prompt = bedrock.converse.call_args.kwargs["messages"][0]["content"][0]["text"]
    assert "What Animals is in the image if you had to guess? One word." in judge_prompt
    assert "giraffe" in judge_prompt
    assert "CONCEPT: Giraffe\n" in judge_prompt
    glue.create_partition.assert_called_once()
    partition_values = glue.create_partition.call_args.kwargs["PartitionInput"]["Values"]
    assert partition_values == ["0", "Animals", "Giraffe", "2", "lenient"]


def test_process_record_does_not_crash_when_judge_output_has_no_result_digit():
    # A verbose judge model can truncate its reasoning before reaching
    # "Result: 0/1" if maxTokens is too low -- this must be logged and
    # skipped, not propagate and fail the whole invocation.
    s3 = MagicMock()
    s3.head_object.side_effect = _not_found_error()
    s3.get_object.return_value = {
        "Body": MagicMock(read=MagicMock(return_value=b"giraffe"), __enter__=lambda self: self, __exit__=lambda *a: None)
    }
    bedrock = MagicMock()
    bedrock.converse.return_value = {"output": {"message": {"content": [{"text": "some truncated reasoning without a verdict"}]}}}
    glue = _glue_stub()

    process_record(
        s3, bedrock, glue,
        bucket="steering-visualization",
        key="results/layer_0/Animals/Giraffe/2/lenient/recognition/recognition_4.txt",
        **_common_kwargs(),
    )

    s3.put_object.assert_not_called()
    glue.create_partition.assert_not_called()


def test_process_record_uses_natural_language_label_in_concept():
    # A snake_cased label (e.g. "Marilyn_Monroe" from an upstream generator)
    # must read naturally in the judge prompt, not leak the underscore.
    s3 = MagicMock()
    s3.head_object.side_effect = _not_found_error()
    s3.get_object.return_value = {
        "Body": MagicMock(read=MagicMock(return_value=b"a woman"), __enter__=lambda self: self, __exit__=lambda *a: None)
    }
    bedrock = MagicMock()
    bedrock.converse.return_value = {"output": {"message": {"content": [{"text": "Result: 1"}]}}}
    glue = _glue_stub()

    process_record(
        s3, bedrock, glue,
        bucket="steering-visualization",
        key="results/layer_16/People/Marilyn_Monroe/0/stringent/recognition/recognition_1.txt",
        **_common_kwargs(),
    )

    judge_prompt = bedrock.converse.call_args.kwargs["messages"][0]["content"][0]["text"]
    assert "Marilyn Monroe" in judge_prompt
    assert "Marilyn_Monroe" not in judge_prompt
