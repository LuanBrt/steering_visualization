import os

import boto3

from common.bedrock_client import build_client as build_bedrock_client
from common.bedrock_client import converse_text, extract_result_digit
from common.glue_partitions import build_client as build_glue_client
from common.glue_partitions import ensure_partition
from common.glue_schema import EVALUATION_COLUMNS
from common.s3_keys import display_name, eval_key, eval_partition_location, parse_result_key
from recognition.handler import object_exists


def _reconstruct_request(prompt_type: str, category: str, stringent_prompt: str, lenient_template: str) -> str:
    if prompt_type == "stringent":
        return stringent_prompt
    return lenient_template.format(category=display_name(category))


def process_record(
    s3_client,
    bedrock_client,
    glue_client,
    *,
    bucket: str,
    key: str,
    model_id: str,
    max_tokens: int,
    database: str,
    table: str,
    stringent_prompt: str,
    lenient_template: str,
    eval_prompt_template: str,
) -> None:
    filename = key.rsplit("/", 1)[-1]
    if filename.startswith("eval_"):
        return  # avoid self-trigger: our own writes also match the S3 notification filter

    result_key = parse_result_key(key)
    out_key = eval_key(
        result_key.layer, result_key.category, result_key.label, result_key.variant, result_key.prompt_type, result_key.round
    )
    if object_exists(s3_client, bucket, out_key):
        return

    with s3_client.get_object(Bucket=bucket, Key=key)["Body"] as body:
        recognition_output = body.read().decode("utf-8")

    # {concept} is the pure label (the concept being steered toward), never
    # the variant -- the judge must never see which specific exemplar image
    # produced this response, only what concept it should be checking for.
    # display_name() turns a snake_cased label (e.g. "Marilyn_Monroe") into
    # natural-language text ("Marilyn Monroe") for the prompt sentence.
    request_text = _reconstruct_request(result_key.prompt_type, result_key.category, stringent_prompt, lenient_template)
    judge_prompt = eval_prompt_template.format(
        request=request_text, response=recognition_output, concept=display_name(result_key.label)
    )

    try:
        judge_output = converse_text(bedrock_client, model_id, judge_prompt, max_tokens)
        digit = extract_result_digit(judge_output)
    except Exception as exc:  # noqa: BLE001 - one bad judge call must not crash the whole invocation
        print(f"evaluation call failed for {out_key!r}: {exc}")
        return

    s3_client.put_object(Bucket=bucket, Key=out_key, Body=digit.encode("utf-8"))
    location = eval_partition_location(
        bucket, result_key.layer, result_key.category, result_key.label, result_key.variant, result_key.prompt_type
    )
    ensure_partition(
        glue_client, database, table, EVALUATION_COLUMNS,
        [result_key.layer, result_key.category, result_key.label, result_key.variant, result_key.prompt_type], location,
    )


def handler(event: dict, context) -> None:
    s3_client = boto3.client("s3")
    bedrock_client = build_bedrock_client()
    glue_client = build_glue_client()

    model_id = os.environ["EVALUATION_MODEL_ID"]
    max_tokens = int(os.environ["EVALUATION_MAX_TOKENS"])
    database = os.environ["GLUE_DATABASE"]
    table = os.environ["EVALUATION_TABLE"]
    stringent_prompt = os.environ["STRINGENT_PROMPT"]
    lenient_template = os.environ["LENIENT_PROMPT_TEMPLATE"]
    eval_prompt_template = os.environ["EVAL_PROMPT_TEMPLATE"]

    for record in event["Records"]:
        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]
        process_record(
            s3_client, bedrock_client, glue_client,
            bucket=bucket, key=key, model_id=model_id, max_tokens=max_tokens,
            database=database, table=table, stringent_prompt=stringent_prompt,
            lenient_template=lenient_template, eval_prompt_template=eval_prompt_template,
        )
