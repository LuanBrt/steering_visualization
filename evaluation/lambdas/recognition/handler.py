import os

import boto3
from botocore.exceptions import ClientError

from common.bedrock_client import build_client as build_bedrock_client
from common.bedrock_client import converse_with_image
from common.glue_partitions import build_client as build_glue_client
from common.glue_partitions import ensure_partition
from common.glue_schema import RECOGNITION_COLUMNS
from common.s3_keys import display_name, parse_data_key, recognition_key, recognition_partition_location

_EXTENSION_TO_FORMAT = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}


def object_exists(s3_client, bucket: str, key: str) -> bool:
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        if exc.response["ResponseMetadata"]["HTTPStatusCode"] == 404:
            return False
        raise


def process_image(
    s3_client,
    bedrock_client,
    glue_client,
    *,
    bucket: str,
    key: str,
    num_rounds: int,
    model_id: str,
    max_tokens: int,
    database: str,
    table: str,
    stringent_prompt: str,
    lenient_template: str,
) -> None:
    data_key = parse_data_key(key)
    image_format = _EXTENSION_TO_FORMAT.get(data_key.ext.lower())
    if image_format is None:
        print(f"unsupported extension {data_key.ext!r} for key {key!r}, skipping")
        return

    with s3_client.get_object(Bucket=bucket, Key=key)["Body"] as body:
        image_bytes = body.read()

    prompts = {
        "stringent": stringent_prompt,
        "lenient": lenient_template.format(category=display_name(data_key.category)),
    }

    for prompt_type, prompt_text in prompts.items():
        for round_number in range(1, num_rounds + 1):
            _process_round(
                s3_client, bedrock_client, glue_client,
                bucket=bucket, data_key=data_key, prompt_type=prompt_type, prompt_text=prompt_text,
                round_number=round_number, image_bytes=image_bytes, image_format=image_format,
                model_id=model_id, max_tokens=max_tokens, database=database, table=table,
            )


def _process_round(
    s3_client, bedrock_client, glue_client, *,
    bucket, data_key, prompt_type, prompt_text, round_number,
    image_bytes, image_format, model_id, max_tokens, database, table,
) -> None:
    out_key = recognition_key(data_key.layer, data_key.category, data_key.label, data_key.variant, prompt_type, round_number)
    if object_exists(s3_client, bucket, out_key):
        return

    try:
        output_text = converse_with_image(bedrock_client, model_id, image_bytes, image_format, prompt_text, max_tokens)
    except Exception as exc:  # noqa: BLE001 - one bad round must not stop the other 19
        print(f"recognition call failed for {out_key!r}: {exc}")
        return

    s3_client.put_object(Bucket=bucket, Key=out_key, Body=output_text.encode("utf-8"))
    location = recognition_partition_location(bucket, data_key.layer, data_key.category, data_key.label, data_key.variant, prompt_type)
    ensure_partition(
        glue_client, database, table, RECOGNITION_COLUMNS,
        [data_key.layer, data_key.category, data_key.label, data_key.variant, prompt_type], location,
    )


def handler(event: dict, context) -> None:
    s3_client = boto3.client("s3")
    bedrock_client = build_bedrock_client()
    glue_client = build_glue_client()

    num_rounds = int(os.environ["NUM_ROUNDS"])
    model_id = os.environ["RECOGNITION_MODEL_ID"]
    max_tokens = int(os.environ["RECOGNITION_MAX_TOKENS"])
    database = os.environ["GLUE_DATABASE"]
    table = os.environ["RECOGNITION_TABLE"]
    stringent_prompt = os.environ["STRINGENT_PROMPT"]
    lenient_template = os.environ["LENIENT_PROMPT_TEMPLATE"]

    for record in event["Records"]:
        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]
        process_image(
            s3_client, bedrock_client, glue_client,
            bucket=bucket, key=key, num_rounds=num_rounds, model_id=model_id, max_tokens=max_tokens,
            database=database, table=table, stringent_prompt=stringent_prompt, lenient_template=lenient_template,
        )
