import os

import boto3

from common.glue_partitions import build_client as build_glue_client
from common.glue_partitions import delete_partition
from common.s3_keys import parse_data_key, results_delete_prefix

_PROMPT_TYPES = ("stringent", "lenient")


def process_deleted_image(
    s3_client,
    glue_client,
    *,
    bucket: str,
    key: str,
    database: str,
    recognition_table: str,
    evaluation_table: str,
) -> None:
    data_key = parse_data_key(key)
    # Scoped to this variant only -- deleting one exemplar image must not
    # touch the results of other images for the same label.
    prefix = results_delete_prefix(data_key.layer, data_key.category, data_key.label, data_key.variant)

    objects_to_delete = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        objects_to_delete.extend({"Key": obj["Key"]} for obj in page.get("Contents", []))

    if objects_to_delete:
        s3_client.delete_objects(Bucket=bucket, Delete={"Objects": objects_to_delete})

    for prompt_type in _PROMPT_TYPES:
        values = [data_key.layer, data_key.category, data_key.label, data_key.variant, prompt_type]
        delete_partition(glue_client, database, recognition_table, values)
        delete_partition(glue_client, database, evaluation_table, values)


def handler(event: dict, context) -> None:
    s3_client = boto3.client("s3")
    glue_client = build_glue_client()

    database = os.environ["GLUE_DATABASE"]
    recognition_table = os.environ["RECOGNITION_TABLE"]
    evaluation_table = os.environ["EVALUATION_TABLE"]

    for record in event["Records"]:
        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]
        process_deleted_image(
            s3_client, glue_client,
            bucket=bucket, key=key, database=database,
            recognition_table=recognition_table, evaluation_table=evaluation_table,
        )
