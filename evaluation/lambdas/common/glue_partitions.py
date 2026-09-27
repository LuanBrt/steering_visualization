import boto3

from common.glue_schema import INPUT_FORMAT, OUTPUT_FORMAT, SERIALIZATION_LIBRARY


def build_client():
    return boto3.client("glue")


def _storage_descriptor(columns: list[dict], location: str) -> dict:
    return {
        "Columns": columns,
        "Location": location,
        "InputFormat": INPUT_FORMAT,
        "OutputFormat": OUTPUT_FORMAT,
        "SerdeInfo": {"SerializationLibrary": SERIALIZATION_LIBRARY},
    }


def ensure_partition(client, database: str, table: str, columns: list[dict], values: list[str], location: str) -> None:
    try:
        client.create_partition(
            DatabaseName=database,
            TableName=table,
            PartitionInput={
                "Values": values,
                "StorageDescriptor": _storage_descriptor(columns, location),
            },
        )
    except client.exceptions.AlreadyExistsException:
        pass


def delete_partition(client, database: str, table: str, values: list[str]) -> None:
    try:
        client.delete_partition(DatabaseName=database, TableName=table, PartitionValues=values)
    except client.exceptions.EntityNotFoundException:
        pass
