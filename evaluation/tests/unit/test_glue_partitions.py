from unittest.mock import MagicMock

from common.glue_partitions import delete_partition, ensure_partition
from common.glue_schema import INPUT_FORMAT, OUTPUT_FORMAT, RECOGNITION_COLUMNS, SERIALIZATION_LIBRARY


class _AlreadyExists(Exception):
    pass


class _EntityNotFound(Exception):
    pass


def _client_with_exceptions():
    client = MagicMock()
    client.exceptions.AlreadyExistsException = _AlreadyExists
    client.exceptions.EntityNotFoundException = _EntityNotFound
    return client


def test_ensure_partition_creates_with_expected_storage_descriptor():
    client = _client_with_exceptions()

    ensure_partition(
        client,
        database="steering_visualization",
        table="recognition_results",
        columns=RECOGNITION_COLUMNS,
        values=["0", "Animals", "Giraffe", "stringent"],
        location="s3://steering-visualization/results/layer_0/Animals/Giraffe/stringent/recognition/",
    )

    client.create_partition.assert_called_once()
    call_kwargs = client.create_partition.call_args.kwargs
    assert call_kwargs["DatabaseName"] == "steering_visualization"
    assert call_kwargs["TableName"] == "recognition_results"
    partition_input = call_kwargs["PartitionInput"]
    assert partition_input["Values"] == ["0", "Animals", "Giraffe", "stringent"]
    storage_descriptor = partition_input["StorageDescriptor"]
    assert storage_descriptor["Columns"] == RECOGNITION_COLUMNS
    assert storage_descriptor["Location"] == "s3://steering-visualization/results/layer_0/Animals/Giraffe/stringent/recognition/"
    assert storage_descriptor["InputFormat"] == INPUT_FORMAT
    assert storage_descriptor["OutputFormat"] == OUTPUT_FORMAT
    assert storage_descriptor["SerdeInfo"] == {"SerializationLibrary": SERIALIZATION_LIBRARY}


def test_ensure_partition_swallows_already_exists():
    client = _client_with_exceptions()
    client.create_partition.side_effect = _AlreadyExists()

    ensure_partition(
        client, database="db", table="t", columns=RECOGNITION_COLUMNS,
        values=["0", "Animals", "Giraffe", "stringent"], location="s3://bucket/prefix/",
    )  # must not raise


def test_delete_partition_calls_with_expected_values():
    client = _client_with_exceptions()

    delete_partition(client, database="steering_visualization", table="evaluation_results", values=["0", "Animals", "Giraffe", "lenient"])

    client.delete_partition.assert_called_once_with(
        DatabaseName="steering_visualization",
        TableName="evaluation_results",
        PartitionValues=["0", "Animals", "Giraffe", "lenient"],
    )


def test_delete_partition_swallows_entity_not_found():
    client = _client_with_exceptions()
    client.delete_partition.side_effect = _EntityNotFound()

    delete_partition(client, database="db", table="t", values=["0", "Animals", "Giraffe", "lenient"])  # must not raise
