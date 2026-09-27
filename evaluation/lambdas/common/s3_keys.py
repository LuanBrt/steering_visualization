from dataclasses import dataclass


@dataclass(frozen=True)
class DataKey:
    layer: str
    category: str
    label: str
    variant: str
    ext: str


@dataclass(frozen=True)
class ResultKey:
    layer: str
    category: str
    label: str
    variant: str
    prompt_type: str
    kind: str
    round: int


def parse_data_key(key: str) -> DataKey:
    parts = key.split("/")
    if len(parts) != 5 or parts[0] != "data" or not parts[1].startswith("layer_"):
        raise ValueError(f"unexpected data key format: {key!r}")
    _, layer_segment, category, label, filename = parts
    layer = layer_segment.removeprefix("layer_")
    variant, sep, ext = filename.rpartition(".")
    if not sep:
        raise ValueError(f"data key filename has no extension: {key!r}")
    return DataKey(layer=layer, category=category, label=label, variant=variant, ext=ext)


def parse_result_key(key: str) -> ResultKey:
    parts = key.split("/")
    if len(parts) != 8 or parts[0] != "results" or not parts[1].startswith("layer_"):
        raise ValueError(f"unexpected result key format: {key!r}")
    _, layer_segment, category, label, variant, prompt_type, kind, filename = parts
    layer = layer_segment.removeprefix("layer_")
    name, sep, _ext = filename.rpartition(".")
    if not sep:
        raise ValueError(f"result key filename has no extension: {key!r}")
    _, _, round_str = name.rpartition("_")
    return ResultKey(
        layer=layer, category=category, label=label, variant=variant,
        prompt_type=prompt_type, kind=kind, round=int(round_str),
    )


def _result_key(layer: str, category: str, label: str, variant: str, prompt_type: str, kind: str, filename: str) -> str:
    return f"results/layer_{layer}/{category}/{label}/{variant}/{prompt_type}/{kind}/{filename}"


def recognition_key(layer: str, category: str, label: str, variant: str, prompt_type: str, round: int) -> str:
    return _result_key(layer, category, label, variant, prompt_type, "recognition", f"recognition_{round}.txt")


def eval_key(layer: str, category: str, label: str, variant: str, prompt_type: str, round: int) -> str:
    return _result_key(layer, category, label, variant, prompt_type, "eval", f"eval_{round}.txt")


def recognition_partition_location(bucket: str, layer: str, category: str, label: str, variant: str, prompt_type: str) -> str:
    return f"s3://{bucket}/results/layer_{layer}/{category}/{label}/{variant}/{prompt_type}/recognition/"


def eval_partition_location(bucket: str, layer: str, category: str, label: str, variant: str, prompt_type: str) -> str:
    return f"s3://{bucket}/results/layer_{layer}/{category}/{label}/{variant}/{prompt_type}/eval/"


def results_delete_prefix(layer: str, category: str, label: str, variant: str) -> str:
    return f"results/layer_{layer}/{category}/{label}/{variant}/"


def display_name(value: str) -> str:
    """Convert a path segment (snake_case, e.g. "Marilyn_Monroe") into
    natural-language text for prompts (e.g. "Marilyn Monroe"). S3 keys and
    Glue partition values keep the original underscored form -- only text
    interpolated into a prompt sentence goes through this."""
    return value.replace("_", " ")
