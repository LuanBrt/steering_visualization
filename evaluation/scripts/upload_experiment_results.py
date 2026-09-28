"""Uploads `best.png` checkpoints from local VQGAN/Gemma steering experiment
output into the S3 layout expected by the evaluation pipeline.

Source layout (produced by the experiment script at the repo root):
    <source>/<category>/<label>/layer-<N>/{best.png, final.png, ema.png,
                                            step_NNNN.png, metrics.csv, latents.pt}

Target layout (see docs/superpowers/specs/2026-09-26-evaluation-pipeline-design.md):
    data/layer_<N>/<category>/<label>/0.png

Only `best.png` is uploaded, always as variant "0" -- no other checkpoint,
intermediate step, or training artifact (metrics.csv, latents.pt) is part of
this pipeline. Already-uploaded images are never re-uploaded (idempotent by
S3 key existence), so this script can be re-run any time new experiment
results land locally; it only picks up what is new.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

_CHECKPOINT_FILENAME = "best.png"
_VARIANT = "0"
_DEFAULT_SOURCE = Path(__file__).resolve().parents[2] / "results" / "word_experiment_vqgan_gemma"


@dataclass(frozen=True)
class Checkpoint:
    layer: str
    category: str
    label: str
    local_path: Path
    s3_key: str


def discover_checkpoints(source_root: Path) -> list[Checkpoint]:
    """Walk <source_root>/<category>/<label>/layer-<N>/ and return one
    Checkpoint per run that already has a best.png. A run with no best.png
    yet (still training, or crashed before finishing) is silently skipped --
    it will show up on a later run of this script once it completes."""
    checkpoints = []
    if not source_root.is_dir():
        return checkpoints

    for category_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
        for label_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
            for layer_dir in sorted(p for p in label_dir.iterdir() if p.is_dir()):
                if not layer_dir.name.startswith("layer-"):
                    continue
                best_png = layer_dir / _CHECKPOINT_FILENAME
                if not best_png.is_file():
                    continue
                layer = layer_dir.name.removeprefix("layer-")
                s3_key = f"data/layer_{layer}/{category_dir.name}/{label_dir.name}/{_VARIANT}.png"
                checkpoints.append(
                    Checkpoint(
                        layer=layer, category=category_dir.name, label=label_dir.name,
                        local_path=best_png, s3_key=s3_key,
                    )
                )
    return checkpoints


def _object_exists(s3_client, bucket: str, key: str) -> bool:
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        if exc.response["ResponseMetadata"]["HTTPStatusCode"] == 404:
            return False
        raise


def upload_missing(
    s3_client, bucket: str, checkpoints: list[Checkpoint], dry_run: bool = False
) -> tuple[list[Checkpoint], list[Checkpoint]]:
    """Uploads every checkpoint whose S3 key does not exist yet.

    Returns (uploaded, already_present). In dry-run mode, existence is still
    checked (so the preview is accurate) but no file is actually uploaded.
    """
    uploaded = []
    already_present = []
    for checkpoint in checkpoints:
        if _object_exists(s3_client, bucket, checkpoint.s3_key):
            already_present.append(checkpoint)
            continue
        if not dry_run:
            s3_client.upload_file(str(checkpoint.local_path), bucket, checkpoint.s3_key)
        uploaded.append(checkpoint)
    return uploaded, already_present


def discover_all_checkpoints(source_roots: list[Path]) -> list[Checkpoint]:
    """Runs discover_checkpoints over every source root and concatenates the
    results. Different colleagues/batches typically land in differently
    named folders (e.g. a fresh "Resultados-<timestamp>/" export each time),
    so this is how multiple such folders get combined into one upload pass.
    A checkpoint's S3 key does not depend on which source root it came from
    (only on its category/label/layer), so two sources describing the same
    run just resolve to the same key -- upload_missing() already treats a
    duplicate key as "already present" the second time it sees it, so no
    special de-duplication is needed here."""
    all_checkpoints: list[Checkpoint] = []
    for source_root in source_roots:
        all_checkpoints.extend(discover_checkpoints(source_root))
    return all_checkpoints


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source", type=Path, nargs="+", default=[_DEFAULT_SOURCE],
        help=f"One or more roots of local experiment output (default: {_DEFAULT_SOURCE})",
    )
    parser.add_argument("--bucket", default="steering-visualization")
    parser.add_argument("--profile", default=None, help="AWS CLI profile to use")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded without uploading")
    args = parser.parse_args()

    checkpoints = discover_all_checkpoints(args.source)
    if not checkpoints:
        print(f"No {_CHECKPOINT_FILENAME} checkpoints found under {args.source}")
        return 0

    session = boto3.Session(profile_name=args.profile) if args.profile else boto3.Session()
    s3_client = session.client("s3")

    uploaded, already_present = upload_missing(s3_client, args.bucket, checkpoints, dry_run=args.dry_run)

    verb = "Would upload" if args.dry_run else "Uploaded"
    for checkpoint in uploaded:
        print(f"{verb}: {checkpoint.local_path} -> s3://{args.bucket}/{checkpoint.s3_key}")

    action = "to upload" if args.dry_run else "uploaded"
    print(f"\n{len(uploaded)} {action}, {len(already_present)} already present, {len(checkpoints)} runs found total.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
