"""Generates one recognition-rate CSV report per prompt type (stringent,
lenient) from the evaluation_results Athena table.

Fetching from Athena is kept separate from shaping the report rows, so the
shaping/splitting logic (the part with actual bugs to have) is unit-testable
without any AWS calls.
"""

import argparse
import csv
import time
from pathlib import Path

import boto3

from recognition_rate_stats import compute_ci

_DATABASE = "steering_visualization"
_WORKGROUP = "steering-visualization"
_QUERY = """
SELECT layer, category, label, prompt_type, count(*) AS n, sum(cast(evaluation AS integer)) AS successes
FROM evaluation_results
GROUP BY layer, category, label, prompt_type
"""
_REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
_FIELDNAMES = [
    "category", "label", "layer", "n", "successes", "success_rate_pct",
    "wilson_ci95_lower_pct", "wilson_ci95_upper_pct",
    "wald_ci95_lower_pct", "wald_ci95_upper_pct",
]


def fetch_athena_rows(athena_client, query: str, database: str, workgroup: str) -> list[dict]:
    execution_id = athena_client.start_query_execution(
        QueryString=query,
        QueryExecutionContext={"Database": database},
        WorkGroup=workgroup,
    )["QueryExecutionId"]

    while True:
        state = athena_client.get_query_execution(QueryExecutionId=execution_id)["QueryExecution"]["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        time.sleep(1)
    if state != "SUCCEEDED":
        raise RuntimeError(f"Athena query did not succeed: {state}")

    header = None
    rows = []
    paginator = athena_client.get_paginator("get_query_results")
    for page in paginator.paginate(QueryExecutionId=execution_id):
        for row in page["ResultSet"]["Rows"]:
            values = [cell.get("VarCharValue", "") for cell in row["Data"]]
            if header is None:
                header = values
                continue
            rows.append(dict(zip(header, values)))
    return rows


def build_report_rows(athena_rows: list[dict]) -> list[dict]:
    """Turns raw Athena rows (all-string values) into report rows with CI
    columns computed, sorted by category/label/layer. `prompt_type` is kept
    on each row so callers can split by it; it is not one of the output CSV
    columns (see split_by_prompt_type)."""
    report_rows = []
    for row in athena_rows:
        n = int(row["n"])
        successes = int(row["successes"])
        ci = compute_ci(n, successes)
        report_rows.append(
            {
                "category": row["category"],
                "label": row["label"],
                "layer": int(row["layer"]),
                "prompt_type": row["prompt_type"],
                "n": n,
                "successes": successes,
                "success_rate_pct": round(ci.rate * 100, 1),
                "wilson_ci95_lower_pct": round(ci.wilson_lower * 100, 1),
                "wilson_ci95_upper_pct": round(ci.wilson_upper * 100, 1),
                "wald_ci95_lower_pct": round(ci.wald_lower * 100, 1),
                "wald_ci95_upper_pct": round(ci.wald_upper * 100, 1),
            }
        )
    report_rows.sort(key=lambda r: (r["category"], r["label"], r["layer"]))
    return report_rows


def split_by_prompt_type(report_rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in report_rows:
        groups.setdefault(row["prompt_type"], []).append(row)
    return groups


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in _FIELDNAMES})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=None, help="AWS CLI profile to use")
    parser.add_argument("--output-dir", type=Path, default=_REPORTS_DIR)
    args = parser.parse_args()

    session = boto3.Session(profile_name=args.profile) if args.profile else boto3.Session()
    athena_client = session.client("athena")

    athena_rows = fetch_athena_rows(athena_client, _QUERY, _DATABASE, _WORKGROUP)
    report_rows = build_report_rows(athena_rows)
    by_prompt_type = split_by_prompt_type(report_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for prompt_type, rows in sorted(by_prompt_type.items()):
        path = args.output_dir / f"recognition_rate_by_layer_{prompt_type}.csv"
        write_csv(rows, path)
        print(f"wrote {path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
