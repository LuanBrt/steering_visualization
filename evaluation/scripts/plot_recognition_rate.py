"""Small-multiples plot of recognition rate vs. layer, one panel per concept,
styled after Figure 3 of the source paper (2601.08017v1): a red line with
markers and a shaded confidence band, faceted by concept.

Reads evaluation/reports/recognition_rate_by_layer.csv (produced from the
Athena evaluation_results table) and writes a PNG next to it.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

_LINE_COLOR = "#e34948"
_BAND_ALPHA = 0.22
_DEFAULT_CSV = Path(__file__).resolve().parent.parent / "reports" / "recognition_rate_by_layer.csv"
_DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "reports" / "recognition_rate_by_layer.png"


def load_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def group_by_concept(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (row["category"], row["label"])
        groups.setdefault(key, []).append(row)
    for concept_rows in groups.values():
        concept_rows.sort(key=lambda r: int(r["layer"]))
    return groups


def plot(rows: list[dict], ci_method: str, output_path: Path, title: str | None = None) -> None:
    groups = group_by_concept(rows)
    concepts = sorted(groups.keys())

    n_cols = 3
    n_rows = -(-len(concepts) // n_cols)  # ceil division
    # sharex=False on purpose: different concepts can have been tested at
    # different layers (e.g. an older batch at 8/16/24 vs a newer one at
    # 5/15/25). Sharing the x-axis would share tick positions/labels too --
    # matplotlib applies whichever panel's set_xticks() ran last to the
    # whole shared group -- silently mislabeling every other panel's axis.
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.1 * n_cols, 2.6 * n_rows), sharex=False, sharey=True,
    )
    axes = axes.flatten()

    lower_col = f"{ci_method}_ci95_lower_pct"
    upper_col = f"{ci_method}_ci95_upper_pct"

    for ax, (category, label) in zip(axes, concepts):
        concept_rows = groups[(category, label)]
        layers = [int(r["layer"]) for r in concept_rows]
        rates = [float(r["success_rate_pct"]) / 100 for r in concept_rows]
        lower = [float(r[lower_col]) / 100 for r in concept_rows]
        upper = [float(r[upper_col]) / 100 for r in concept_rows]

        ax.fill_between(layers, lower, upper, color=_LINE_COLOR, alpha=_BAND_ALPHA, linewidth=0)
        ax.plot(layers, rates, color=_LINE_COLOR, linewidth=2, marker="o", markersize=6, zorder=3)

        ax.set_title(f"{category} — {label}", fontsize=10, color="#2b2b2b")
        ax.set_ylim(0, 1)
        ax.set_xticks(layers)
        ax.set_yticks([0, 0.5, 1])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#c9c9c9")
        ax.spines["bottom"].set_color("#c9c9c9")
        ax.tick_params(colors="#6b6b6b", labelsize=9)
        ax.grid(False)

    for ax in axes[len(concepts):]:
        ax.axis("off")

    fig.supxlabel("Layer", fontsize=11, color="#2b2b2b")
    fig.supylabel("Proportion recognised", fontsize=11, color="#2b2b2b")
    fig.suptitle(
        title or f"Recognition rate by layer, per concept ({ci_method.capitalize()} 95% CI)",
        fontsize=12, color="#2b2b2b",
    )
    fig.tight_layout(rect=(0.03, 0.02, 1, 0.94))
    fig.savefig(output_path, dpi=200, facecolor="white")
    print(f"wrote {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=_DEFAULT_CSV)
    parser.add_argument("--ci", choices=["wald", "wilson"], default="wald")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    rows = load_rows(args.csv)
    plot(rows, args.ci, args.output, title=args.title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
