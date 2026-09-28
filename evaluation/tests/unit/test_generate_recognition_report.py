from generate_recognition_report import build_report_rows, split_by_prompt_type


def _athena_row(layer, category, label, prompt_type, n, successes):
    return {
        "layer": layer, "category": category, "label": label,
        "prompt_type": prompt_type, "n": str(n), "successes": str(successes),
    }


def test_build_report_rows_converts_types_and_computes_ci():
    athena_rows = [_athena_row("8", "Animals", "giraffe", "stringent", 10, 10)]

    rows = build_report_rows(athena_rows)

    assert len(rows) == 1
    row = rows[0]
    assert row["layer"] == 8  # int, not "8"
    assert row["n"] == 10
    assert row["successes"] == 10
    assert row["success_rate_pct"] == 100.0
    assert row["wilson_ci95_upper_pct"] == 100.0
    assert row["prompt_type"] == "stringent"


def test_build_report_rows_sorts_by_category_label_layer():
    athena_rows = [
        _athena_row("24", "Animals", "giraffe", "stringent", 10, 5),
        _athena_row("8", "Animals", "giraffe", "stringent", 10, 5),
        _athena_row("8", "Animals", "octopus", "stringent", 10, 5),
    ]

    rows = build_report_rows(athena_rows)

    assert [(r["label"], r["layer"]) for r in rows] == [
        ("giraffe", 8), ("giraffe", 24), ("octopus", 8),
    ]


def test_split_by_prompt_type_separates_stringent_and_lenient():
    athena_rows = [
        _athena_row("8", "Animals", "giraffe", "stringent", 10, 10),
        _athena_row("8", "Animals", "giraffe", "lenient", 10, 8),
    ]
    rows = build_report_rows(athena_rows)

    groups = split_by_prompt_type(rows)

    assert set(groups.keys()) == {"stringent", "lenient"}
    assert groups["stringent"][0]["successes"] == 10
    assert groups["lenient"][0]["successes"] == 8
