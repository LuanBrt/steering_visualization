import pytest

from recognition_rate_stats import compute_ci


def test_compute_ci_midpoint_rate():
    ci = compute_ci(n=20, successes=10)
    assert ci.rate == pytest.approx(0.5)


def test_wilson_never_collapses_to_a_point_at_zero_successes():
    ci = compute_ci(n=60, successes=0)
    assert ci.rate == 0.0
    assert ci.wilson_lower == 0.0
    assert ci.wilson_upper > 0.0  # Wilson still allows for a nonzero true rate


def test_wilson_never_collapses_to_a_point_at_all_successes():
    ci = compute_ci(n=60, successes=60)
    assert ci.rate == 1.0
    assert ci.wilson_upper == pytest.approx(1.0)
    assert ci.wilson_lower < 1.0


def test_wilson_stays_within_zero_and_one():
    for successes in range(0, 21):
        ci = compute_ci(n=20, successes=successes)
        assert 0.0 <= ci.wilson_lower <= ci.wilson_upper <= 1.0


def test_wald_collapses_to_a_single_point_at_zero_successes():
    # This is exactly the failure mode Wilson was chosen to avoid.
    ci = compute_ci(n=20, successes=0)
    assert ci.wald_lower == ci.wald_upper == 0.0


def test_wald_can_go_negative_at_small_n_and_extreme_rate():
    ci = compute_ci(n=20, successes=3)
    assert ci.wald_lower < 0.0
