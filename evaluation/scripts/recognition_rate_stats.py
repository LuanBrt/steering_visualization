"""Wilson and Wald 95% confidence intervals for a binomial proportion.

Shared by the CSV report generator and the chart script so the two never
drift out of sync on the formula.
"""

import math
from dataclasses import dataclass

_Z_95 = 1.959963984540054


@dataclass(frozen=True)
class ProportionCI:
    rate: float
    wilson_lower: float
    wilson_upper: float
    wald_lower: float
    wald_upper: float


def compute_ci(n: int, successes: int) -> ProportionCI:
    phat = successes / n
    z = _Z_95

    denom = 1 + z**2 / n
    center = phat + z**2 / (2 * n)
    adj = z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    wilson_lower = max(0.0, (center - adj) / denom)
    wilson_upper = min(1.0, (center + adj) / denom)

    # Wald (normal approximation) is left unclipped on purpose: it can go
    # below 0% or above 100% at small n / extreme rates, which is exactly the
    # failure mode Wilson is meant to avoid.
    se = math.sqrt(phat * (1 - phat) / n)
    wald_lower = phat - z * se
    wald_upper = phat + z * se

    return ProportionCI(
        rate=phat,
        wilson_lower=wilson_lower, wilson_upper=wilson_upper,
        wald_lower=wald_lower, wald_upper=wald_upper,
    )
