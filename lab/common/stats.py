"""Small statistics helpers for reporting results across random seeds."""
import math
from dataclasses import dataclass

from scipy import stats


@dataclass(frozen=True)
class Summary:
    mean: float
    std: float
    ci95: float  # half-width of the 95% confidence interval for the mean
    n: int

    def fmt(self, digits: int = 3) -> str:
        """Format as 'mean ± ci95' in scientific notation for small values."""
        if self.mean != 0 and abs(self.mean) < 1e-2:
            return f"{self.mean:.{digits - 1}e} ± {self.ci95:.1e}"
        return f"{self.mean:.{digits}f} ± {self.ci95:.{digits}f}"


def summarize(values: list[float]) -> Summary:
    """Mean, sample std and a t-distribution 95% CI over seeds."""
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return Summary(mean, 0.0, float("nan"), n)
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    half = stats.t.ppf(0.975, n - 1) * std / math.sqrt(n)
    return Summary(mean, std, half, n)


def welch_p_value(a: list[float], b: list[float]) -> float:
    """Two-sided Welch t-test p-value: is the difference between a and b real?"""
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)
