# stats_utils.py — bootstrap CIs over items.
import random

def bootstrap_ci(questions, metric_fn, n_boot=2000, alpha=0.05, seed=42):
    """
    metric_fn: list[question] -> float | None
    Resamples ITEMS (the independent unit); model answers within an item
    are correlated and must move together.
    Returns (point, lo, hi).
    """
    rng = random.Random(seed)
    point, n, samples = metric_fn(questions), len(questions), []
    for _ in range(n_boot):
        r = [questions[rng.randrange(n)] for _ in range(n)]
        v = metric_fn(r)
        if v is not None:
            samples.append(v)
    if not samples:
        return point, None, None
    samples.sort()
    return (point,
            samples[int(alpha / 2 * len(samples))],
            samples[int((1 - alpha / 2) * len(samples)) - 1])


def paired_bootstrap_delta(questions, metric_a, metric_b, n_boot=2000,
                           alpha=0.05, seed=42):
    """CI on (a − b) evaluated on the SAME resampled items. Use for
    layer-vs-layer and aggregator-vs-majority-vote comparisons."""
    rng = random.Random(seed)
    n, deltas = len(questions), []
    for _ in range(n_boot):
        r = [questions[rng.randrange(n)] for _ in range(n)]
        a, b = metric_a(r), metric_b(r)
        if a is not None and b is not None:
            deltas.append(a - b)
    if not deltas:
        return None, None, None
    deltas.sort()
    point = (metric_a(questions) or 0) - (metric_b(questions) or 0)
    return (point,
            deltas[int(alpha / 2 * len(deltas))],
            deltas[int((1 - alpha / 2) * len(deltas)) - 1])