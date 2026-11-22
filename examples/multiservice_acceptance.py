"""Fixed model-performance gate, separate from deterministic software tests.

A development pass is not confirmation. Confirmation requires two independent
training seeds, 32 matched worlds per task, and the reserved task suite below.
Do not retune a model on confirmation results and reuse them as fresh evidence.
"""

import argparse
import json
from pathlib import Path
import numpy as np
from examples.multiservice_adaptation import Task

TASKS = {
    "four_stable": Task(4, 1.0, 0.0, 0, 128),
    "eight_stable": Task(8, 1.0, 0.0, 0, 128),
    "four_changes": Task(4, 1.0, 0.0, 22, 128),
    "eight_changes": Task(8, 1.0, 0.0, 34, 128),
    "four_noisy": Task(4, 0.82, 0.18, 22, 128),
    "eight_noisy": Task(8, 0.82, 0.18, 34, 128),
    "eight_harder": Task(8, 0.68, 0.32, 34, 128),
}
BASELINES = ("last_outcome", "window_mean", "window_ucb")
MAX_GAP = 0.05
STABLE_FLOORS = {"four_stable": 0.85, "eight_stable": 0.80}


def assess(results, *, minimum_trials=32):
    checks = {}
    for name in TASKS:
        task = results[name]
        learned = task["language"]
        baseline = max(BASELINES, key=lambda k: task[k]["mean_reward"])
        a = np.asarray(learned["per_trial_reward"], dtype=float)
        b = np.asarray(task[baseline]["per_trial_reward"], dtype=float)
        if a.shape != b.shape or a.ndim != 1 or not len(a):
            raise ValueError("matched per-world rewards required")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("non-finite evaluation rewards")
        for k in ("language", *BASELINES):
            row = task[k]
            values = np.asarray(row["per_trial_reward"], dtype=float)
            if values.shape != a.shape or row["trial_count"] != len(a):
                raise ValueError("all methods must use the same number of worlds")
            if not np.isfinite(values).all() or not np.isclose(
                values.mean(), row["mean_reward"]
            ):
                raise ValueError("inconsistent evaluation summary")
        delta = a - b
        floor = STABLE_FLOORS.get(name, 0.0)
        checks[name] = dict(
            passed=bool(
                len(a) >= minimum_trials
                and delta.mean() >= -MAX_GAP - 1e-12
                and a.mean() >= floor - 1e-12
            ),
            baseline=baseline,
            reward=float(a.mean()),
            gap=float(delta.mean()),
            paired_standard_error=(
                float(delta.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else None
            ),
            stable_floor=floor,
            trial_count=len(a),
        )
    return {
        "passed": all(c["passed"] for c in checks.values()),
        "tasks": checks,
        "maximum_baseline_gap": MAX_GAP,
        "minimum_trials": minimum_trials,
        "note": "Per-run numerical gate only; independent seeds and untouched task provenance must also be checked.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    args = p.parse_args()
    report = assess(json.loads(args.results.read_text()))
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
