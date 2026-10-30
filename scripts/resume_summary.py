"""Compare the resumed seed-1 runs (steps 51-100) with their uninterrupted runs."""

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "/root/kivi/wt/final_f333cdc/examples/kivi")
from summarize_rl import LOGP, REWARD, _scalars  # noqa: E402

RUNS = Path("/root/kivi/runs/rl_final")


def part(scalars, start, stop):
    rewards = [value for _, value in sorted(scalars[REWARD])][start:stop]
    logp = [value for _, value in sorted(scalars.get(LOGP, []))][start:stop]
    return {
        "steps": len(rewards),
        "mean_reward": statistics.fmean(rewards),
        "final_window_reward": statistics.fmean(rewards[-20:]),
        "logp_abs_diff_mean": statistics.fmean(logp),
        "reward_curve": rewards,
        "logp_curve": logp,
    }


report = {}
for arm in ("auto", "kivi2"):
    report[arm] = {
        "uninterrupted": part(_scalars(RUNS / f"s1_{arm}"), 50, 100),
        "resumed": part(_scalars(RUNS / f"s1_{arm}_resume50"), 0, 50),
    }
(RUNS / "summary_resume.json").write_text(json.dumps(report, indent=2) + "\n")
for arm, entry in report.items():
    for key, x in entry.items():
        print(arm, key, x["steps"], round(x["mean_reward"], 4), round(x["final_window_reward"], 4), round(x["logp_abs_diff_mean"], 4))
