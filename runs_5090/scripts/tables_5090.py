"""Markdown tables for the KIVI #590 re-run on the laptop RTX 5090.

Reads the new results under NEW and, for like-for-like comparisons, the
report's raw logs (RTX PRO 5000, flash backend, main e17794c) under OLD.
"""

import glob
import json
import lzma
import os
import re
import statistics
import sys

NEW = sys.argv[1] if len(sys.argv) > 1 else "/root/kivi590/runs_5090"
OLD = sys.argv[2] if len(sys.argv) > 2 else "../runs"
NAMES = {"auto": "BF16", "bf16": "BF16", "kivi4": "KIVI-4", "kivi2": "KIVI-2", "kivi2_r512": "KIVI-2 R=512"}


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def read(path, default=""):
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return default


def peak_gib(name):
    value = read(f"{NEW}/{name}/peak_mem_mib.txt")
    return f"{int(value) / 1024:.1f} GiB" if value else "-"


def kivi_bytes(name):
    match = re.search(r"KIVI KV cache: .*?bytes=(\d+)", read(f"{NEW}/{name}/run.log"))
    return int(match.group(1)) if match else None


def section(title):
    print(f"\n#### {title}\n")


print("Environment:", read(f"{NEW}/env/versions.txt").replace("\n", "; "))
print("Commit:", read(f"{NEW}/env/commit.txt").splitlines()[0] if read(f"{NEW}/env/commit.txt") else "-")
print("Extension:", read(f"{NEW}/extension_source.txt").replace("\n", "; "))

section("Tests")
for name in ("tests_kivi", "tests_upstream_gpu", "tests_upstream_gpu_pr", "tests_upstream_gpu_main"):
    lines = [line for line in read(f"{NEW}/{name}/run.log").splitlines() if re.search(r"\d+ (passed|failed)", line)]
    print(f"- {name}: {lines[-1] if lines else 'no result line'}")

section("Decode attention step, one layer, BF16 native backend vs KIVI (median of 20 after 2 warmup)")
rows = []
for path in ("bench_graph/decode_graph.json", "bench_graph_16k/decode_graph_16k.json",
             "bench_graph_128x16k_kivi/decode_graph_128x16k_kivi.json", "bench_graph_32k/decode_graph_32k.json",
             "bench_eager/decode_eager.json"):
    full = f"{NEW}/{path}"
    if not os.path.exists(full):
        continue
    data = load(full)
    mode = "graph" if data["args"].get("cuda_graph") else "eager"
    by_key = {}
    for row in data["results"]:
        by_key.setdefault((row["rows"], row["context"]), {})[row["dtype"]] = row
    for (n, context), entry in sorted(by_key.items(), key=lambda item: (item[0][1], item[0][0])):
        bf16 = entry.get("bf16")
        cells = [mode, str(n), str(context)]
        for dtype in ("bf16", "kivi4", "kivi2"):
            row = entry.get(dtype)
            if row is None:
                cells.append("-")
            elif dtype == "bf16":
                cells.append(f"{row['median_ms']:.3f}")
            elif bf16 is None:
                cells.append(f"{row['median_ms']:.3f}")
            else:
                cells.append(f"{row['median_ms']:.3f} ({bf16['median_ms'] / row['median_ms']:.2f}x)")
        for dtype in ("kivi4", "kivi2"):
            row = entry.get(dtype)
            cells.append(f"{row['total_bytes'] / bf16['total_bytes']:.3f}" if row and bf16 else "-")
        rows.append(cells)
print("| mode | rows | cached tokens | BF16 native ms | KIVI-4 ms (speedup) | KIVI-2 ms (speedup) | KIVI-4 / BF16 bytes | KIVI-2 / BF16 bytes |")
print("|---|---|---|---|---|---|---|---|")
for cells in rows:
    print("| " + " | ".join(cells) + " |")

section("Teacher-forced WikiText-2, Qwen3-0.6B, 4 x 16384 tokens, G = 32, R = 128: perplexity (change vs BF16)")
new_tf = f"{NEW}/tf_len16384/len16384_rows4.json"
old_tf = f"{OLD}/teacher_forced/len16384_rows4.json"
if os.path.exists(new_tf):
    new = load(new_tf)
    old = load(old_tf) if os.path.exists(old_tf) else None
    print("| positions | BF16 | KIVI-4 | KIVI-2 | report (RTX PRO 5000): BF16 / KIVI-4 / KIVI-2 |")
    print("|---|---|---|---|---|")
    for bucket, entry in new["buckets"].items():
        base = entry["auto"]["ppl"]
        cells = [bucket, f"{base:.3f}"]
        for dtype in ("kivi4", "kivi2"):
            ppl = entry[dtype]["ppl"]
            cells.append(f"{ppl:.3f} ({(ppl / base - 1) * 100:+.2f}%)")
        if old and bucket in old["buckets"]:
            o = old["buckets"][bucket]
            cells.append(" / ".join(f"{o[d]['ppl']:.3f}" for d in ("auto", "kivi4", "kivi2")))
        else:
            cells.append("-")
        print("| " + " | ".join(cells) + " |")
    print(f"\ncache bytes: {new['cache_bytes']}, decode seconds {new['decode_seconds']:.0f}")


def scalars(tb_dir, tags):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(tb_dir, size_guidance={"scalars": 0})
    accumulator.Reload()
    present = set(accumulator.Tags()["scalars"])
    return {tag: [(event.step, event.value) for event in accumulator.Scalars(tag)] for tag in tags if tag in present}


TAGS = ("rollout/rewards_mean", "train/logp_abs_diff_mean", "train/response_len", "rollout/response_len_mean")


def rl_stats(tb_dir, first=None, last=None):
    if not os.path.isdir(tb_dir):
        return None
    data = scalars(tb_dir, TAGS)
    out = {}
    for tag, points in data.items():
        values = [value for step, value in sorted(points)]
        if first is not None:
            values = values[:first]
        if last is not None:
            values = values[-last:]
        out[tag] = (statistics.fmean(values), len(values)) if values else None
    return out


section("GSPO with areno train (seed 1): trainer-native train/logp_abs_diff_mean and reward")
arms = [("auto", "rl_s1_auto", "s1_auto"), ("kivi4", "rl_s1_kivi4", "s1_kivi4"), ("kivi2", "rl_s1_kivi2", "s1_kivi2"),
        ("kivi2_r512", "rl_s1_kivi2_r512", "s1_kivi2_r512")]
print("| KV cache | steps | logp_abs_diff_mean | reward | report, same first steps: logp_abs_diff_mean / reward | report, 100 steps: logp_abs_diff_mean |")
print("|---|---|---|---|---|---|")
for arm, new_dir, old_dir in arms:
    new = rl_stats(f"{NEW}/{new_dir}/tb")
    if not new:
        continue
    steps = new["train/logp_abs_diff_mean"][1]
    old_first = rl_stats(f"{OLD}/rl_final/{old_dir}/tb", first=steps)
    old_all = rl_stats(f"{OLD}/rl_final/{old_dir}/tb")
    cells = [NAMES[arm], str(steps), f"{new['train/logp_abs_diff_mean'][0]:.4f}", f"{new['rollout/rewards_mean'][0]:.3f}"]
    cells.append(
        f"{old_first['train/logp_abs_diff_mean'][0]:.4f} / {old_first['rollout/rewards_mean'][0]:.3f}" if old_first else "-"
    )
    cells.append(f"{old_all['train/logp_abs_diff_mean'][0]:.4f}" if old_all else "-")
    print("| " + " | ".join(cells) + " |")

resume_dirs = sorted(glob.glob(f"{NEW}/rl_s1_kivi4_resume*"))
if resume_dirs:
    resumed_dir = resume_dirs[0]
    half = int(re.search(r"resume(\d+)", resumed_dir).group(1))
    section(f"Save at step {half} and resume (KIVI-4, seed 1): steps after the checkpoint")
    full = scalars(f"{NEW}/rl_s1_kivi4/tb", TAGS)
    resumed = scalars(f"{resumed_dir}/tb", TAGS)
    print("| step | uninterrupted reward / logp_abs_diff_mean | resumed reward / logp_abs_diff_mean |")
    print("|---|---|---|")
    full_reward = [v for _, v in sorted(full.get("rollout/rewards_mean", []))]
    full_logp = [v for _, v in sorted(full.get("train/logp_abs_diff_mean", []))]
    res_reward = [v for _, v in sorted(resumed.get("rollout/rewards_mean", []))]
    res_logp = [v for _, v in sorted(resumed.get("train/logp_abs_diff_mean", []))]
    for index in range(len(res_reward)):
        step = half + index + 1
        a = f"{full_reward[step - 1]:.3f} / {full_logp[step - 1]:.4f}" if step - 1 < len(full_reward) else "-"
        b = f"{res_reward[index]:.3f} / {res_logp[index]:.4f}" if index < len(res_logp) else "-"
        print(f"| {step} | {a} | {b} |")


def old_rollout_accuracy(path, limit, sample=None):
    rewards = []
    opener = lzma.open if path.endswith(".xz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["index"] < limit and (sample is None or row["sample"] == sample):
                rewards.append(row["reward"])
    return statistics.fmean(rewards) if rewards else None


section("GSM8K after ~8K-token passages, first 64 problems of the report's set, greedy, max 2048 new tokens")
print("| KV cache | concurrent | accuracy | mean response tokens | tokens/s | peak device memory | KIVI cache bytes | report, same 64 problems at 32 concurrent |")
print("|---|---|---|---|---|---|---|---|")
for name in sorted(glob.glob(f"{NEW}/lc64_*")):
    base = os.path.basename(name)
    if not os.path.exists(f"{name}/summary.json"):
        print(f"| {base} | - | failed or stopped: {read(f'{name}/STOPPED', 'see run.log')} | | | | | |")
        continue
    summary = load(f"{name}/summary.json")
    dtype = summary["kv_cache_dtype"]
    old_name = {"auto": "auto_mrp32", "kivi4": "kivi4_mrp32", "kivi2": "kivi2_mrp32"}[dtype]
    old_path = f"{OLD}/long_context/{old_name}/rollouts.jsonl.xz"
    old_acc = old_rollout_accuracy(old_path, 64) if os.path.exists(old_path) else None
    size = kivi_bytes(base)
    print(
        f"| {NAMES[dtype]} | {summary['max_running_prompts']} | {summary['accuracy']:.3f} | "
        f"{summary['mean_response_tokens']:.0f} | {summary['generated_tokens_per_s']:.0f} | {peak_gib(base)} | "
        f"{f'{size / 2**30:.2f} GiB' if size else '-'} | {f'{old_acc:.3f}' if old_acc is not None else '-'} |"
    )

section("GSM8K test, first 512 problems, one sample at T = 1, seed 1234, 32 concurrent")
print("| KV cache | accuracy +- SE | mean response tokens | tokens/s | peak device memory | report, sample 0 of the same 512 problems |")
print("|---|---|---|---|---|---|")
for dtype in ("auto", "kivi4", "kivi2"):
    name = f"gsm8k512_{dtype}"
    if not os.path.exists(f"{NEW}/{name}/summary.json"):
        continue
    summary = load(f"{NEW}/{name}/summary.json")
    old_path = f"{OLD}/eval_sampled/{dtype}/completions.jsonl"
    old_acc = old_rollout_accuracy(old_path, 512, sample=0) if os.path.exists(old_path) else None
    print(
        f"| {NAMES[dtype]} | {summary['accuracy']:.3f} +- {summary['accuracy_stderr']:.3f} | "
        f"{summary['mean_response_tokens']:.0f} | {summary['generated_tokens_per_s']:.0f} | {peak_gib(name)} | "
        f"{f'{old_acc:.3f}' if old_acc is not None else '-'} |"
    )

section("Run log")
print(read(f"{NEW}/g.log"))
