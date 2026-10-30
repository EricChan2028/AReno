"""Render markdown tables for the KIVI report from the raw result files."""

import glob
import json
import os

RUNS = "/root/kivi/runs"
out = []


def load(path):
    with open(path) as handle:
        return json.load(handle)


# 1. Kernel decode benchmark (CUDA graph replay).
bench = f"{RUNS}/bench_final/decode_graph.json"
if not os.path.exists(bench):
    bench = f"{RUNS}/bench/decode_v6_graph.json"
data = load(bench)
rows = {}
for r in data["results"]:
    rows.setdefault((r["context"], r["rows"]), {})[r["dtype"]] = r
out.append(f"### Decode attention, one layer (`{os.path.relpath(bench, RUNS)}`; {data['device']}; torch {data['torch']})\n")
out.append(
    "| context | rows | BF16 ms | KIVI-2 ms (speedup) | KIVI-4 ms (speedup) "
    "| BF16 cache MiB | KIVI-2 cache MiB | KIVI-4 cache MiB | KIVI-2 workspace MiB |"
)
out.append("|---|---|---|---|---|---|---|---|---|")
for (ctx, nrows), d in sorted(rows.items()):
    b, k2, k4 = d["bf16"], d["kivi2"], d["kivi4"]
    out.append(
        f"| {ctx} | {nrows} | {b['median_ms']:.3f} "
        f"| {k2['median_ms']:.3f} ({b['median_ms'] / k2['median_ms']:.2f}x) "
        f"| {k4['median_ms']:.3f} ({b['median_ms'] / k4['median_ms']:.2f}x) "
        f"| {b['cache_bytes'] / 2**20:.1f} | {k2['cache_bytes'] / 2**20:.1f} | {k4['cache_bytes'] / 2**20:.1f} "
        f"| {k2['step_workspace_bytes'] / 2**20:.1f} |"
    )

# 2. GSM8K sampled evaluation at equal concurrency.
out.append("\n### GSM8K test, 1319 prompts x 4 samples, temperature 1.0, seed 1234, 128 concurrent prompts\n")
out.append("| KV cache | group | residual | accuracy (+- stderr) | generated tok/s | mean response tokens |")
out.append("|---|---|---|---|---|---|")
for path in sorted(glob.glob(f"{RUNS}/eval_sampled/*/summary.json")):
    s = load(path)
    rt = s["runtime"]
    out.append(
        f"| {s['kv_cache_dtype']} | {rt.get('kivi_group_size', '-')} | {rt.get('kivi_residual_length', '-')} "
        f"| {100 * s['accuracy']:.2f}% +- {100 * s['accuracy_stderr']:.2f} "
        f"| {s['generated_tokens_per_s']:.0f} | {s['mean_response_tokens']:.0f} |"
    )

# 3. Throughput at higher concurrency.
out.append("\n### Greedy GSM8K rollout, first 1024 test prompts: concurrency vs device memory\n")
out.append("| KV cache | max running prompts | generated tok/s | peak device memory GiB | accuracy |")
out.append("|---|---|---|---|---|")
for path in sorted(glob.glob(f"{RUNS}/throughput/*/summary.json")):
    s = load(path)
    with open(os.path.join(os.path.dirname(path), "peak_mem_mib.txt")) as handle:
        peak = int(handle.read().split()[0])
    out.append(
        f"| {s['kv_cache_dtype']} | {s['max_running_prompts']} | {s['generated_tokens_per_s']:.0f} "
        f"| {peak / 1024:.1f} | {100 * s['accuracy']:.1f}% |"
    )

# 4. Teacher-forced long-context quality.
paths = sorted(glob.glob(f"{RUNS}/teacher_forced/*.json")) + sorted(glob.glob(f"{RUNS}/teacher_forced/ablation/*.json"))
for path in paths:
    r = load(path)
    model = os.path.basename(r["model"])
    out.append(
        f"\n### Teacher-forced WikiText-2 ({model}, {r['rows']} x {r['length']} tokens, "
        f"G={r['kivi_group_size']}, R={r['kivi_residual_length']}; `{os.path.relpath(path, RUNS)}`)\n"
    )
    names = [n for n in next(iter(r["buckets"].values())) if n != "auto"]
    out.append("| positions | BF16 PPL | " + " | ".join(f"{n}: PPL (delta) / KL / top-1" for n in names) + " |")
    out.append("|---|---|" + "---|" * len(names))
    for bucket, d in r["buckets"].items():
        base = d["auto"]["ppl"]
        cells = [
            f"{d[n]['ppl']:.3f} ({100 * (d[n]['ppl'] / base - 1):+.2f}%) / {d[n]['kl_vs_auto']:.4f} "
            f"/ {100 * d[n]['top1_agreement_vs_auto']:.1f}%"
            for n in names
        ]
        out.append(f"| {bucket} | {base:.3f} | " + " | ".join(cells) + " |")
    ratios = ", ".join(f"{k}={v / r['cache_bytes']['auto']:.3f}" for k, v in r["cache_bytes"].items() if k != "auto")
    out.append(f"\ncache bytes relative to BF16 (incl. scales and residual rings): {ratios}")

# 5. RL.
for path in sorted(glob.glob(f"{RUNS}/rl_final/summary_*.json")):
    s = load(path)
    out.append(
        f"\n### GSPO on GSM8K: final-window mean reward (last {s['window']} steps; `{os.path.relpath(path, RUNS)}`)\n"
    )
    out.append("| KV cache | per-seed final-window reward | mean (std) | delta vs BF16 | train/logp_abs_diff_mean |")
    out.append("|---|---|---|---|---|")
    for arm, e in s["arms"].items():
        per_seed = ", ".join(f"{x['final_window_reward']:.4f}" for x in e["seeds"])
        std = e["final_window_reward_std"]
        delta = e.get("final_window_reward_delta_vs_baseline")
        mean = f"{e['final_window_reward_mean']:.4f}" + (f" ({std:.4f})" if std is not None else "")
        delta_text = "-" if delta is None else f"{delta:+.4f}"
        out.append(f"| {arm} | {per_seed} | {mean} | {delta_text} | {e['logp_abs_diff_mean']:.4f} |")

os.makedirs("/root/kivi/report", exist_ok=True)
with open("/root/kivi/report/tables.md", "w") as handle:
    handle.write("\n".join(out) + "\n")
print("\n".join(out))
