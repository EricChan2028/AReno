"""Render markdown tables for the KIVI report from the raw result files."""

import glob
import json
import math
import os
import re
import statistics

RUNS = "/root/kivi/runs"
NAMES = {
    "auto": "BF16",
    "bf16": "BF16",
    "bf16repeat": "BF16 repeat",
    "kivi2": "KIVI-2",
    "kivi4": "KIVI-4",
    "kivi2_r512": "KIVI-2, R = 512",
}
out = []


def load(path):
    with open(path) as handle:
        return json.load(handle)


def mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else None)


def fmt_ms(value, std, digits=4):
    if value is None:
        return "-"
    return f"{value:.{digits}f}" + (f" +- {std:.{digits}f}" if std is not None else "")


def bucket_label(bucket, prefill):
    edges = [0, 1024, 2048, 4096, 8192, 16384, 32768]
    if bucket.startswith(">="):
        return f"[{bucket[2:]}, end)"
    edge = int(bucket[1:])
    lower = max(edges[edges.index(edge) - 1], prefill + 1)
    return f"[{lower}, {edge})"


def tf_nll(report, name):
    tokens = sum(d[name]["tokens"] for d in report["buckets"].values())
    nll = sum(math.log(d[name]["ppl"]) * d[name]["tokens"] for d in report["buckets"].values())
    return nll / tokens


def peak_gib(run_dir):
    path = os.path.join(run_dir, "peak_mem_mib.txt")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        text = handle.read().split()
    return int(text[0]) / 1024 if text else None


# 1. Decode attention kernel.
for name, title in (("decode_graph", "CUDA graph replay"), ("decode_eager", "eager launches")):
    bench = f"{RUNS}/bench_final/{name}.json"
    if not os.path.exists(bench):
        continue
    data = load(bench)
    rows = {}
    for r in data["results"]:
        rows.setdefault((r["context"], r["rows"]), {})[r["dtype"]] = r
    args = data["args"]
    out.append(
        f"\n### Decode attention step, one layer, {title} (`{os.path.relpath(bench, RUNS)}`)\n\n"
        f"{data['device']}, torch {data['torch']}; {args['q_heads']} query / {args['kv_heads']} KV heads, "
        f"head dim {args['head_dim']}, G={args['group_size']}, R={args['residual_length']}; "
        f"{args['warmup']} warmup + {args['iters']} measured steps, median (p10-p90) in ms.\n"
    )
    out.append("| context | rows | BF16 ms | KIVI-2 ms | KIVI-2 speedup | KIVI-4 ms | KIVI-4 speedup |")
    out.append("|---|---|---|---|---|---|---|")
    for (ctx, nrows), d in sorted(rows.items()):
        b, k2, k4 = d["bf16"], d["kivi2"], d["kivi4"]

        def cell(r):
            return f"{r['median_ms']:.3f} ({r['p10_ms']:.3f}-{r['p90_ms']:.3f})"

        out.append(
            f"| {ctx} | {nrows} | {cell(b)} | {cell(k2)} | {b['median_ms'] / k2['median_ms']:.2f}x "
            f"| {cell(k4)} | {b['median_ms'] / k4['median_ms']:.2f}x |"
        )
    if name != "decode_graph":
        continue
    out.append(
        "\nMemory of the same steps: cache bytes (codes, scales and residual rings for KIVI), the extra "
        "allocation of one step (split-K partials and temporaries) and the process-wide allocator peaks "
        "while stepping, in MiB.\n"
    )
    out.append(
        "| context | rows | cache BF16 / KIVI-2 / KIVI-4 | KIVI-2 / KIVI-4 cache ratio "
        "| step workspace BF16 / KIVI-2 / KIVI-4 | peak allocated BF16 / KIVI-2 / KIVI-4 "
        "| peak reserved BF16 / KIVI-2 / KIVI-4 |"
    )
    out.append("|---|---|---|---|---|---|---|")
    mib = 2**20
    for (ctx, nrows), d in sorted(rows.items()):
        if ctx not in (8192, 32768):
            continue
        b, k2, k4 = d["bf16"], d["kivi2"], d["kivi4"]
        out.append(
            f"| {ctx} | {nrows} "
            f"| {b['cache_bytes'] / mib:.1f} / {k2['cache_bytes'] / mib:.1f} / {k4['cache_bytes'] / mib:.1f} "
            f"| {k2['cache_bytes'] / b['cache_bytes']:.3f} / {k4['cache_bytes'] / b['cache_bytes']:.3f} "
            f"| {b['step_workspace_bytes'] / mib:.1f} / {k2['step_workspace_bytes'] / mib:.1f} "
            f"/ {k4['step_workspace_bytes'] / mib:.1f} "
            f"| {b['peak_allocated_bytes'] / mib:.0f} / {k2['peak_allocated_bytes'] / mib:.0f} "
            f"/ {k4['peak_allocated_bytes'] / mib:.0f} "
            f"| {b['peak_reserved_bytes'] / mib:.0f} / {k2['peak_reserved_bytes'] / mib:.0f} "
            f"/ {k4['peak_reserved_bytes'] / mib:.0f} |"
        )

# 2. Teacher-forced long-context quality.
paths = sorted(glob.glob(f"{RUNS}/teacher_forced/*.json")) + sorted(glob.glob(f"{RUNS}/teacher_forced/ablation/*.json"))
for path in paths:
    r = load(path)
    model = os.path.basename(r["model"].rstrip("/"))
    out.append(
        f"\n### Teacher-forced WikiText-2: {model}, {r['rows']} x {r['length']} tokens, "
        f"G={r['kivi_group_size']}, R={r['kivi_residual_length']} (`{os.path.relpath(path, RUNS)}`)\n"
    )
    names = [n for n in next(iter(r["buckets"].values())) if n != "auto"]
    out.append(
        "| positions | BF16 PPL | "
        + " | ".join(f"{NAMES[n]} PPL (change) | {NAMES[n]} KL | {NAMES[n]} top-1" for n in names)
        + " |"
    )
    out.append("|---|---|" + "---|---|---|" * len(names))
    for bucket, d in r["buckets"].items():
        base = d["auto"]["ppl"]
        cells = [
            f"{d[n]['ppl']:.3f} ({100 * (d[n]['ppl'] / base - 1):+.2f}%) | {d[n]['kl_vs_auto']:.4f} "
            f"| {100 * d[n]['top1_agreement_vs_auto']:.1f}%"
            for n in names
        ]
        out.append(f"| {bucket_label(bucket, r['prefill'])} | {base:.3f} | " + " | ".join(cells) + " |")
    ratios = ", ".join(
        f"{NAMES[k]} {v / r['cache_bytes']['auto']:.3f}" for k, v in r["cache_bytes"].items() if k != "auto"
    )
    out.append(f"\nCache bytes relative to BF16, including scales and residual rings: {ratios}.")

# 3. GSM8K sampled evaluation at equal concurrency.
for subdir, model in (("eval_sampled", "Qwen3-0.6B"), ("eval_sampled_17b", "Qwen3-1.7B")):
    paths = sorted(glob.glob(f"{RUNS}/{subdir}/*/summary.json"))
    if not paths:
        continue
    out.append(
        f"\n### GSM8K test with {model}, 1319 prompts x 4 samples, temperature 1.0, seed 1234, "
        "128 concurrent sequences\n"
    )
    out.append("| KV cache | G | R | accuracy +- stderr | generated tok/s | mean response tokens | run |")
    out.append("|---|---|---|---|---|---|---|")
    for path in paths:
        s = load(path)
        rt = s["runtime"]
        out.append(
            f"| {NAMES[s['kv_cache_dtype']]} | {rt.get('kivi_group_size', '-')} "
            f"| {rt.get('kivi_residual_length', '-')} "
            f"| {100 * s['accuracy']:.2f}% +- {100 * s['accuracy_stderr']:.2f} "
            f"| {s['generated_tokens_per_s']:.0f} | {s['mean_response_tokens']:.0f} "
            f"| `{os.path.relpath(os.path.dirname(path), RUNS)}` |"
        )

# 4. Throughput at higher concurrency, short and long prompts.
for subdir, title in (
    ("throughput", "Greedy GSM8K rollout, first 1024 test prompts (up to 2048 new tokens)"),
    ("long_context", "Greedy rollout of 512 GSM8K test problems, each after a ~8K-token WikiText-2 passage"),
):
    pattern = re.compile(r"^(auto|kivi2|kivi4)(?:_r(\d+))?_mrp(\d+)$")
    runs = []
    for run in glob.glob(f"{RUNS}/{subdir}/*"):
        match = pattern.match(os.path.basename(run))
        if match and os.path.isdir(run):
            dtype, residual, mrp = match.groups()
            runs.append((dtype, int(residual or 128), int(mrp), run))
    if not runs:
        continue
    out.append(f"\n### {title}: concurrency vs device memory\n")
    out.append(
        "| KV cache | max running prompts | generated tok/s | peak device memory GiB | accuracy "
        "| mean prompt tokens | outcome |"
    )
    out.append("|---|---|---|---|---|---|---|")
    order = {"auto": 0, "kivi4": 1, "kivi2": 2}
    for dtype, residual, mrp, run in sorted(runs, key=lambda x: (order[x[0]], x[1], x[2])):
        label = NAMES[dtype] + (f" (R={residual})" if residual != 128 else "")
        peak = peak_gib(run)
        peak_text = "-" if peak is None else f"{peak:.1f}"
        summary = os.path.join(run, "summary.json")
        if os.path.exists(summary):
            s = load(summary)
            prompt_tokens = []
            with open(os.path.join(run, "rollouts.jsonl")) as handle:
                for line in handle:
                    prompt_tokens.append(json.loads(line)["prompt_tokens"])
            out.append(
                f"| {label} | {mrp} | {s['generated_tokens_per_s']:.0f} | {peak_text} "
                f"| {100 * s['accuracy']:.1f}% | {statistics.fmean(prompt_tokens):.0f} | completed |"
            )
        else:
            with open(os.path.join(run, "run.log"), errors="replace") as handle:
                log = handle.read()
            outcome = "CUDA out of memory" if "out of memory" in log.lower() else "failed (see run.log)"
            out.append(f"| {label} | {mrp} | - | {peak_text} | - | - | {outcome} |")
    comparisons = sorted(glob.glob(f"{RUNS}/{subdir}/cmp_*.json"))
    if comparisons:
        out.append(
            "\nGreedy token agreement with the first run (`compare_rollouts.py`): mean position of the first "
            "differing response token, and the mean absolute log-probability difference of the shared prefix "
            "(same tokens after the same context, so it isolates the cache's effect on behavior log-probabilities).\n"
        )
        out.append("| baseline vs candidate | first differing token (mean) | abs logprob diff on shared prefix | accuracy change |")
        out.append("|---|---|---|---|")
        for path in comparisons:
            c = load(path)
            name = os.path.basename(path)[4:-5].replace("_vs_", " vs ")
            out.append(
                f"| {name} | {c['mean_first_divergence_token']:.0f} | {c['logprob_abs_diff_mean_on_agreed_prefix']:.4f} "
                f"| {100 * c['accuracy_delta']:+.1f} pts |"
            )

# 5. RL: final-window reward, native logp divergence, curves.
summary_path = f"{RUNS}/rl_final/summary_final.json"
if os.path.exists(summary_path):
    s = load(summary_path)
    window = s["window"]
    out.append(
        f"\n### GSPO on GSM8K train, 3-6 seeds per arm: reward and native log-probability divergence "
        f"(`{os.path.relpath(summary_path, RUNS)}`)\n"
    )
    out.append(
        f"| KV cache | final-window reward per seed (last {window} steps) | mean +- std | change vs BF16 "
        "| train/logp_abs_diff_mean | median step s (rollout s) |"
    )
    out.append("|---|---|---|---|---|---|")
    for arm, e in s["arms"].items():
        per_seed = ", ".join(f"{x['final_window_reward']:.4f}" for x in e["seeds"])
        delta = e.get("final_window_reward_delta_vs_baseline")
        step = statistics.median(x.get("median_train/step_e2e_time_s", float("nan")) for x in e["seeds"])
        rollout = statistics.median(x.get("median_time/rollout", float("nan")) for x in e["seeds"])
        out.append(
            f"| {NAMES[arm]} | {per_seed} | {fmt_ms(e['final_window_reward_mean'], e['final_window_reward_std'])} "
            f"| {'-' if delta is None else f'{delta:+.4f}'} | {e['logp_abs_diff_mean']:.4f} "
            f"| {step:.1f} ({rollout:.1f}) |"
        )
    out.append("\nMean reward over seeds in 10-step windows (the reward curve; per-step values are in the JSON):\n")
    arms = list(s["arms"])
    out.append("| steps | " + " | ".join(NAMES[a] for a in arms) + " |")
    out.append("|---|" + "---|" * len(arms))
    steps = min(len(x["reward_curve"]) for e in s["arms"].values() for x in e["seeds"])
    for start in range(0, steps, 10):
        cells = []
        for arm in arms:
            values = [v for x in s["arms"][arm]["seeds"] for v in x["reward_curve"][start : start + 10]]
            cells.append(f"{statistics.fmean(values):.3f}")
        out.append(f"| {start + 1}-{min(start + 10, steps)} | " + " | ".join(cells) + " |")

# 6. Save/resume.
resume_path = f"{RUNS}/rl_final/summary_resume.json"
if os.path.exists(resume_path):
    s = load(resume_path)
    out.append(
        "\n### Save at step 50 and resume (seed 1): resumed steps 51-100 vs the uninterrupted run "
        f"(`{os.path.relpath(resume_path, RUNS)}`)\n"
    )
    out.append(
        "| KV cache | run | reward steps 51-100 | reward steps 81-100 | train/logp_abs_diff_mean steps 51-100 |"
    )
    out.append("|---|---|---|---|---|")
    for arm, e in s.items():
        for label, key in (("uninterrupted", "uninterrupted"), ("resumed from step 50", "resumed")):
            x = e[key]
            out.append(
                f"| {NAMES[arm]} | {label} | {x['mean_reward']:.4f} | {x['final_window_reward']:.4f} "
                f"| {x['logp_abs_diff_mean']:.4f} |"
            )

# 7. Held-out validation of the trained policies (BF16 KV cache for every policy).
val_runs = sorted(glob.glob(f"{RUNS}/validation/*"))
if val_runs:
    rows = {}
    for run in val_runs:
        name = os.path.basename(run)
        summary = os.path.join(run, "gsm8k_greedy", "summary.json")
        tf = os.path.join(run, "tf_gsm8k_test.json")
        entry = {}
        if os.path.exists(summary):
            entry["accuracy"] = load(summary)["accuracy"]
        if os.path.exists(tf):
            report = load(tf)
            entry["nll"] = tf_nll(report, "auto")
            entry["nll_kivi2"] = tf_nll(report, "kivi2")
            entry["nll_kivi4"] = tf_nll(report, "kivi4")
        rows[name] = entry
    out.append(
        "\n### Held-out validation of the final policies with the BF16 KV cache: greedy GSM8K test accuracy "
        "(1319 problems) and teacher-forced NLL of GSM8K test solutions (8 x 4096 tokens)\n"
    )
    out.append(
        "| policy | accuracy per seed | accuracy mean +- std | change vs BF16 arm "
        "| NLL per seed | NLL mean | NLL change vs BF16 arm |"
    )
    out.append("|---|---|---|---|---|---|---|")
    labels = {
        "auto": "trained with BF16 KV",
        "bf16repeat": "trained with BF16 KV, repeat runs",
        "kivi4": "trained with KIVI-4 KV",
        "kivi2": "trained with KIVI-2 KV",
        "kivi2_r512": "trained with KIVI-2 KV, R = 512",
    }
    base_acc = base_nll = None
    for arm, label in labels.items():
        pattern = re.compile(rf"^s(\d+)_{re.escape(arm)}$")
        names = sorted((int(m.group(1)), name) for name in rows if (m := pattern.match(name)))
        if not names:
            continue
        seeds = [rows[name] for _, name in names]
        label = f"{label} (seeds {', '.join(str(seed) for seed, _ in names)})"
        acc = [x.get("accuracy") for x in seeds]
        nll = [x.get("nll") for x in seeds]
        acc_mean, acc_std = mean_std(acc)
        nll_mean, _ = mean_std(nll)
        if arm == "auto":
            base_acc, base_nll = acc_mean, nll_mean
            acc_change = nll_change = "-"
        else:
            acc_change = (
                "-" if acc_mean is None or base_acc is None else f"{100 * (acc_mean - base_acc):+.2f} pts"
            )
            nll_change = (
                "-" if nll_mean is None or base_nll is None else f"{100 * (nll_mean / base_nll - 1):+.2f}%"
            )
        out.append(
            f"| {label} | {', '.join('-' if a is None else f'{100 * a:.2f}%' for a in acc)} "
            f"| {'-' if acc_mean is None else f'{100 * acc_mean:.2f}%'}"
            f"{'' if acc_std is None else f' +- {100 * acc_std:.2f}'} | {acc_change} "
            f"| {', '.join('-' if v is None else f'{v:.4f}' for v in nll)} "
            f"| {'-' if nll_mean is None else f'{nll_mean:.4f}'} | {nll_change} |"
        )
    extra = {
        "base": "base model",
        "base_serverC": "base model, second host",
        "s1_auto_resume50": "BF16 seed 1, resumed at step 50",
        "s1_kivi2_resume50": "KIVI-2 seed 1, resumed at step 50",
    }
    for name, label in extra.items():
        if name in rows:
            x = rows[name]
            acc = x.get("accuracy")
            nll = x.get("nll")
            out.append(
                f"| {label} | {'-' if acc is None else f'{100 * acc:.2f}%'} | | | "
                f"{'-' if nll is None else f'{nll:.4f}'} | | |"
            )

os.makedirs("/root/kivi/report", exist_ok=True)
with open("/root/kivi/report/tables.md", "w") as handle:
    handle.write("\n".join(out) + "\n")
print("\n".join(out))
