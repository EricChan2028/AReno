# Raw logs for the KIVI KV cache report (inclusionAI/AReno#590)

These are the raw outputs behind `examples/kivi/README.md` in the KIVI pull
request. Paths below are relative to this directory; every result directory
holds the commit it ran (`commit.txt` or `*.commit`).

Those files record the hashes the development commits had before their
messages were edited. `commit-map.txt` pairs each recorded hash with the
commit on branch `kivi-590-dev` that has the same tree. In
`runs/teacher_forced/`, `commit_16384.txt` and `commit_32768.txt` come from a
first launch at 7a71205 that was stopped and relaunched at 5706d51
(`len*.commit`); the relaunch produced the reported JSON.

| Directory | Report section | Contents |
|---|---|---|
| `runs/bench_final/` | 3 | decode benchmark JSON (CUDA graph and eager), logs, GPU state before the run |
| `runs/teacher_forced/` | 4 | teacher-forced JSON (perplexity, KL, top-1 per position range) and logs |
| `runs/eval_sampled/`, `runs/eval_sampled_17b/` | 5 | GSM8K evaluation summaries, one line per completion (`completions.jsonl`), logs |
| `runs/throughput/`, `runs/long_context/` | 6 | rollout summaries, 1 Hz `nvidia-smi` memory traces, logs; long-context runs keep full token dumps (`rollouts.jsonl.xz`) for `compare_rollouts.py`, whose outputs are `cmp_*.json` |
| `runs/rl_final/` | 7 | per-run `train.log`, TensorBoard events (`tb/`), run config, software versions, GPU; `summary_final.json`, `rewards_final.png`, `summary_resume.json` |
| `runs/validation/` | 7 | held-out validation of every final policy: greedy GSM8K summary and teacher-forced NLL JSON |
| `runs/from_server1/` | 7 | sha256 of the step-50 checkpoint files copied between hosts for the BF16 resume run |
| `runs/ci_emulation/` | 1 | full CPU suite with triton, flash-attn and CUDA hidden, on this change and on upstream e17794c |
| `scripts/`, `scripts_serverC/` | all | the shell and Python drivers that launched every run on the two hosts |
| `env/` | setup | `pip freeze`, `nvidia-smi`, `nvcc`, OS release, sha256 of data and model files |
| `tables.md` | all | tables generated from these files by `scripts/make_report.py` |

`runs/rl_final/s*_bf16repeat` were launched as KIVI-2 runs through the first
version of `train_seeded.py`, whose override never reached the GSPO trainer
config, so they ran the BF16 cache with the same seeds (see `NOTE.txt` in each).
They serve as a run-to-run noise control for the BF16 arm. Their first step
reproduces the BF16 run of the same seed exactly (identical rollouts: response
length, loss and `logp_abs_diff_mean` agree to every printed digit), which is
how the problem was found; afterwards the runs drift apart through
nondeterministic gradient reductions. The KIVI arms were relaunched with the
fixed harness, and their logs show the engine's `KIVI KV cache:` allocation
line.
