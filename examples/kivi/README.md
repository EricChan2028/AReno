# KIVI KV cache: measurement report

This directory holds the harnesses behind the measurements of the opt-in KIVI
KV cache ([concept guide](../../docs/concepts/kivi-kv-cache.rst), issue
[#590](https://github.com/inclusionAI/AReno/issues/590)). Every number below
was produced by these scripts; the raw outputs (JSON, JSONL, TensorBoard
events and logs) are listed under [Raw logs](#raw-logs).

"BF16" is the default model-dtype cache (`kv_cache_dtype="auto"`). KIVI uses
group size G = 32 and residual window R = 128 unless a table says otherwise.

## Summary

<!-- FILL:summary -->

## Acceptance criteria

| Issue #590 requirement | Evidence | Status |
|---|---|---|
| Opt-in KIVI storage: per-channel keys, per-token values, scales and a recent unquantized residual window | `RuntimeConfig.kv_cache_dtype = "kivi2" \| "kivi4"`, `kivi_group_size`, `kivi_residual_length`; [concept guide](../../docs/concepts/kivi-kv-cache.rst) | done; default unchanged |
| Incremental decode, window transitions and cache reset; explicit fallback for unsupported attention | fused flush + append per decode step; `clear_kv_cache` releases the cache; `RuntimeWarning` fallback (section 1) | done |
| Actual behavior-policy log-probabilities; cached attention and metadata lifetime validated | rollouts keep the KIVI log-probabilities (section 7); slot reuse, CUDA graph and clear tests (section 1) | done |
| Memory, throughput, generation quality and native training log-probability divergence vs BF16 | sections 2-7 | reported |
| Tests at 1024/4096/8192 cached tokens and residual boundaries; dequantization matches an independent implementation; reset leaks no storage; cached attention within BF16 `atol = rtol = 1e-2` | `tests/test_kivi_kv_cache_cuda.py` (section 1) | 61 tests pass |
| 8192-token case: cache bytes with scales and residual below BF16; total peak including dequantization workspace; decode throughput; quality | sections 2, 3 and 4-6 | <!-- FILL:acc-8192 --> |
| Fixed commit, model revision, seed, dtype, batch/context and software; 2 warmup + at least 10 measured steps; allocated/reserved peaks and step time | [Setup](#setup), section 3 (2 + 20 steps, allocator peaks) | done; discrete GPU, so no unified-memory accounting applies |
| Public `Trainer` end to end with updates and save/resume; native `train/logp_abs_diff_mean` | section 7 | <!-- FILL:acc-e2e --> |
| At least 3 seeds with fixed data and budget; final validation loss at most +2%; final-window reward no more than 0.05 below BF16, with reward curves | section 7 | <!-- FILL:acc-rl --> |
| Core tests and repository checks | `pytest tests/ -k cpu` (same failures as upstream e17794c), `pre-commit run --all-files` | pass |

## Setup

| | |
|---|---|
| GPU | NVIDIA RTX PRO 5000 72GB Blackwell (sm_120), driver 580.126.20, CUDA 12.8; one GPU per run, two hosts with the same image |
| Software | Ubuntu 22.04.5, Python 3.12.11, torch 2.8.0+cu128, triton 3.4.0, flash-attn 2.8.3, transformers 5.22.0, datasets 4.0.0 |
| Code | AReno `main` at e17794c plus this change; each result directory records the commit it ran (`commit.txt`) |
| Model | `Qwen/Qwen3-0.6B` from ModelScope (revision `master`, created 2025-07-26), BF16 weights, `model.safetensors` sha256 `f47f7117…6874b`; `Qwen/Qwen3-1.7B` for one scaling check |
| GSM8K | `gsm8k` `main` from ModelScope through `areno.cli.train`'s dataset resolver: 7473 train / 1319 test rows |
| WikiText-2 | `wikitext-2-raw-v1` test split from the raw parquet file of `Salesforce/wikitext` (sha256 `5f1bea06…6d91`), joined into one text |
| Seeds | rollout sampling seed `seed * 1_000_003 + step` per RL step (seeds 1, 2, 3); evaluation seed 1234 |

The measurements ran on development commits of this change, kept on branch
`kivi-590-dev` of the contributor's fork: 390ac48 (BF16 RL arm, generation and
throughput runs, greedy validation), 5c49afe (KIVI RL arms and the resumed
runs), 5706d51 (teacher-forced runs) and f333cdc (kernel benchmark). Their
`areno/` trees match this change apart from formatting, the G ≥ 32 check and
the allocation log line (`git diff -w <commit> -- areno/`).

## 1. Correctness tests

`tests/test_kivi_kv_cache_cuda.py` checks the Triton kernels against an
independent PyTorch implementation of KIVI quantization written in the test
file:

- stored codes, scales and zero points match the reference for 1024, 4096 and
  8192 cached tokens and for lengths on both sides of every residual-window
  transition (`n - R` just below, at and above a multiple of G);
- decode attention over the paged quantized cache matches attention computed
  from the reference's dequantized tensors within BF16 `atol = rtol = 1e-2`;
- chunked prefill (flash and native backends) and multi-turn continuation read
  the quantized prefix exactly as decode does;
- a released rollout slot is reused without stale data, CUDA graph replay
  matches eager execution, and clearing the KV cache frees every KIVI tensor;
- at 8192 tokens the cache, including scales, zero points and residual rings,
  is smaller than the BF16 cache.

`tests/test_kivi_kv_cache_cpu.py` covers configuration validation, the explicit
fallbacks, residual-window bookkeeping and prefill planning without a GPU.

```bash
pytest tests/test_kivi_kv_cache_cpu.py tests/test_kivi_kv_cache_cuda.py   # 61 passed on the GPU host
pytest tests/ -k cpu                                                       # same failures as upstream e17794c
```

## 2. Cache memory

Per token and KV head (head dim 128, G = 32) BF16 stores 512 bytes of K and V.
KIVI-2 stores 64 bytes of codes plus 32 bytes of FP16 scales and zero points
(18.75%); KIVI-4 stores 128 + 32 bytes (31.25%). With G = 64 the metadata
halves (KIVI-4: 28.1%). Each rollout slot also keeps R + G full-precision
tokens in its residual ring, a cost independent of the context length.

In the RL runs of section 7 (128 concurrent prompts, prompts of up to 512
tokens plus 2048 new tokens, so 1281 paged blocks of 256 tokens and 129 rollout
slots), the engine allocates:

| KV cache | allocated bytes (from the engine log) | relative to BF16 |
|---|---|---|
| BF16 | 37,609,488,384 (1281 × 256 tokens × 114,688 bytes) | 1.000 |
| KIVI-4 | 14,120,386,560 | 0.375 |
| KIVI-2 | 9,419,096,064 | 0.250 |

Here the rings (129 slots × 160 tokens, 2.2 GiB) are a quarter of the KIVI-2
pool; for longer sequences the ratio approaches 0.1875 / 0.3125, as the
8192-token rows below show.

<!-- FILL:memory-8192 -->

## 3. Decode attention kernel

`bench_decode_attention.py` times one decode step of one attention layer
(Qwen3-0.6B shape: 16 query heads, 8 KV heads, head dim 128) at a fixed number
of rows and cached tokens. The KIVI step includes everything the runtime
launches per layer: the residual-group flush, the ring append and the fused
dequantize + split-K attention. The BF16 step is `flash_attn_with_kvcache`
with the in-place append used by AReno's flash backend. Each configuration
warms up for 2 steps and is measured for 20.

<!-- FILL:decode -->

## 4. Quality: teacher-forced long context

`teacher_forced_eval.py` feeds the same WikiText-2 windows through the model
token by token with each cache and compares next-token predictions with the
BF16 cache: perplexity, KL(BF16 ‖ KIVI) and top-1 agreement, per position
range. The first 256 tokens are prefilled.

<!-- FILL:teacher_forced -->

## 5. Quality: generation

`rollout_eval.py` runs the public `Trainer` rollout on GSM8K test problems and
scores them with `examples/math/math_verify_reward.py`. Sampled runs use
4 samples per problem at temperature 1.0 (the RL setting) with seed 1234.

<!-- FILL:gsm8k -->

## 6. Rollout throughput and concurrency

The same harness with greedy decoding at different `max_running_prompts`.
Peak device memory is the largest `nvidia-smi` reading sampled once per second
during the run (weights, KV pool, CUDA graphs and allocator cache). AReno sizes
the KV pool as `max_running_prompts × ceil((prompt + max_new_tokens) / 256)`
blocks, so the pool, not the model, decides how many rollouts fit.

<!-- FILL:throughput -->

## 7. RL with the public Trainer

`train_seeded.py` runs `areno train` unchanged except for the KV-cache dtype
and a per-step rollout seed: GSPO on GSM8K train, Qwen3-0.6B, 32 prompts × 8
samples per step, up to 2048 new tokens at temperature 1.0, 128 concurrent
prompts, learning rate 1e-6 with cosine decay, 100 steps, checkpoints every 50
steps, seeds 1-3 per arm. Rollout log-probabilities are the ones the KIVI
attention produced while sampling; the trainer recomputes training
log-probabilities with the BF16 forward, so the native
`train/logp_abs_diff_mean` measures the KV-cache error in the behavior policy.
Rewards are the verifier's 0/1 scores.

<!-- FILL:rl -->

## 8. Limitations

- Scope follows the issue: one GPU, the dense `qwen3` family and private
  per-request caches. Other model families, speculative decoding and NPU fall
  back to the BF16 cache with a `RuntimeWarning`; tensor-parallel rollout was
  not measured.
- Only decode attention reads quantized keys and values. Attention over fresh
  prompt tokens runs at full precision, so prefill cost is unchanged.
- AReno sizes the KV pool from `max_running_prompts` and the longest
  prompt + response, not from free memory. KIVI lowers the memory of a given
  configuration; using the headroom for more concurrent rollouts means raising
  `max_running_prompts`.
- The residual ring costs R + G model-dtype tokens per rollout slot regardless
  of context length (2.2 GiB for the 129 slots of the RL runs below, Qwen3-0.6B
  at R = 128). It dominates the cache for short sequences.
- With one row, few KV heads and 16K tokens the decode kernel is slower than
  flash-attn (table in section 3): the split-K grid cannot fill the GPU.
- KIVI-2 with R = 128 loses measurable quality on Qwen3-0.6B (sections 4-6);
  the loss is smaller on Qwen3-1.7B and with R = 512.
- The RL comparison covers 100 GSPO steps of one 0.6B model. AReno checkpoints
  store model weights only, so a resumed run starts fresh AdamW moments and a
  fresh learning-rate schedule; the BF16 resume control goes through the same
  path. Qwen3-0.6B declares tied embeddings, which `main` does not tie yet
  (#592); any effect on save/load applies to both arms equally.

## Reproducing

Assets come from ModelScope, except WikiText-2: ModelScope's `wikitext`
dataset ships a loading script, which `datasets` 4.x rejects ("Dataset scripts
are no longer supported"), so the raw parquet file of the `Salesforce/wikitext`
repository is read directly.

```bash
modelscope download --model Qwen/Qwen3-0.6B --local_dir models/Qwen3-0.6B
python - <<'PY'
import pyarrow.parquet as pq
from modelscope.msdatasets import MsDataset

for split in ("train", "test"):
    dataset = MsDataset.load("gsm8k", subset_name="main", split=split, trust_remote_code=True)
    dataset.to_hf_dataset().save_to_disk(f"data/gsm8k_main_{split}")
# wikitext-2-raw-v1/test-00000-of-00001.parquet from the Salesforce/wikitext dataset repository
table = pq.read_table("data/wikitext2_raw_test.parquet")
with open("data/wikitext2_test.txt", "w", encoding="utf-8") as handle:
    handle.write("".join(table.column("text").to_pylist()))
test = MsDataset.load("gsm8k", subset_name="main", split="test", trust_remote_code=True).to_hf_dataset()
with open("data/gsm8k_test_solutions.txt", "w", encoding="utf-8") as handle:
    for row in test:
        handle.write(f"Question: {row['question']}\nAnswer: {row['answer']}\n\n")
PY
M=models/Qwen3-0.6B
D=data
```

Section 3, decode kernel:

```bash
python examples/kivi/bench_decode_attention.py --cuda-graph --warmup 2 --iters 20 \
    --rows 1 8 32 128 256 --context 1024 4096 8192 16384 32768 --output decode_graph.json
python examples/kivi/bench_decode_attention.py --warmup 2 --iters 20 \
    --rows 1 32 128 --context 1024 8192 32768 --output decode_eager.json
```

Section 4, teacher-forced quality (ablations add `--kivi-group-size`,
`--kivi-residual-length` and `--kv-cache-dtypes`; the 1.7B row uses
`--model models/Qwen3-1.7B --length 4096`):

```bash
python examples/kivi/teacher_forced_eval.py --model $M --text $D/wikitext2_test.txt \
    --length 16384 --rows 4 --prefill 256 --output tf_len16384.json
python examples/kivi/teacher_forced_eval.py --model $M --text $D/wikitext2_test.txt \
    --length 32768 --rows 2 --prefill 256 --output tf_len32768.json
```

Sections 5 and 6, generation quality and throughput:

```bash
for dtype in auto kivi2 kivi4; do
    python examples/kivi/rollout_eval.py --model $M --dataset $D/gsm8k_main_test --kv-cache-dtype $dtype \
        --n-samples 4 --temperature 1.0 --seed 1234 --max-running-prompts 128 --output gsm8k/$dtype
done
# Greedy, first 1024 problems, at a chosen concurrency; peak memory sampled with
# `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits` once per second.
python examples/kivi/rollout_eval.py --model $M --dataset $D/gsm8k_main_test --kv-cache-dtype kivi2 \
    --limit 1024 --max-running-prompts 512 --output throughput/kivi2_mrp512
# Long context: 512 problems after ~8K-token passages.
python examples/kivi/build_long_context_gsm8k.py --tokenizer $M --gsm8k $D/gsm8k_main_test \
    --text $D/wikitext2_test.txt --output $D/gsm8k_test512_ctx8k
python examples/kivi/rollout_eval.py --model $M --dataset $D/gsm8k_test512_ctx8k --kv-cache-dtype kivi4 \
    --max-prompt-tokens 9216 --max-running-prompts 128 --output long_context/kivi4_mrp128
python examples/kivi/compare_rollouts.py long_context/auto_mrp32 long_context/kivi4_mrp32 \
    --output long_context/cmp_auto_mrp32_vs_kivi4_mrp32.json
```

Section 7, RL (three seeds per arm), save/resume and held-out validation:

```bash
TRAIN_ARGS="--algo gspo --dataset-path $D/gsm8k_main_train --dataset-loader-fn examples/math/dataset_loader.py \
    --reward-fn-path examples/math/math_verify_reward.py --world-size 1 --tp-size 1 --batch-size 32 \
    --n-samples 8 --max-running-prompts 128 --max-prompt-tokens 512 --max-new-tokens 2048"
for seed in 1 2 3; do
    for dtype in auto kivi2 kivi4; do
        python examples/kivi/train_seeded.py --seed $seed --kv-cache-dtype $dtype -- $TRAIN_ARGS --ckpt $M \
            --max-steps 100 --save-interval 50 \
            --metrics-log-dir rl/s${seed}_$dtype/tb --save-path rl/s${seed}_$dtype/ckpt
    done
done
python examples/kivi/summarize_rl.py --window 20 --output rl/summary.json --plot rl/rewards.png \
    auto=rl/s1_auto,rl/s2_auto,rl/s3_auto kivi2=rl/s1_kivi2,rl/s2_kivi2,rl/s3_kivi2 \
    kivi4=rl/s1_kivi4,rl/s2_kivi4,rl/s3_kivi4
# Resume seed 1 from its step-50 checkpoint: skip the 50 x 32 prompts already
# used and continue the per-step seed schedule for the remaining 50 steps.
python examples/kivi/train_seeded.py --seed 1 --kv-cache-dtype kivi2 --resume-step 50 --resume-prompts 1600 -- \
    $TRAIN_ARGS --ckpt rl/s1_kivi2/ckpt/step_000050 --max-steps 50 --save-interval 50 \
    --metrics-log-dir rl/s1_kivi2_resume50/tb --save-path rl/s1_kivi2_resume50/ckpt
# Validate a final policy with the BF16 cache.
python examples/kivi/rollout_eval.py --model rl/s1_kivi2/ckpt/step_000100 --dataset $D/gsm8k_main_test \
    --max-running-prompts 128 --output validation/s1_kivi2/gsm8k_greedy
python examples/kivi/teacher_forced_eval.py --model rl/s1_kivi2/ckpt/step_000100 \
    --text $D/gsm8k_test_solutions.txt --length 4096 --rows 8 --prefill 256 --kv-cache-dtypes kivi2 kivi4 \
    --output validation/s1_kivi2/tf_gsm8k_test.json
```

Every KIVI run logs the allocated cache once per engine, for example
`KIVI KV cache: dtype=kivi2 group_size=32 residual_length=128 blocks=1281 slots=129 bytes=9419096064`;
a run without that line used the BF16 cache.

## Raw logs

<!-- FILL:raw -->
