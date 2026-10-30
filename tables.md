
### Decode attention step, one layer, CUDA graph replay (`bench_final/decode_graph.json`)

NVIDIA RTX PRO 5000 72GB Blackwell, torch 2.8.0+cu128; 16 query / 8 KV heads, head dim 128, G=32, R=128; 2 warmup + 20 measured steps, median (p10-p90) in ms.

| context | rows | BF16 ms | KIVI-2 ms | KIVI-2 speedup | KIVI-4 ms | KIVI-4 speedup |
|---|---|---|---|---|---|---|
| 1024 | 1 | 0.012 (0.011-0.013) | 0.012 (0.011-0.013) | 1.00x | 0.012 (0.011-0.014) | 1.00x |
| 1024 | 8 | 0.028 (0.028-0.029) | 0.028 (0.028-0.030) | 1.00x | 0.028 (0.027-0.029) | 1.01x |
| 1024 | 32 | 0.126 (0.126-0.128) | 0.084 (0.083-0.086) | 1.50x | 0.084 (0.083-0.085) | 1.51x |
| 1024 | 128 | 0.464 (0.462-0.466) | 0.333 (0.331-0.334) | 1.39x | 0.338 (0.337-0.339) | 1.37x |
| 1024 | 256 | 0.902 (0.899-0.906) | 0.624 (0.622-0.625) | 1.45x | 0.623 (0.621-0.627) | 1.45x |
| 4096 | 1 | 0.021 (0.019-0.022) | 0.018 (0.017-0.019) | 1.15x | 0.018 (0.017-0.019) | 1.15x |
| 4096 | 8 | 0.120 (0.119-0.122) | 0.060 (0.059-0.064) | 2.00x | 0.059 (0.058-0.061) | 2.03x |
| 4096 | 32 | 0.460 (0.458-0.461) | 0.304 (0.304-0.306) | 1.51x | 0.308 (0.308-0.310) | 1.49x |
| 4096 | 128 | 1.765 (1.764-1.767) | 1.083 (1.082-1.088) | 1.63x | 1.066 (1.064-1.069) | 1.66x |
| 4096 | 256 | 3.492 (3.489-3.496) | 2.106 (2.104-2.115) | 1.66x | 2.069 (2.067-2.077) | 1.69x |
| 8192 | 1 | 0.026 (0.026-0.027) | 0.023 (0.021-0.024) | 1.16x | 0.022 (0.021-0.023) | 1.19x |
| 8192 | 8 | 0.230 (0.228-0.232) | 0.102 (0.101-0.105) | 2.25x | 0.103 (0.102-0.104) | 2.24x |
| 8192 | 32 | 0.908 (0.907-0.911) | 0.562 (0.561-0.564) | 1.62x | 0.554 (0.554-0.556) | 1.64x |
| 8192 | 128 | 3.516 (3.515-3.518) | 2.075 (2.074-2.078) | 1.69x | 2.040 (2.038-2.043) | 1.72x |
| 8192 | 256 | 6.961 (6.959-6.966) | 4.098 (4.095-4.101) | 1.70x | 3.999 (3.997-4.005) | 1.74x |
| 16384 | 1 | 0.037 (0.036-0.039) | 0.047 (0.046-0.048) | 0.78x | 0.045 (0.044-0.047) | 0.81x |
| 16384 | 8 | 0.449 (0.447-0.451) | 0.265 (0.265-0.268) | 1.69x | 0.281 (0.280-0.282) | 1.60x |
| 16384 | 32 | 1.804 (1.802-1.806) | 1.064 (1.062-1.066) | 1.70x | 1.042 (1.040-1.043) | 1.73x |
| 16384 | 128 | 7.008 (7.005-7.013) | 4.058 (4.056-4.060) | 1.73x | 3.945 (3.943-3.950) | 1.78x |
| 16384 | 256 | 13.873 (13.869-13.877) | 8.061 (8.058-8.065) | 1.72x | 7.828 (7.824-7.835) | 1.77x |
| 32768 | 1 | 0.123 (0.122-0.125) | 0.080 (0.078-0.083) | 1.55x | 0.078 (0.077-0.082) | 1.57x |
| 32768 | 8 | 0.882 (0.880-0.884) | 0.533 (0.532-0.536) | 1.65x | 0.521 (0.519-0.522) | 1.69x |
| 32768 | 32 | 3.592 (3.589-3.596) | 2.062 (2.061-2.066) | 1.74x | 2.011 (2.009-2.014) | 1.79x |
| 32768 | 128 | 14.000 (13.996-14.007) | 8.028 (8.025-8.032) | 1.74x | 7.793 (7.789-7.798) | 1.80x |
| 32768 | 256 | 27.717 (27.713-27.728) | 16.006 (16.002-16.014) | 1.73x | 15.532 (15.529-15.540) | 1.78x |

Memory of the same steps: cache bytes (codes, scales and residual rings for KIVI), the extra allocation of one step (split-K partials and temporaries) and the process-wide allocator peaks while stepping, in MiB.

| context | rows | cache BF16 / KIVI-2 / KIVI-4 | KIVI-2 / KIVI-4 cache ratio | step workspace BF16 / KIVI-2 / KIVI-4 | peak allocated BF16 / KIVI-2 / KIVI-4 | peak reserved BF16 / KIVI-2 / KIVI-4 |
|---|---|---|---|---|---|---|
| 8192 | 1 | 33.0 / 6.8 / 10.9 | 0.206 / 0.331 | 0.2 / 0.5 / 0.5 | 33 / 7 / 11 | 42 / 28 / 28 |
| 8192 | 8 | 264.0 / 54.5 / 87.5 | 0.206 / 0.331 | 0.2 / 4.1 / 4.1 | 264 / 60 / 95 | 270 / 102 / 134 |
| 8192 | 32 | 1056.0 / 218.0 / 350.0 | 0.206 / 0.331 | 0.1 / 16.4 / 16.4 | 1056 / 235 / 367 | 1062 / 246 / 378 |
| 8192 | 128 | 4224.0 / 872.0 / 1400.0 | 0.206 / 0.331 | 0.5 / 16.8 / 16.8 | 4226 / 890 / 1418 | 4230 / 894 / 1422 |
| 8192 | 256 | 8448.0 / 1744.0 / 2800.0 | 0.206 / 0.331 | 1.0 / 17.3 / 17.3 | 8451 / 1763 / 2819 | 8456 / 1768 / 2824 |
| 32768 | 1 | 129.0 / 24.8 / 40.9 | 0.192 / 0.317 | 0.2 / 0.5 / 0.5 | 129 / 25 / 41 | 138 / 46 / 62 |
| 32768 | 8 | 1032.0 / 198.5 / 327.5 | 0.192 / 0.317 | 0.2 / 4.1 / 4.1 | 1032 / 203 / 334 | 1038 / 250 / 378 |
| 32768 | 32 | 4128.0 / 794.0 / 1310.0 | 0.192 / 0.317 | 0.1 / 16.4 / 16.4 | 4128 / 811 / 1327 | 4134 / 822 / 1338 |
| 32768 | 128 | 16512.0 / 3176.0 / 5240.0 | 0.192 / 0.317 | 0.5 / 16.8 / 16.8 | 16514 / 3194 / 5258 | 16518 / 3198 / 5262 |
| 32768 | 256 | 33024.0 / 6352.0 / 10480.0 | 0.192 / 0.317 | 1.0 / 17.3 / 17.3 | 33027 / 6371 / 10499 | 33032 / 6376 / 10504 |

### Decode attention step, one layer, eager launches (`bench_final/decode_eager.json`)

NVIDIA RTX PRO 5000 72GB Blackwell, torch 2.8.0+cu128; 16 query / 8 KV heads, head dim 128, G=32, R=128; 2 warmup + 20 measured steps, median (p10-p90) in ms.

| context | rows | BF16 ms | KIVI-2 ms | KIVI-2 speedup | KIVI-4 ms | KIVI-4 speedup |
|---|---|---|---|---|---|---|
| 1024 | 1 | 0.019 (0.018-0.021) | 0.084 (0.082-0.101) | 0.22x | 0.085 (0.082-0.093) | 0.22x |
| 1024 | 32 | 0.130 (0.130-0.132) | 0.142 (0.140-0.148) | 0.92x | 0.140 (0.138-0.147) | 0.93x |
| 1024 | 128 | 0.468 (0.465-0.471) | 0.376 (0.374-0.385) | 1.24x | 0.381 (0.379-0.387) | 1.23x |
| 8192 | 1 | 0.034 (0.033-0.034) | 0.083 (0.081-0.090) | 0.41x | 0.084 (0.082-0.094) | 0.40x |
| 8192 | 32 | 0.914 (0.913-0.916) | 0.618 (0.616-0.627) | 1.48x | 0.610 (0.607-0.614) | 1.50x |
| 8192 | 128 | 3.520 (3.519-3.523) | 2.118 (2.116-2.130) | 1.66x | 2.084 (2.082-2.092) | 1.69x |
| 32768 | 1 | 0.131 (0.129-0.132) | 0.138 (0.137-0.147) | 0.95x | 0.136 (0.135-0.142) | 0.96x |
| 32768 | 32 | 3.595 (3.594-3.597) | 2.119 (2.116-2.126) | 1.70x | 2.067 (2.064-2.076) | 1.74x |
| 32768 | 128 | 14.003 (14.002-14.010) | 8.074 (8.071-8.088) | 1.73x | 7.842 (7.836-7.894) | 1.79x |

### Teacher-forced WikiText-2: Qwen3-0.6B, 4 x 16384 tokens, G=32, R=128 (`teacher_forced/len16384_rows4.json`)

| positions | BF16 PPL | KIVI-2 PPL (change) | KIVI-2 KL | KIVI-2 top-1 | KIVI-4 PPL (change) | KIVI-4 KL | KIVI-4 top-1 |
|---|---|---|---|---|---|---|---|
| [257, 1024) | 15.355 | 16.659 (+8.49%) | 0.1278 | 83.1% | 15.425 (+0.46%) | 0.0048 | 96.8% |
| [1024, 2048) | 25.206 | 27.040 (+7.28%) | 0.1386 | 81.9% | 25.343 (+0.54%) | 0.0054 | 95.9% |
| [2048, 4096) | 23.770 | 25.566 (+7.56%) | 0.1460 | 80.8% | 23.837 (+0.28%) | 0.0055 | 96.3% |
| [4096, 8192) | 22.852 | 25.009 (+9.44%) | 0.1597 | 80.0% | 22.911 (+0.26%) | 0.0057 | 95.7% |
| [8192, 16384) | 16.143 | 18.035 (+11.72%) | 0.1787 | 80.3% | 16.187 (+0.27%) | 0.0059 | 96.0% |

Cache bytes relative to BF16, including scales and residual rings: KIVI-2 0.197, KIVI-4 0.322.

### Teacher-forced WikiText-2: Qwen3-0.6B, 4 x 16384 tokens, G=64, R=128 (`teacher_forced/len16384_rows4_kivi4_g64.json`)

| positions | BF16 PPL | KIVI-4 PPL (change) | KIVI-4 KL | KIVI-4 top-1 |
|---|---|---|---|---|
| [257, 1024) | 15.355 | 15.427 (+0.47%) | 0.0072 | 96.2% |
| [1024, 2048) | 25.206 | 25.305 (+0.39%) | 0.0073 | 95.9% |
| [2048, 4096) | 23.770 | 23.788 (+0.08%) | 0.0077 | 95.6% |
| [4096, 8192) | 22.852 | 22.936 (+0.37%) | 0.0078 | 94.9% |
| [8192, 16384) | 16.143 | 16.169 (+0.16%) | 0.0080 | 95.3% |

Cache bytes relative to BF16, including scales and residual rings: KIVI-4 0.293.

### Teacher-forced WikiText-2: Qwen3-0.6B, 2 x 32768 tokens, G=32, R=128 (`teacher_forced/len32768_rows2.json`)

| positions | BF16 PPL | KIVI-2 PPL (change) | KIVI-2 KL | KIVI-2 top-1 | KIVI-4 PPL (change) | KIVI-4 KL | KIVI-4 top-1 |
|---|---|---|---|---|---|---|---|
| [257, 1024) | 12.629 | 14.185 (+12.32%) | 0.1529 | 83.4% | 12.717 (+0.70%) | 0.0058 | 97.1% |
| [1024, 2048) | 15.602 | 17.336 (+11.12%) | 0.1420 | 83.6% | 15.678 (+0.49%) | 0.0059 | 96.7% |
| [2048, 4096) | 18.654 | 19.940 (+6.90%) | 0.1279 | 82.3% | 18.781 (+0.68%) | 0.0057 | 96.4% |
| [4096, 8192) | 19.110 | 20.791 (+8.79%) | 0.1411 | 82.1% | 19.259 (+0.78%) | 0.0054 | 96.0% |
| [8192, 16384) | 12.203 | 13.660 (+11.94%) | 0.1733 | 81.3% | 12.295 (+0.75%) | 0.0059 | 96.3% |
| [16384, 32768) | 19.053 | 23.729 (+24.54%) | 0.3224 | 73.8% | 19.236 (+0.96%) | 0.0068 | 95.7% |

Cache bytes relative to BF16, including scales and residual rings: KIVI-2 0.192, KIVI-4 0.317.

### Teacher-forced WikiText-2: Qwen3-0.6B, 4 x 4096 tokens, G=32, R=128 (`teacher_forced/ablation/kivi24_g32_r128.json`)

| positions | BF16 PPL | KIVI-2 PPL (change) | KIVI-2 KL | KIVI-2 top-1 | KIVI-4 PPL (change) | KIVI-4 KL | KIVI-4 top-1 |
|---|---|---|---|---|---|---|---|
| [257, 1024) | 20.310 | 22.606 (+11.30%) | 0.1456 | 80.9% | 20.460 (+0.74%) | 0.0051 | 96.6% |
| [1024, 2048) | 19.013 | 20.574 (+8.21%) | 0.1445 | 80.9% | 19.150 (+0.72%) | 0.0060 | 95.3% |
| [2048, 4096) | 20.548 | 22.296 (+8.51%) | 0.1407 | 81.8% | 20.674 (+0.61%) | 0.0053 | 96.2% |

Cache bytes relative to BF16, including scales and residual rings: KIVI-2 0.227, KIVI-4 0.352.

### Teacher-forced WikiText-2: Qwen3-0.6B, 4 x 4096 tokens, G=32, R=256 (`teacher_forced/ablation/kivi2_g32_r256.json`)

| positions | BF16 PPL | KIVI-2 PPL (change) | KIVI-2 KL | KIVI-2 top-1 |
|---|---|---|---|---|
| [257, 1024) | 20.310 | 22.388 (+10.23%) | 0.1296 | 83.3% |
| [1024, 2048) | 19.013 | 20.357 (+7.07%) | 0.1365 | 81.8% |
| [2048, 4096) | 20.548 | 21.987 (+7.00%) | 0.1323 | 82.3% |

Cache bytes relative to BF16, including scales and residual rings: KIVI-2 0.258.

### Teacher-forced WikiText-2: Qwen3-0.6B, 4 x 4096 tokens, G=128, R=128 (`teacher_forced/ablation/kivi4_g128_r128.json`)

| positions | BF16 PPL | KIVI-4 PPL (change) | KIVI-4 KL | KIVI-4 top-1 |
|---|---|---|---|---|
| [257, 1024) | 20.310 | 20.411 (+0.50%) | 0.0105 | 94.6% |
| [1024, 2048) | 19.013 | 19.145 (+0.69%) | 0.0112 | 94.5% |
| [2048, 4096) | 20.548 | 20.592 (+0.21%) | 0.0106 | 94.4% |

Cache bytes relative to BF16, including scales and residual rings: KIVI-4 0.328.

### Teacher-forced WikiText-2: Qwen3-0.6B, 4 x 4096 tokens, G=64, R=128 (`teacher_forced/ablation/kivi4_g64_r128.json`)

| positions | BF16 PPL | KIVI-4 PPL (change) | KIVI-4 KL | KIVI-4 top-1 |
|---|---|---|---|---|
| [257, 1024) | 20.310 | 20.401 (+0.45%) | 0.0078 | 95.7% |
| [1024, 2048) | 19.013 | 19.140 (+0.67%) | 0.0075 | 95.3% |
| [2048, 4096) | 20.548 | 20.600 (+0.25%) | 0.0075 | 95.8% |

Cache bytes relative to BF16, including scales and residual rings: KIVI-4 0.328.

### Teacher-forced WikiText-2: Qwen3-1.7B, 4 x 4096 tokens, G=32, R=128 (`teacher_forced/ablation/qwen17_kivi24_g32_r128.json`)

| positions | BF16 PPL | KIVI-2 PPL (change) | KIVI-2 KL | KIVI-2 top-1 | KIVI-4 PPL (change) | KIVI-4 KL | KIVI-4 top-1 |
|---|---|---|---|---|---|---|---|
| [257, 1024) | 16.940 | 17.754 (+4.81%) | 0.1157 | 85.9% | 17.046 (+0.63%) | 0.0045 | 97.2% |
| [1024, 2048) | 15.902 | 16.199 (+1.87%) | 0.1129 | 85.8% | 15.975 (+0.46%) | 0.0041 | 96.9% |
| [2048, 4096) | 16.201 | 16.446 (+1.51%) | 0.1003 | 87.3% | 16.329 (+0.79%) | 0.0039 | 97.2% |

Cache bytes relative to BF16, including scales and residual rings: KIVI-2 0.227, KIVI-4 0.352.

### GSM8K test with Qwen3-0.6B, 1319 prompts x 4 samples, temperature 1.0, seed 1234, 128 concurrent sequences

| KV cache | G | R | accuracy +- stderr | generated tok/s | mean response tokens | run |
|---|---|---|---|---|---|---|
| BF16 | - | - | 72.16% +- 0.62 | 6766 | 1396 | `eval_sampled/auto` |
| KIVI-2 | 32 | 128 | 68.82% +- 0.64 | 8077 | 1424 | `eval_sampled/kivi2` |
| KIVI-2 | 32 | 256 | 70.24% +- 0.63 | 8044 | 1391 | `eval_sampled/kivi2_g32_r256` |
| KIVI-2 | 32 | 512 | 72.25% +- 0.62 | 7749 | 1356 | `eval_sampled/kivi2_g32_r512` |
| KIVI-4 | 32 | 128 | 72.25% +- 0.62 | 8208 | 1404 | `eval_sampled/kivi4` |
| KIVI-4 | 128 | 128 | 71.91% +- 0.62 | 8467 | 1385 | `eval_sampled/kivi4_g128_r128` |
| KIVI-4 | 64 | 128 | 72.04% +- 0.62 | 8048 | 1401 | `eval_sampled/kivi4_g64_r128` |

### GSM8K test with Qwen3-1.7B, 1319 prompts x 4 samples, temperature 1.0, seed 1234, 128 concurrent sequences

| KV cache | G | R | accuracy +- stderr | generated tok/s | mean response tokens | run |
|---|---|---|---|---|---|---|
| BF16 | - | - | 83.23% +- 0.51 | 5939 | 1603 | `eval_sampled_17b/auto` |
| KIVI-2 | 32 | 128 | 80.48% +- 0.55 | 6959 | 1733 | `eval_sampled_17b/kivi2` |
| KIVI-4 | 32 | 128 | 84.08% +- 0.50 | 7013 | 1595 | `eval_sampled_17b/kivi4` |

### Greedy GSM8K rollout, first 1024 test prompts (up to 2048 new tokens): concurrency vs device memory

| KV cache | max running prompts | generated tok/s | peak device memory GiB | accuracy | mean prompt tokens | outcome |
|---|---|---|---|---|---|---|
| BF16 | 128 | 7228 | 41.9 | 74.8% | 95 | completed |
| BF16 | 192 | 7716 | 59.5 | 74.0% | 95 | completed |
| KIVI-4 | 512 | 10594 | 60.2 | 72.7% | 95 | completed |
| KIVI-2 | 256 | 9627 | 25.0 | 71.8% | 95 | completed |
| KIVI-2 | 512 | 10268 | 42.7 | 69.8% | 95 | completed |

### Greedy rollout of 512 GSM8K test problems, each after a ~8K-token WikiText-2 passage: concurrency vs device memory

| KV cache | max running prompts | generated tok/s | peak device memory GiB | accuracy | mean prompt tokens | outcome |
|---|---|---|---|---|---|---|
| BF16 | 32 | 854 | 45.1 | 70.7% | 8095 | completed |
| BF16 | 48 | 889 | 64.4 | 70.7% | 8095 | completed |
| BF16 | 64 | - | 70.4 | - | - | CUDA out of memory |
| KIVI-4 | 32 | 1206 | 19.5 | 70.5% | 8095 | completed |
| KIVI-4 | 128 | 1345 | 57.6 | 71.1% | 8095 | completed |
| KIVI-2 | 32 | 1175 | 14.7 | 64.6% | 8095 | completed |
| KIVI-2 | 128 | 1286 | 38.4 | 67.2% | 8095 | completed |
| KIVI-2 | 256 | 1281 | 69.7 | 63.9% | 8095 | completed |
| KIVI-2 (R=512) | 32 | 1167 | 16.0 | 69.3% | 8095 | completed |
| KIVI-2 (R=512) | 128 | 1247 | 43.6 | 67.6% | 8095 | completed |

Greedy token agreement with the first run (`compare_rollouts.py`): mean position of the first differing response token, and the mean absolute log-probability difference of the shared prefix (same tokens after the same context, so it isolates the cache's effect on behavior log-probabilities).

| baseline vs candidate | first differing token (mean) | abs logprob diff on shared prefix | accuracy change |
|---|---|---|---|
| auto_mrp32 vs auto_mrp48 | 138 | 0.0078 | +0.0 pts |
| auto_mrp32 vs kivi2_mrp32 | 20 | 0.0405 | -6.1 pts |
| auto_mrp32 vs kivi2_r512_mrp32 | 20 | 0.0408 | -1.4 pts |
| auto_mrp32 vs kivi4_mrp32 | 71 | 0.0154 | -0.2 pts |

### GSPO on GSM8K train, 3-6 seeds per arm: reward and native log-probability divergence (`rl_final/summary_final.json`)

| KV cache | final-window reward per seed (last 20 steps) | mean +- std | change vs BF16 | train/logp_abs_diff_mean | median step s (rollout s) |
|---|---|---|---|---|---|
| BF16 | 0.8080, 0.8104, 0.8031, 0.8057, 0.8016, 0.8053 | 0.8057 +- 0.0032 | - | 0.0152 | 81.6 (47.1) |
| BF16 repeat | 0.8070, 0.8090, 0.8037 | 0.8066 +- 0.0027 | +0.0009 | 0.0152 | 82.1 (47.2) |
| KIVI-4 | 0.8115, 0.8033, 0.8098, 0.8088, 0.8027, 0.7979 | 0.8057 +- 0.0052 | +0.0000 | 0.0316 | 78.5 (42.5) |
| KIVI-2 | 0.6965, 0.6912, 0.7371 | 0.7083 +- 0.0251 | -0.0974 | 0.2086 | 87.8 (47.7) |
| KIVI-2, R = 512 | 0.7793, 0.7602, 0.7686 | 0.7693 +- 0.0096 | -0.0363 | 0.1167 | 67.6 (37.4) |

Mean reward over seeds in 10-step windows (the reward curve; per-step values are in the JSON):

| steps | BF16 | BF16 repeat | KIVI-4 | KIVI-2 | KIVI-2, R = 512 |
|---|---|---|---|---|---|
| 1-10 | 0.722 | 0.724 | 0.720 | 0.693 | 0.717 |
| 11-20 | 0.750 | 0.748 | 0.751 | 0.719 | 0.743 |
| 21-30 | 0.805 | 0.810 | 0.806 | 0.781 | 0.802 |
| 31-40 | 0.776 | 0.779 | 0.780 | 0.743 | 0.773 |
| 41-50 | 0.813 | 0.816 | 0.809 | 0.785 | 0.791 |
| 51-60 | 0.777 | 0.784 | 0.781 | 0.747 | 0.765 |
| 61-70 | 0.799 | 0.796 | 0.804 | 0.764 | 0.792 |
| 71-80 | 0.799 | 0.799 | 0.799 | 0.730 | 0.790 |
| 81-90 | 0.827 | 0.826 | 0.828 | 0.740 | 0.796 |
| 91-100 | 0.784 | 0.787 | 0.783 | 0.677 | 0.743 |

### Save at step 50 and resume (seed 1): resumed steps 51-100 vs the uninterrupted run (`rl_final/summary_resume.json`)

| KV cache | run | reward steps 51-100 | reward steps 81-100 | train/logp_abs_diff_mean steps 51-100 |
|---|---|---|---|---|
| BF16 | uninterrupted | 0.7977 | 0.8080 | 0.0150 |
| BF16 | resumed from step 50 | 0.7916 | 0.7975 | 0.0148 |
| KIVI-2 | uninterrupted | 0.7206 | 0.6965 | 0.2299 |
| KIVI-2 | resumed from step 50 | 0.7473 | 0.7369 | 0.2154 |

### Held-out validation of the final policies with the BF16 KV cache: greedy GSM8K test accuracy (1319 problems) and teacher-forced NLL of GSM8K test solutions (8 x 4096 tokens)

| policy | accuracy per seed | accuracy mean +- std | change vs BF16 arm | NLL per seed | NLL mean | NLL change vs BF16 arm |
|---|---|---|---|---|---|---|
| trained with BF16 KV (seeds 1, 2, 3, 4, 5, 6) | 76.42%, 77.33%, 76.35%, 75.82%, 76.65%, 75.66% | 76.37% +- 0.60 | - | 1.1701, 1.1721, 1.1717, 1.1737, 1.1719, 1.1761 | 1.1726 | - |
| trained with BF16 KV, repeat runs (seeds 1, 2, 3) | 77.94%, 78.01%, 77.03% | 77.66% +- 0.55 | +1.29 pts | 1.1766, 1.1715, 1.1738 | 1.1740 | +0.12% |
| trained with KIVI-4 KV (seeds 1, 2, 3, 4, 5, 6) | 73.62%, 74.91%, 76.95%, 77.71%, 75.97%, 76.35% | 75.92% +- 1.47 | -0.45 pts | 1.1740, 1.1772, 1.1759, 1.1747, 1.1752, 1.1724 | 1.1749 | +0.20% |
| trained with KIVI-2 KV (seeds 1, 2, 3) | 72.63%, 66.57%, 77.33% | 72.18% +- 5.40 | -4.20 pts | 1.1854, 1.1856, 1.1921 | 1.1877 | +1.28% |
| trained with KIVI-2 KV, R = 512 (seeds 1, 2, 3) | 74.68%, 74.53%, 73.92% | 74.37% +- 0.40 | -2.00 pts | 1.1860, 1.1872, 1.1874 | 1.1869 | +1.22% |
| base model | 73.16% | | | 1.1693 | | |
| base model, second host | 73.16% | | | 1.1693 | | |
| BF16 seed 1, resumed at step 50 | 75.97% | | | 1.1708 | | |
| KIVI-2 seed 1, resumed at step 50 | 67.93% | | | 1.1785 | | |
