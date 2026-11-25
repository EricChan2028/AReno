Environment: python 3.10.12 Linux-6.6.114.1-microsoft-standard-WSL2-x86_64-with-glibc2.35; torch 2.9.1+cu128 cuda 12.8 triton 3.5.1 transformers 5.20.2; gpu NVIDIA GeForce RTX 5090 Laptop GPU capability (12, 0) memory_gib 23.89; flash_attn not installed (optional since the native default)
Commit: ed72d98 test(kv): follow the native default in KIVI tests and harnesses (WIP)
Extension: reused /root/actc591/AReno/areno/accel/_areno_accel.cpython-310-x86_64-linux-gnu.so sha256=daeeac8f36873f399aef7b614e943c4c9ba8dea0d3f9588f0a5cde3dd9a9117c (built on this machine for the AReno #591 work from the same areno/accel/csrc tree and setup.py); daeeac8f36873f399aef7b614e943c4c9ba8dea0d3f9588f0a5cde3dd9a9117c  /root/kivi590/AReno-pr/areno/accel/_areno_accel.cpython-310-x86_64-linux-gnu.so; csrc_tree=f7818d61ba1f4a271be0ed55166c4734123a975a setup_py=6b25073a2c82bdc8a949310ba8e51a9660fc30a5 torch=2.9.1+cu128

#### Tests

- tests_kivi: 59 passed, 2 skipped in 30.14s
- tests_upstream_gpu: 2 failed, 59 passed, 5 skipped in 67.27s (0:01:07)
- tests_upstream_gpu_pr: 61 passed, 5 skipped in 189.67s (0:03:09)
- tests_upstream_gpu_main: 61 passed, 5 skipped in 10.61s

#### Decode attention step, one layer, BF16 native backend vs KIVI (median of 20 after 2 warmup)

| mode | rows | cached tokens | BF16 native ms | KIVI-4 ms (speedup) | KIVI-2 ms (speedup) | KIVI-4 / BF16 bytes | KIVI-2 / BF16 bytes |
|---|---|---|---|---|---|---|---|
| graph | 1 | 1024 | 0.046 | 0.146 (0.31x) | 0.041 (1.13x) | 0.535 | 0.411 |
| graph | 8 | 1024 | 0.102 | 0.040 (2.54x) | 0.092 (1.10x) | 0.560 | 0.411 |
| graph | 32 | 1024 | 0.259 | 0.157 (1.65x) | 0.134 (1.93x) | 0.535 | 0.411 |
| graph | 128 | 1024 | 0.986 | 0.476 (2.07x) | 0.556 (1.77x) | 0.460 | 0.336 |
| graph | 1 | 4096 | 0.025 | 0.132 (0.19x) | 0.059 (0.42x) | 0.376 | 0.252 |
| graph | 8 | 4096 | 0.191 | 0.098 (1.96x) | 0.094 (2.02x) | 0.376 | 0.252 |
| graph | 32 | 4096 | 0.661 | 0.427 (1.55x) | 0.454 (1.46x) | 0.376 | 0.252 |
| graph | 128 | 4096 | 2.957 | 1.489 (1.99x) | 1.519 (1.95x) | 0.354 | 0.230 |
| graph | 1 | 8192 | 0.032 | 0.042 (0.76x) | 0.039 (0.80x) | 0.344 | 0.220 |
| graph | 8 | 8192 | 0.334 | 0.201 (1.66x) | 0.166 (2.02x) | 0.344 | 0.220 |
| graph | 32 | 8192 | 1.286 | 0.770 (1.67x) | 0.788 (1.63x) | 0.344 | 0.220 |
| graph | 128 | 8192 | 6.800 | 3.054 (2.23x) | 3.599 (1.89x) | 0.333 | 0.209 |
| graph | 1 | 16384 | 0.070 | 0.055 (1.27x) | 0.084 (0.84x) | 0.327 | 0.203 |
| graph | 8 | 16384 | 0.983 | 0.374 (2.63x) | 0.432 (2.27x) | 0.327 | 0.203 |
| graph | 32 | 16384 | 2.591 | 1.610 (1.61x) | 1.592 (1.63x) | 0.327 | 0.203 |
| graph | 128 | 16384 | - | 7.401 | 7.170 | - | - |
| graph | 1 | 32768 | 0.268 | 0.090 (2.97x) | 0.097 (2.78x) | 0.318 | 0.194 |
| graph | 8 | 32768 | 1.947 | 0.783 (2.49x) | 0.807 (2.41x) | 0.319 | 0.195 |
| graph | 32 | 32768 | 6.719 | 2.853 (2.35x) | 3.482 (1.93x) | 0.319 | 0.195 |
| eager | 1 | 1024 | 0.148 | 0.239 (0.62x) | 0.236 (0.62x) | 0.535 | 0.411 |
| eager | 32 | 1024 | 0.341 | 0.319 (1.07x) | 0.277 (1.23x) | 0.535 | 0.411 |
| eager | 128 | 1024 | 1.073 | 0.727 (1.48x) | 0.731 (1.47x) | 0.460 | 0.336 |
| eager | 1 | 8192 | 0.132 | 0.235 (0.56x) | 0.271 (0.49x) | 0.344 | 0.220 |
| eager | 32 | 8192 | 1.392 | 0.952 (1.46x) | 0.997 (1.40x) | 0.344 | 0.220 |
| eager | 128 | 8192 | 6.682 | 3.407 (1.96x) | 3.741 (1.79x) | 0.333 | 0.209 |

#### Teacher-forced WikiText-2, Qwen3-0.6B, 4 x 16384 tokens, G = 32, R = 128: perplexity (change vs BF16)

| positions | BF16 | KIVI-4 | KIVI-2 | report (RTX PRO 5000): BF16 / KIVI-4 / KIVI-2 |
|---|---|---|---|---|
| <1024 | 15.346 | 15.422 (+0.50%) | 16.628 (+8.36%) | 15.355 / 15.425 / 16.659 |
| <2048 | 25.224 | 25.371 (+0.58%) | 27.218 (+7.91%) | 25.206 / 25.343 / 27.040 |
| <4096 | 23.747 | 23.826 (+0.33%) | 25.696 (+8.21%) | 23.770 / 23.837 / 25.566 |
| <8192 | 22.862 | 22.902 (+0.17%) | 24.976 (+9.25%) | 22.852 / 22.911 / 25.009 |
| <16384 | 16.146 | 16.192 (+0.28%) | 18.030 (+11.67%) | 16.143 / 16.187 / 18.035 |

cache bytes: {'auto': 7516192768, 'kivi2': 1482686464, 'kivi4': 2422210560}, decode seconds 2195

#### GSPO with areno train (seed 1): trainer-native train/logp_abs_diff_mean and reward

| KV cache | steps | logp_abs_diff_mean | reward | report, same first steps: logp_abs_diff_mean / reward | report, 100 steps: logp_abs_diff_mean |
|---|---|---|---|---|---|
| BF16 | 8 | 0.0000 | 0.770 | 0.0156 / 0.729 | 0.0151 |
| KIVI-4 | 8 | 0.0319 | 0.734 | 0.0326 / 0.727 | 0.0318 |
| KIVI-2 | 8 | 0.1933 | 0.725 | 0.1911 / 0.696 | 0.2110 |
| KIVI-2 R=512 | 8 | 0.1154 | 0.766 | 0.1183 / 0.727 | 0.1140 |

#### Save at step 4 and resume (KIVI-4, seed 1): steps after the checkpoint

| step | uninterrupted reward / logp_abs_diff_mean | resumed reward / logp_abs_diff_mean |
|---|---|---|
| 5 | 0.828 / 0.0306 | 0.844 / 0.0307 |
| 6 | 0.781 / 0.0317 | 0.812 / 0.0320 |
| 7 | 0.719 / 0.0335 | 0.766 / 0.0327 |
| 8 | 0.594 / 0.0319 | 0.703 / 0.0331 |

#### GSM8K after ~8K-token passages, first 64 problems of the report's set, greedy, max 2048 new tokens

| KV cache | concurrent | accuracy | mean response tokens | tokens/s | peak device memory | KIVI cache bytes | report, same 64 problems at 32 concurrent |
|---|---|---|---|---|---|---|---|
| BF16 | 8 | 0.719 | 924 | 306 | 15.7 GiB | - | 0.656 |
| KIVI-2 | 40 | 0.609 | 916 | 425 | 16.1 GiB | 9.73 GiB | 0.625 |
| KIVI-2 | 8 | 0.594 | 851 | 358 | 8.1 GiB | 1.96 GiB | 0.625 |
| KIVI-4 | 24 | 0.750 | 920 | 415 | 15.9 GiB | 9.46 GiB | 0.719 |
| KIVI-4 | 8 | 0.719 | 900 | 364 | 9.3 GiB | 3.17 GiB | 0.719 |

#### GSM8K test, first 512 problems, one sample at T = 1, seed 1234, 32 concurrent

| KV cache | accuracy +- SE | mean response tokens | tokens/s | peak device memory | report, sample 0 of the same 512 problems |
|---|---|---|---|---|---|
| BF16 | 0.746 +- 0.019 | 1394 | 2338 | 13.4 GiB | 0.756 |
| KIVI-4 | 0.734 +- 0.020 | 1404 | 2296 | 8.2 GiB | 0.729 |
| KIVI-2 | 0.691 +- 0.020 | 1413 | 2290 | 7.1 GiB | 0.691 |

#### Run log

2026-11-25 10:26:07 extension: mode=reuse csrc=f7818d61ba1f4a271be0ed55166c4734123a975a setup.py=6b25073a2c82bdc8a949310ba8e51a9660fc30a5 torch=2.9.1+cu128 suffix=.cpython-310-x86_64-linux-gnu.so
2026-11-25 10:26:15 EXTENSION_READY
2026-11-25 12:46:16 AReno extension missing; run g0_extension.sh first
2026-11-25 12:47:05 G1 start at ed72d98 test(kv): follow the native default in KIVI tests and harnesses (WIP)
python 3.10.12 Linux-6.6.114.1-microsoft-standard-WSL2-x86_64-with-glibc2.35
torch 2.9.1+cu128 cuda 12.8 triton 3.5.1 transformers 5.20.2
gpu NVIDIA GeForce RTX 5090 Laptop GPU capability (12, 0) memory_gib 23.89
flash_attn not installed (optional since the native default)
2026-11-25 12:47:17 start tests_kivi
2026-11-25 12:47:48 end tests_kivi exit=0 seconds=31 peak_vram=860MiB peak_job_rss=1568MiB peak_wsl_used=4775MiB
2026-11-25 12:47:48 start tests_upstream_gpu
2026-11-25 12:48:55 start bench_graph
2026-11-25 12:49:07 start bench_graph_32k
2026-11-25 12:49:13 end bench_graph_32k exit=0 seconds=6 peak_vram=1072MiB peak_job_rss=798MiB peak_wsl_used=7083MiB
2026-11-25 12:49:13 start bench_eager
2026-11-25 12:49:17 end bench_eager exit=0 seconds=4 peak_vram=4578MiB peak_job_rss=754MiB peak_wsl_used=8579MiB
2026-11-25 12:49:18 start tf_len16384
2026-11-25 13:24:47 end tf_len16384 exit=0 seconds=2129 peak_vram=14118MiB peak_job_rss=1423MiB peak_wsl_used=11037MiB
2026-11-25 13:24:47 G1_DONE
2026-11-25 13:25:04 G1b start at ed72d98 test(kv): follow the native default in KIVI tests and harnesses (WIP)
2026-11-25 13:25:04 start tests_upstream_gpu_pr
2026-11-25 13:28:10 end tests_upstream_gpu_pr exit=0 seconds=186 peak_vram=780MiB peak_job_rss=2653MiB peak_wsl_used=10350MiB
2026-11-25 13:28:10 start tests_upstream_gpu_main
2026-11-25 13:28:22 end tests_upstream_gpu_main exit=0 seconds=12 peak_vram=634MiB peak_job_rss=2192MiB peak_wsl_used=7246MiB
2026-11-25 13:28:22 start bench_graph
2026-11-25 13:28:29 end bench_graph exit=0 seconds=7 peak_vram=3442MiB peak_job_rss=757MiB peak_wsl_used=6300MiB
2026-11-25 13:28:29 start bench_graph_16k
2026-11-25 13:28:34 end bench_graph_16k exit=0 seconds=5 peak_vram=1024MiB peak_job_rss=753MiB peak_wsl_used=6636MiB
2026-11-25 13:28:34 start bench_graph_128x16k_kivi
2026-11-25 13:28:38 end bench_graph_128x16k_kivi exit=0 seconds=4 peak_vram=12372MiB peak_job_rss=750MiB peak_wsl_used=6710MiB
2026-11-25 13:28:38 G1B_DONE
2026-11-25 13:28:59 G2 start at ed72d98 test(kv): follow the native default in KIVI tests and harnesses (WIP), STEPS=8
2026-11-25 13:28:59 start rl_s1_auto
2026-11-25 13:38:52 end rl_s1_auto exit=0 seconds=593 peak_vram=16472MiB peak_job_rss=12441MiB peak_wsl_used=20906MiB
2026-11-25 13:38:52 start rl_s1_kivi4
2026-11-25 13:38:53 switch: stopping runner 860432 at the start of KIVI-4 to apply the per-arm caps
2026-11-25 13:39:07 switch: continuing with g2b.sh (kivi4 kivi2 kivi2_r512 resume)
2026-11-25 13:39:10 G2 start at ed72d98 test(kv): follow the native default in KIVI tests and harnesses (WIP), STEPS=8
2026-11-25 13:39:10 start rl_s1_kivi4 (job RAM cap 16384 MiB)
2026-11-25 13:49:10 end rl_s1_kivi4 exit=0 seconds=600 peak_vram=16574MiB peak_job_rss=12227MiB peak_wsl_used=13860MiB
2026-11-25 13:49:10 start rl_s1_kivi2 (job RAM cap 14336 MiB)
2026-11-25 13:59:49 guard bug: runner exited at the start of rl_s1_kivi2; unguarded job stopped, to be rerun
2026-11-25 14:00:28 G2 start at ed72d98 test(kv): follow the native default in KIVI tests and harnesses (WIP), STEPS=8
2026-11-25 14:00:28 start rl_s1_kivi2 (job RAM cap 14336 MiB)
2026-11-25 14:10:25 end rl_s1_kivi2 exit=0 seconds=597 peak_vram=16536MiB peak_job_rss=12401MiB peak_wsl_used=13407MiB
2026-11-25 14:10:25 start rl_s1_kivi2_r512 (job RAM cap 14336 MiB)
2026-11-25 14:19:58 end rl_s1_kivi2_r512 exit=0 seconds=573 peak_vram=16566MiB peak_job_rss=12481MiB peak_wsl_used=20585MiB
2026-11-25 14:19:58 start rl_s1_kivi4_resume4 (job RAM cap 16384 MiB)
2026-11-25 14:25:02 end rl_s1_kivi4_resume4 exit=0 seconds=304 peak_vram=16562MiB peak_job_rss=12003MiB peak_wsl_used=19010MiB
auto: final-window reward 0.7695 [baseline]; logp_abs_diff_mean 0.0000
kivi4: final-window reward 0.7344 delta vs baseline -0.0352; logp_abs_diff_mean 0.0319
kivi2: final-window reward 0.7539 delta vs baseline -0.0156; logp_abs_diff_mean 0.1927
kivi2_r512: final-window reward 0.7656 delta vs baseline -0.0039; logp_abs_diff_mean 0.1154
2026-11-25 14:25:03 G2_RL_DONE
2026-11-25 14:25:23 G2 start at ed72d98 test(kv): follow the native default in KIVI tests and harnesses (WIP), STEPS=8
2026-11-25 14:25:30 start lc64_auto_mrp8 (job RAM cap 14336 MiB)
2026-11-25 14:28:59 end lc64_auto_mrp8 exit=0 seconds=209 peak_vram=16126MiB peak_job_rss=3915MiB peak_wsl_used=6158MiB
2026-11-25 14:28:59 start lc64_kivi4_mrp8 (job RAM cap 14336 MiB)
2026-11-25 14:45:58 end lc64_kivi4_mrp8 exit=0 seconds=1019 peak_vram=9540MiB peak_job_rss=3989MiB peak_wsl_used=8114MiB
2026-11-25 14:45:58 start lc64_kivi2_mrp8 (job RAM cap 14336 MiB)
2026-11-25 15:02:53 end lc64_kivi2_mrp8 exit=0 seconds=1015 peak_vram=8308MiB peak_job_rss=3968MiB peak_wsl_used=8974MiB
2026-11-25 15:02:53 start lc64_kivi4_mrp24 (job RAM cap 14336 MiB)
2026-11-25 15:41:32 end lc64_kivi4_mrp24 exit=124 seconds=2319 peak_vram=16312MiB peak_job_rss=2960MiB peak_wsl_used=12736MiB
2026-11-25 15:41:32 start lc64_kivi2_mrp40 (job RAM cap 14336 MiB)
2026-11-25 16:20:23 end lc64_kivi2_mrp40 exit=124 seconds=2331 peak_vram=16470MiB peak_job_rss=2958MiB peak_wsl_used=21464MiB
2026-11-25 16:20:23 G2_LC_DONE
2026-11-25 16:22:44 start probe_prefix (job RAM cap 14336 MiB)
2026-11-25 16:23:39 end probe_prefix exit=0 seconds=55 peak_vram=832MiB peak_job_rss=836MiB peak_wsl_used=1452MiB
2026-11-25 16:27:08 G3 start at ccd0016 fix(kv): run KIVI cached-prefix chunks through the native tiled paged kernel (WIP)
2026-11-25 16:27:10 start tests_kivi_fix (job RAM cap 14336 MiB)
2026-11-25 16:27:22 end tests_kivi_fix exit=0 seconds=12 peak_vram=860MiB peak_job_rss=1407MiB peak_wsl_used=2002MiB
59 passed, 2 skipped in 10.99s
2026-11-25 16:27:25 G2 start at ccd0016 fix(kv): run KIVI cached-prefix chunks through the native tiled paged kernel (WIP), STEPS=8
2026-11-25 16:27:25 start lc64_auto_mrp8 (job RAM cap 14336 MiB)
2026-11-25 16:30:44 end lc64_auto_mrp8 exit=0 seconds=199 peak_vram=16126MiB peak_job_rss=3702MiB peak_wsl_used=3949MiB
2026-11-25 16:30:44 start lc64_kivi4_mrp8 (job RAM cap 14336 MiB)
2026-11-25 16:33:31 end lc64_kivi4_mrp8 exit=0 seconds=167 peak_vram=9538MiB peak_job_rss=3667MiB peak_wsl_used=3870MiB
2026-11-25 16:33:31 start lc64_kivi2_mrp8 (job RAM cap 14336 MiB)
2026-11-25 16:36:11 end lc64_kivi2_mrp8 exit=0 seconds=160 peak_vram=8306MiB peak_job_rss=3935MiB peak_wsl_used=4162MiB
2026-11-25 16:36:11 start lc64_kivi4_mrp24 (job RAM cap 14336 MiB)
2026-11-25 16:38:40 end lc64_kivi4_mrp24 exit=0 seconds=149 peak_vram=16310MiB peak_job_rss=3429MiB peak_wsl_used=3675MiB
2026-11-25 16:38:40 start lc64_kivi2_mrp40 (job RAM cap 14336 MiB)
2026-11-25 16:41:06 end lc64_kivi2_mrp40 exit=0 seconds=146 peak_vram=16468MiB peak_job_rss=3854MiB peak_wsl_used=4095MiB
2026-11-25 16:41:06 G2_LC_DONE
2026-11-25 16:41:09 G2 start at ccd0016 fix(kv): run KIVI cached-prefix chunks through the native tiled paged kernel (WIP), STEPS=8
2026-11-25 16:41:09 start gsm8k512_auto (job RAM cap 14336 MiB)
2026-11-25 16:46:17 end gsm8k512_auto exit=0 seconds=308 peak_vram=13686MiB peak_job_rss=4062MiB peak_wsl_used=4251MiB
2026-11-25 16:46:17 start gsm8k512_kivi4 (job RAM cap 14336 MiB)
2026-11-25 16:51:34 end gsm8k512_kivi4 exit=0 seconds=317 peak_vram=8404MiB peak_job_rss=4044MiB peak_wsl_used=4263MiB
2026-11-25 16:51:34 start gsm8k512_kivi2 (job RAM cap 14336 MiB)
2026-11-25 16:56:53 end gsm8k512_kivi2 exit=0 seconds=319 peak_vram=7284MiB peak_job_rss=3998MiB peak_wsl_used=4205MiB
      1 KIVI KV cache: dtype=kivi2 group_size=32 residual_length=128 blocks=1761 slots=41 bytes=10446700544
      9 KIVI KV cache: dtype=kivi2 group_size=32 residual_length=128 blocks=321 slots=33 bytes=2372665344
      1 KIVI KV cache: dtype=kivi2 group_size=32 residual_length=128 blocks=353 slots=9 bytes=2108424192
      8 KIVI KV cache: dtype=kivi2 group_size=32 residual_length=512 blocks=321 slots=33 bytes=3825991680
      1 KIVI KV cache: dtype=kivi4 group_size=32 residual_length=128 blocks=1057 slots=25 bytes=10156769280
     13 KIVI KV cache: dtype=kivi4 group_size=32 residual_length=128 blocks=321 slots=33 bytes=3550740480
      1 KIVI KV cache: dtype=kivi4 group_size=32 residual_length=128 blocks=353 slots=9 bytes=3403939840
2026-11-25 16:56:53 G2_GSM8K_DONE
2026-11-25 16:56:53 G3_DONE
2026-11-25 16:58:11 rl_s1_kivi2: files of the first attempt (pid 889952, 13:49-13:59, ran without the guard after the runner crash) moved to rl_s1_kivi2/unguarded_first_attempt; results use the guarded rerun only
