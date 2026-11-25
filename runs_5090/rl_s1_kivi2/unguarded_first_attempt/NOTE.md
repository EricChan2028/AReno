# Unguarded first attempt of rl_s1_kivi2 (not used as evidence)

- Process 889952 started at 13:49:12 (2026-11-25, UTC+8) when the G2 runner exited on a bug in its
  memory-guard helper, and completed all 8 GSPO steps at 13:59:17 without the guard (no VRAM, job-RSS
  or WSL-total enforcement, no per-run memory trace).
- The reported KIVI-2 result uses only the guarded rerun (14:00:28-14:10:25, process 914369) in ../tb.
  Its per-step values reproduce this attempt closely (train/logp_abs_diff_mean step 0: 0.1980 vs
  0.1980, step 1: 0.2029 vs 0.2034, step 2: 0.1803 vs 0.1812), kept here only as a repeat-run check.
- WSL total memory during the window, from the memory log of another job that ran on this machine at
  the time (a VSAG build; `free -m` used, about every 20 s): 31 samples between 13:49:12 and 13:59:17, maximum
  17,977 MB (13:57:12), minimum 3,344 MB (13:49:28); just outside the window 2,054 MB (13:49:09) and
  811 MB (13:59:26). That job's own guard (stop above 34 GB WSL total) stayed active and did
  not trigger.
- The guard helper was fixed (always one integer; guard readings sanitized) before the rerun.
