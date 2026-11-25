# Section 9: re-run on the current main (RTX 5090 Laptop GPU, 24 GB)

Raw outputs behind section 9 of `examples/kivi/README.md`. Every run directory
holds the command (`cmd.txt`), the commit of the tree it ran (`commit.txt`), the
device memory before the run (`mem_before.txt`), a 1 Hz trace
(`mem_trace.txt`: unix time, device MiB, host MiB of the job's processes, WSL
MiB in use) and its peaks (`peak_*.txt`). Runs were kept below 20 GB of device
memory and 14 GiB of host memory per job (16 GiB for the KIVI-4 GSPO run and
its resume) and stopped if WSL as a whole used more than 34 GB; no run hit a
limit.

## Commits

The runs used local commits that were later squashed into the pull request
commit. Their trees relate as follows (`git diff --stat`):

| Recorded commit | Base | Difference to the next row |
|---|---|---|
| 1ec2a7a (CPU suite in `ci_cpu/`) | `main` bc3ee6f | ed72d98 adds only upstream #697 (`areno/accel/kernels/nf4_moe.py`, `tests/test_qlora.py`) |
| ed72d98 (tests, benchmark, teacher-forced, GSPO, `before_fix/`, `probe_prefix/`) | `main` d9667ad | ccd0016 changes only `areno/engine/layers/attention_backend/kivi.py` (cached-prefix chunks through the paged tiled kernel) and its CUDA test |
| ccd0016 (`tests_kivi_fix/`, `lc64_*/`, `gsm8k512_*/`) | `main` d9667ad | 4a1794f adds upstream #698 and the section 9 documentation |
| 4a1794f (`tests_kivi_final/`) | `main` 5f7a976 | the pull request commit differs from it only in `examples/kivi/README.md`; code identical |

`areno/accel/kernels/kivi.py` is identical in all of them; the KIVI backend of
ccd0016, 4a1794f and the pull request commit is identical. `tests_upstream_gpu_main/`
ran in a checkout of `main` d9667ad (see its `cmd.txt`); its `commit.txt`
names the PR tree of the same machine.

## Directories

| Directory | Section 9 part | Contents |
|---|---|---|
| `env/` | setup | Python and library versions, `nvidia-smi`, `pip freeze`, sha256 of the model and data files, environment setup log |
| `extension_source.txt` | setup | origin and sha256 of the AReno CUDA extension (sm_120, unchanged `areno/accel/csrc` and `setup.py`) |
| `tests_kivi/`, `tests_kivi_fix/`, `tests_kivi_final/` | tests | KIVI CPU and CUDA tests on ed72d98, ccd0016 and the pull request commit |
| `tests_upstream_gpu/` | tests | first run of the upstream GPU tests, before flash-linear-attention (in `requirements/default.txt`) was installed: 2 GatedDeltaNet tests fail on the missing package |
| `tests_upstream_gpu_pr/`, `tests_upstream_gpu_main/` | tests | the same tests with the package, with and without this change |
| `ci_cpu/` | tests | `pytest tests/ -k cpu` with triton, flash-attn and CUDA hidden, in chunks of six files per process; `single_process_*` are the first attempts, stopped by the 4 GiB host-memory cap at about 80% |
| `tf_len16384/` | teacher-forced | perplexity, KL and top-1 agreement per position range |
| `bench_graph/`, `bench_graph_16k/`, `bench_graph_32k/`, `bench_graph_128x16k_kivi/`, `bench_eager/` | decode benchmark | per-configuration JSON and logs |
| `bench_graph_native_fault_128x16k/` | decode benchmark | the first benchmark run, which faulted in the native BF16 decode kernel at 128 rows x 16K tokens (32-bit cache offsets); its log holds every configuration up to that point |
| `rl_s1_*/`, `rl_summary.*` | GSPO, save/resume | `areno train` logs and TensorBoard events (`tb/`); checkpoints are not included, `rl_s1_kivi4/step_*.sha256` list their files |
| `rl_s1_kivi2/unguarded_first_attempt/` | none | a first KIVI-2 run that ran without the memory guard after the runner exited on a guard bug; not used, see its `NOTE.md` |
| `probe_prefix/` | prefill fix | one-layer timing of `areno_causal_attention` with a query offset against the paged tiled kernel for chunks after a cached prefix |
| `before_fix/lc64_*/` | prefill fix | the long-context runs before the fix (the 24- and 40-sequence runs stopped at the 40-minute limit) |
| `lc64_*/` | long context | rollout summaries and token dumps (`rollouts.jsonl.xz`) after the fix |
| `gsm8k512_*/` | GSM8K | rollout summaries and token dumps |
| `scripts/` | all | the drivers that launched these runs and `tables_5090.py`, which produced `tables.md` |
| `g.log` | all | the run log with start and end time, exit code and memory peaks of every run |
