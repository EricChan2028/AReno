"""Collect the raw outputs behind the KIVI report into one browsable directory.

Token-level dumps are kept (xz-compressed) for the greedy long-context runs that
compare_rollouts.py reads; other rollout dumps are reduced to one line per
completion (lengths, reward, mean log-probability). Checkpoints are left out.
"""

import json
import lzma
import shutil
import subprocess
import sys
from pathlib import Path

RUNS = Path("/root/kivi/runs")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/kivi/report/kivi-590-raw")
# Further arguments name run directories to leave out (runs still in progress).
SKIP_DIRS = {"ckpt", "__pycache__", *sys.argv[2:]}


def copy_tree(src: Path, dst: Path, rollouts: str) -> None:
    for path in sorted(src.rglob("*")):
        rel = path.relative_to(src)
        if any(part in SKIP_DIRS for part in rel.parts) or not path.is_file():
            continue
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.name == "rollouts.jsonl":
            if rollouts == "full":
                with path.open("rb") as handle, lzma.open(target.with_suffix(".jsonl.xz"), "wb") as out:
                    shutil.copyfileobj(handle, out)
            else:
                with path.open() as handle, (target.parent / "completions.jsonl").open("w") as out:
                    for line in handle:
                        row = json.loads(line)
                        logprobs = row.get("resp_logprobs") or []
                        out.write(
                            json.dumps(
                                {
                                    "index": row["index"],
                                    "sample": row["sample"],
                                    "prompt_tokens": row["prompt_tokens"],
                                    "response_tokens": len(row["resp_tokens"]),
                                    "reward": row["reward"],
                                    "mean_logprob": sum(logprobs) / len(logprobs) if logprobs else None,
                                }
                            )
                            + "\n"
                        )
            continue
        if path.stat().st_size > 50 * 2**20:
            print("skipping large file", path)
            continue
        shutil.copy2(path, target)


if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)
for name, rollouts in (
    ("rl_final", "reduced"),
    ("validation", "reduced"),
    ("eval_sampled", "reduced"),
    ("eval_sampled_17b", "reduced"),
    ("throughput", "reduced"),
    ("long_context", "full"),
    ("teacher_forced", "reduced"),
    ("bench_final", "reduced"),
    ("ci_emulation", "reduced"),
    ("from_server1", "reduced"),
):
    if (RUNS / name).exists():
        copy_tree(RUNS / name, OUT / "runs" / name, rollouts)
for source in (Path("/root/kivi/scripts"), Path("/root/kivi/scripts_serverC")):
    if source.exists():
        dst = OUT / source.name
        dst.mkdir(parents=True, exist_ok=True)
        for path in sorted(source.glob("*")):
            if path.suffix in {".sh", ".py"}:
                shutil.copy2(path, dst / path.name)
shutil.copy2("/root/kivi/report/tables.md", OUT / "tables.md")
shutil.copy2("/root/kivi/report/artifacts_README.md", OUT / "README.md")
env = OUT / "env"
env.mkdir()
for filename, command in (
    ("pip_freeze.txt", "bash -c 'source /root/kivi/env.sh && pip freeze'"),
    ("nvidia_smi.txt", "nvidia-smi"),
    ("os_release.txt", "cat /etc/os-release"),
    ("nvcc.txt", "nvcc --version"),
    (
        "sha256.txt",
        "bash -c 'cd /root/kivi && sha256sum data/wikitext2_raw_test.parquet data/wikitext2_test.txt "
        "data/gsm8k_test_solutions.txt data/gsm8k_main_train/data-00000-of-00001.arrow "
        "data/gsm8k_main_test/data-00000-of-00001.arrow data/gsm8k_test512_ctx8k/data-00000-of-00001.arrow "
        "models/Qwen3-0.6B/model.safetensors models/Qwen3-1.7B/*.safetensors models/*/.mv'",
    ),
):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    (env / filename).write_text(result.stdout + result.stderr)
size = sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file())
print(f"{OUT}: {sum(1 for _ in OUT.rglob('*'))} entries, {size / 2**20:.1f} MiB")
