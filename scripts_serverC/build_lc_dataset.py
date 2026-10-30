"""GSM8K test problems, each preceded by a distinct ~8K-token WikiText-2 passage.

Rows already use the trainer schema (`prompt`, `solutions`), so the math
dataset loader passes them through unchanged.
"""

import argparse

from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--tokenizer", required=True)
parser.add_argument("--gsm8k", required=True)
parser.add_argument("--text", required=True)
parser.add_argument("--context", type=int, default=8000)
parser.add_argument("--limit", type=int, default=512)
parser.add_argument("--output", required=True)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
with open(args.text) as handle:
    ids = tokenizer(handle.read(), add_special_tokens=False)["input_ids"]
span = len(ids) - args.context
rows = []
for index, record in enumerate(load_from_disk(args.gsm8k).select(range(args.limit))):
    start = (index * 7919) % span
    passage = tokenizer.decode(ids[start : start + args.context])
    final = str(record["answer"]).rsplit("####", 1)[-1].strip()
    prompt = (
        f"{passage}\n\n"
        "Solve the following grade-school math problem. Show your reasoning "
        "and put the final answer in \boxed{}.\n\n"
        f"Problem: {record['question']}\nSolution:"
    )
    rows.append({"prompt": prompt, "solutions": [final], "solution": final, "passage_start": start})
Dataset.from_list(rows).save_to_disk(args.output)
lengths = [len(tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]) for row in rows]
print(f"{len(rows)} rows, wikitext tokens {len(ids)}, prompt tokens min {min(lengths)} max {max(lengths)}")
