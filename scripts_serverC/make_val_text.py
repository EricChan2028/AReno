"""Concatenate GSM8K test questions and reference solutions into one text file."""

import sys

from datasets import load_from_disk

rows = load_from_disk(sys.argv[1])
with open(sys.argv[2], "w") as handle:
    for row in rows:
        handle.write(f"Question: {row['question']}\nAnswer: {row['answer']}\n\n")
print(len(rows), "rows")
