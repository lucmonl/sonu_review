"""Tokenize the three datasets of Section 3.1 with explicit selection rules.

Chain-of-thought (MegaScience), agentic function calling (APIGen) and
instruction following (OASST2). Every selection and length filter is an
explicit flag or a documented constant; nothing is capped silently. Pin
--dataset-revision and --revision to make a build reproducible.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from sonu.prompts import (
    _build_prompt_and_response,
    instruction_prompt_and_response,
    tokenize_pair,
)

DATASETS = {
    "megascience": "MegaScience/MegaScience",
    "func_call": "Salesforce/xlam-function-calling-60k",
    "oasst2": "OpenAssistant/oasst2",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=list(DATASETS), required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--revision", help="Tokenizer revision")
    p.add_argument(
        "--dataset-revision", help="Dataset revision; pin this for a release"
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--language", default="en")
    p.add_argument(
        "--oasst-selection",
        choices=["labeled", "rank_zero"],
        help="Required for OASST2; see README. labeled: any human-labeled reply. "
        "rank_zero: additionally require the best-ranked sibling.",
    )
    p.add_argument(
        "--limit", type=int, help="Explicit raw example cap for development only"
    )
    args = p.parse_args()
    if args.max_length < 2 or (args.limit is not None and args.limit <= 0):
        p.error("max_length must be >=2 and limit must be positive")
    if args.dataset == "oasst2" and args.oasst_selection is None:
        p.error(
            "--oasst-selection is required: the manuscript says highly ranked "
            "responses without giving a threshold, so the policy must be explicit"
        )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    stats = {}

    def encode(rows, cap, formatter):
        encoded = []
        for row in rows:
            prompt, response = formatter(row)
            item = tokenize_pair(prompt, response, tokenizer, cap)
            if any(label != -100 for label in item["labels"][1:]):
                encoded.append(item)
        return Dataset.from_list(encoded) if encoded else None

    if args.dataset == "oasst2":
        raw = load_dataset(DATASETS[args.dataset], revision=args.dataset_revision)
        splits = {}
        for name in ("train", "validation"):
            messages = [m for m in raw[name] if m["lang"] == args.language]
            lookup = {m["message_id"]: m for m in messages}
            rows = []
            for message in messages:
                parent = lookup.get(message["parent_id"])
                if message["role"] != "assistant" or message.get("labels") is None:
                    continue
                if parent is None or parent["role"] != "prompter":
                    continue
                if args.oasst_selection == "rank_zero" and message.get("rank") != 0:
                    continue
                rows.append(dict(instruction=parent["text"], response=message["text"]))
            if args.limit:
                rows = rows[: args.limit]
            splits[name] = encode(
                rows, args.max_length, instruction_prompt_and_response
            )
            stats[name] = {"selected_pairs": len(rows)}
    else:
        raw = load_dataset(
            DATASETS[args.dataset], split="train", revision=args.dataset_revision
        )
        stats["raw_fingerprint"] = raw._fingerprint
        if args.limit:
            raw = raw.select(range(min(args.limit, len(raw))))
        rows = []
        for example in raw:
            if args.dataset == "megascience":
                row = dict(instruction=example["question"], response=example["answer"])
                # Filter on raw instruction+response length, before chat markers.
                size = sum(
                    len(tokenizer.encode(row[k], add_special_tokens=False))
                    for k in ("instruction", "response")
                )
                keep = size <= 2048
            else:
                row = {k: example[k] for k in ("query", "tools", "answers")}
                prompt, response = _build_prompt_and_response(row)
                keep = (
                    len(tokenizer.encode(prompt, add_special_tokens=False))
                    + len(tokenizer.encode(response, add_special_tokens=False))
                ) <= args.max_length
            if keep:
                rows.append(row)
        stats["after_raw_filter"] = len(rows)
        if len(rows) < 20:
            raise ValueError(
                "Fewer than 20 examples remain; cannot make useful 90/5/5 splits"
            )
        temporary = Dataset.from_list(rows).train_test_split(
            test_size=0.05, seed=args.seed
        )
        train_val = temporary["train"].train_test_split(
            test_size=0.05 / 0.95, seed=args.seed
        )
        raw_splits = {
            "train": train_val["train"],
            "validation": train_val["test"],
            "test": temporary["test"],
        }
        formatter = (
            instruction_prompt_and_response
            if args.dataset == "megascience"
            else _build_prompt_and_response
        )
        # MegaScience tokenizes at 2048, then filters splits by final length.
        cap = 2048 if args.dataset == "megascience" else args.max_length
        splits = {
            name: encode(rows, cap, formatter) for name, rows in raw_splits.items()
        }
        if args.dataset == "megascience":
            splits = {
                name: (
                    ds.filter(lambda row: len(row["input_ids"]) <= args.max_length)
                    if ds is not None
                    else None
                )
                for name, ds in splits.items()
            }
    if any(ds is None or len(ds) == 0 for ds in splits.values()):
        raise ValueError("A split is empty after filtering; no output was written")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, ds in splits.items():
        ds.save_to_disk(str(args.output_dir / name))
        stats.setdefault(name, {})["saved_examples"] = len(ds)
        stats[name]["fingerprint"] = ds._fingerprint
    metadata = dict(
        arguments={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        dataset_id=DATASETS[args.dataset],
        statistics=stats,
        versions={
            k: importlib.metadata.version(k) for k in ("datasets", "transformers")
        },
    )
    (args.output_dir / "preprocessing.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
