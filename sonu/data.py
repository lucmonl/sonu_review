"""Tokenized data, response-only padding, and checkpointable batch sampling."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import torch


COLUMNS = ("input_ids", "attention_mask", "labels")


def load_tokenized(path, max_length):
    """Read JSONL/JSON rows or one Hugging Face save_to_disk split directory."""
    path = Path(path)
    if path.is_dir():
        from datasets import load_from_disk

        dataset = load_from_disk(str(path))
        if not hasattr(dataset, "column_names") or isinstance(
            dataset.column_names, dict
        ):
            raise ValueError("Pass one saved dataset split, not a DatasetDict")
        dataset = dataset.select_columns(list(COLUMNS))
        fingerprint = dataset._fingerprint
    elif path.suffix == ".jsonl":
        dataset = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    elif path.suffix == ".json":
        dataset = json.loads(path.read_text())
        fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        raise ValueError(f"Expected a saved dataset directory, JSON, or JSONL: {path}")
    if len(dataset) == 0:
        raise ValueError(f"Empty dataset: {path}")
    # Full validation is intentional: no silent truncation or all-masked losses.
    for index, row in enumerate(dataset):
        validate_row(row, max_length, f"{path}, row {index}")
    return dataset, fingerprint


def validate_row(row, max_length, location="sample"):
    if any(key not in row for key in COLUMNS):
        raise ValueError(f"{location}: required columns are {COLUMNS}")
    n = len(row["input_ids"])
    if not 2 <= n <= max_length or any(len(row[key]) != n for key in COLUMNS):
        raise ValueError(f"{location}: invalid lengths; max_length={max_length}")
    if any(not isinstance(x, int) or x < 0 for x in row["input_ids"]):
        raise ValueError(f"{location}: input_ids must be nonnegative integers")
    if any(x != 1 for x in row["attention_mask"]):
        raise ValueError(f"{location}: store unpadded sequences with attention_mask=1")
    if any(
        label != -100 and label != token
        for label, token in zip(row["labels"], row["input_ids"])
    ):
        raise ValueError(f"{location}: labels must be input_ids or -100")
    if not any(x != -100 for x in row["labels"][1:]):
        raise ValueError(f"{location}: no supervised next-token targets")


def collate(rows, pad_token_id, device="cpu"):
    width = (max(len(x["input_ids"]) for x in rows) + 7) // 8 * 8
    batch = {
        "input_ids": torch.full((len(rows), width), pad_token_id, dtype=torch.long),
        "attention_mask": torch.zeros((len(rows), width), dtype=torch.long),
        "labels": torch.full((len(rows), width), -100, dtype=torch.long),
    }
    for i, row in enumerate(rows):
        n = len(row["input_ids"])
        for key in COLUMNS:
            batch[key][i, :n] = torch.tensor(row[key], dtype=torch.long)
    return {key: value.to(device) for key, value in batch.items()}


class BatchStream:
    """A shared shuffled stream, as in the source single-process worker simulator."""

    def __init__(self, dataset, batch_size, seed):
        if len(dataset) == 0 or batch_size <= 0:
            raise ValueError("Nonempty data and positive batch_size are required")
        self.dataset, self.batch_size = dataset, batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(len(dataset), generator=self.generator)
        self.position, self.epoch = 0, 0

    def next(self):
        if self.position == len(self.dataset):
            self.order = torch.randperm(len(self.dataset), generator=self.generator)
            self.position, self.epoch = 0, self.epoch + 1
        indices = self.order[self.position : self.position + self.batch_size].tolist()
        self.position += len(indices)
        return [self.dataset[i] for i in indices]

    def state_dict(self):
        return dict(
            order=self.order.clone(),
            position=self.position,
            epoch=self.epoch,
            generator=self.generator.get_state(),
            size=len(self.dataset),
            batch_size=self.batch_size,
        )

    def load_state_dict(self, saved):
        if saved["size"] != len(self.dataset) or saved["batch_size"] != self.batch_size:
            raise ValueError("Checkpoint dataset size or batch size changed")
        self.order, self.position, self.epoch = (
            saved["order"],
            saved["position"],
            saved["epoch"],
        )
        self.generator.set_state(saved["generator"])


def smoke_data(count=24, length=24, vocab_size=97):
    generator = torch.Generator().manual_seed(1729)
    rows = []
    for _ in range(count):
        ids = torch.randint(3, vocab_size, (length,), generator=generator).tolist()
        rows.append(
            dict(
                input_ids=ids, attention_mask=[1] * length, labels=[-100] * 4 + ids[4:]
            )
        )
    return rows
