"""Single-process SONU training: K sequential worker minibatches per round.

Each of the K workers differentiates the same weights on its own minibatch;
only the two low-dimensional factor gradients are averaged, as in Eq. (1). The
workers are simulated sequentially in one process, so this measures the
optimizer, **not** wall-clock throughput or real network communication.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import sys
import torch

from sonu import SONU, SONUConfig, attach_sketches, merge_sketches
from sonu.data import BatchStream, collate, load_tokenized, smoke_data


def learning_rate(step, total_steps, peak, minimum):
    """Cosine schedule with no warmup, read at t = step - 1."""
    return (
        minimum
        + (peak - minimum) * (1 + math.cos(math.pi * (step - 1) / total_steps)) / 2
    )


def collect_sketches(model, layers, stream, workers, pad_id, device, accumulate):
    """Eq. (1)/(17): average the K workers' two model-dtype factor gradients.

    Each worker differentiates the same weights on its own minibatch; only the
    two low-dimensional factor gradients are ever combined. The sum is
    accumulated at ``accumulate`` (the optimizer's solve dtype) so that
    averaging K=64 BF16 sketches does not drop the small ones.
    """
    model.train()
    totals = {
        name: (
            torch.zeros_like(layer.B, dtype=accumulate),
            torch.zeros_like(layer.A, dtype=accumulate),
        )
        for name, layer in layers.items()
    }
    loss_sum = 0.0
    for _ in range(workers):
        model.zero_grad(set_to_none=True)
        batch = collate(stream.next(), pad_id, device)
        loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        loss_sum += loss.item()
        for name, layer in layers.items():
            if layer.B.grad is None or layer.A.grad is None:
                raise RuntimeError(f"Target layer was not used in forward: {name}")
            totals[name][0].add_(layer.B.grad)
            totals[name][1].add_(layer.A.grad)
    gradients = {name: (B / workers, A / workers) for name, (B, A) in totals.items()}
    model.zero_grad(set_to_none=True)
    return gradients, loss_sum / workers


@torch.no_grad()
def evaluate(model, dataset, batch_size, pad_id, device):
    model.eval()
    token_loss = example_loss = tokens = examples = 0
    for start in range(0, len(dataset), batch_size):
        rows = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        batch = collate(rows, pad_id, device)
        loss = model(**batch).loss.item()
        if not math.isfinite(loss):
            raise FloatingPointError("Non-finite evaluation loss")
        count = int((batch["labels"][:, 1:] != -100).sum())
        token_loss += loss * count
        example_loss += loss * len(rows)
        tokens += count
        examples += len(rows)
    mean = token_loss / tokens
    return dict(
        eval_loss=mean,
        perplexity=math.exp(mean) if mean < 709 else float("inf"),
        eval_loss_example_weighted=example_loss / examples,
        perplexity_example_weighted=(
            math.exp(example_loss / examples)
            if example_loss / examples < 709
            else float("inf")
        ),
        eval_tokens=tokens,
        eval_examples=examples,
    )


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config", type=Path, help="JSON configuration; CLI flags override it"
    )
    p.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    p.add_argument("--revision", default=None, help="Optional model/tokenizer revision")
    p.add_argument("--train-data", type=str)
    p.add_argument("--eval-data", type=str)
    p.add_argument("--output-dir", type=Path, default=Path("runs/sonu"))
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument(
        "--workers", type=int, default=64, help="Simulated workers per global step"
    )
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--evolution-rate", type=float, default=10.0)
    p.add_argument("--switch-interval", type=int, default=50)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp64"], default="bf16")
    p.add_argument(
        "--optimizer-dtype",
        choices=["fp32", "bf16", "fp64"],
        default=None,
        help="Storage dtype for momenta and ambient residuals; defaults to "
        "--dtype. Linear algebra is always evaluated at FP32 or above.",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument(
        "--eval-samples", type=int, default=128, help="Prefix size; 0 uses full split"
    )
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument(
        "--save-every", type=int, default=0, help="0 saves only the final checkpoint"
    )
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument(
        "--stop-after",
        type=int,
        help="Stop early without changing the LR schedule horizon",
    )
    p.add_argument(
        "--resume", type=Path, help="Trusted checkpoint produced by this trainer"
    )
    p.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--export-merged", action="store_true")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Offline tiny Llama, synthetic data; explicit small settings required",
    )
    return p


def parse_args(argv=None):
    p = parser()
    preliminary, _ = p.parse_known_args(argv)
    if preliminary.config:
        config = json.loads(preliminary.config.read_text())
        unknown = set(config) - {action.dest for action in p._actions}
        if unknown:
            p.error(f"Unknown JSON settings: {sorted(unknown)}")
        p.set_defaults(**config)
    args = p.parse_args(argv)
    args.output_dir = Path(args.output_dir)
    if args.resume:
        args.resume = Path(args.resume)
    for key in ("steps", "workers", "batch_size", "rank", "max_length", "log_every"):
        if getattr(args, key) <= 0:
            p.error(f"{key} must be positive")
    if args.eval_samples < 0 or args.eval_every < 0 or args.save_every < 0:
        p.error("eval_samples, eval_every, save_every must be nonnegative")
    if (
        not math.isfinite(args.lr)
        or not math.isfinite(args.lr_min)
        or not 0 <= args.lr_min <= args.lr
    ):
        p.error("Require finite 0 <= lr_min <= lr")
    if args.stop_after is not None and not 1 <= args.stop_after <= args.steps:
        p.error("stop_after must be within [1, steps]")
    if not args.smoke and (not args.train_data or not args.eval_data):
        p.error("--train-data and --eval-data are required outside --smoke")
    return args


def serializable_args(args):
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def rng_state():
    return dict(
        python=random.getstate(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def restore_rng(saved):
    random.setstate(saved["python"])
    torch.set_rng_state(saved["torch"])
    if saved["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(saved["cuda"])


def save_checkpoint(path, model, optimizer, stream, args, fingerprints):
    temporary = path.with_suffix(".tmp")
    torch.save(
        dict(
            format_version=1,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            stream=stream.state_dict(),
            rng=rng_state(),
            args=serializable_args(args),
            fingerprints=fingerprints,
        ),
        temporary,
    )
    temporary.replace(path)


def main(argv=None):
    args = parse_args(argv)
    # Imported lazily: --help and the optimizer do not require Transformers.
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        LlamaConfig,
        LlamaForCausalLM,
    )

    saved = (
        torch.load(args.resume, map_location="cpu", weights_only=False)
        if args.resume
        else None
    )
    if saved:
        if saved["format_version"] != 1:
            raise ValueError("Unsupported checkpoint format")
        mutable = {
            "config",
            "output_dir",
            "resume",
            "device",
            "eval_every",
            "save_every",
            "log_every",
            "stop_after",
            "export_merged",
        }
        now = serializable_args(args)
        changed = [
            key for key in now if key not in mutable and now[key] != saved["args"][key]
        ]
        if changed:
            raise ValueError(f"Resume settings changed: {changed}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp64": torch.float64}[
        args.dtype
    ]
    tokenizer = None
    if args.smoke:
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=97,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=128,
                attention_dropout=0.0,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
            )
        ).to(dtype=dtype)
        training, validation = smoke_data(), smoke_data(count=8)
        fingerprints, pad_id = {"train": "smoke-v1", "eval": "smoke-v1"}, 0
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer needs a padding or EOS token")
        pad_id = tokenizer.pad_token_id
        if saved:
            from transformers import AutoConfig

            model = AutoModelForCausalLM.from_config(
                AutoConfig.from_pretrained(args.model, revision=args.revision),
                torch_dtype=dtype,
                attn_implementation="sdpa",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                revision=args.revision,
                torch_dtype=dtype,
                attn_implementation="sdpa",
            )
        training, train_hash = load_tokenized(args.train_data, args.max_length)
        validation, eval_hash = load_tokenized(args.eval_data, args.max_length)
        fingerprints = {"train": train_hash, "eval": eval_hash}
    if args.eval_samples:
        count = min(args.eval_samples, len(validation))
        validation = (
            validation.select(range(count))
            if hasattr(validation, "select")
            else validation[:count]
        )
    if saved and saved["fingerprints"] != fingerprints:
        raise ValueError("Dataset fingerprints changed since checkpoint")
    model.to(args.device)
    model.config.use_cache = False
    layers = attach_sketches(
        model, args.rank, args.gamma, initialize=saved is None
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    optimizer = SONU(
        layers,
        SONUConfig(
            args.momentum,
            args.evolution_rate,
            args.switch_interval,
            args.optimizer_dtype or args.dtype,
        ),
    )
    stream = BatchStream(training, args.batch_size, args.seed)
    if saved:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        stream.load_state_dict(saved["stream"])
        restore_rng(saved["rng"])
        del saved
    metadata = dict(
        arguments=serializable_args(args),
        optimizer=asdict(optimizer.config),
        fingerprints=fingerprints,
        train_samples=len(training),
        eval_samples=len(validation),
        execution="single-process sequential worker simulation",
        warmup_steps=0,
        versions={
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "datasets")
        },
        python=platform.python_version(),
        layers=list(layers),
        model_revision=getattr(model.config, "_commit_hash", None),
        optimizer_state_dtype=optimizer.config.dtype,
        optimizer_solve_dtype=str(optimizer.config.solve),
        server_residual_bytes_when_both_allocated=sum(
            2 * optimizer.config.store.itemsize * l.out_features * l.in_features
            for l in layers.values()
        ),
        sketch_elements_per_worker_per_direction=sum(
            l.A.numel() + l.B.numel() for l in layers.values()
        ),
    )
    (args.output_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if tokenizer:
        tokenizer.save_pretrained(args.output_dir / "tokenizer")

    def log(record):
        print(json.dumps(record), flush=True)
        with (args.output_dir / "metrics.jsonl").open("a") as file:
            file.write(json.dumps(record) + "\n")

    if optimizer.steps == 0:
        log(
            dict(
                step=0,
                **evaluate(model, validation, args.batch_size, pad_id, args.device),
            )
        )
    limit = args.stop_after or args.steps
    for step in range(optimizer.steps + 1, limit + 1):
        gradients, loss = collect_sketches(
            model,
            layers,
            stream,
            args.workers,
            pad_id,
            args.device,
            optimizer.config.solve,
        )
        lr = learning_rate(step, args.steps, args.lr, args.lr_min)
        optimizer.step(lr, gradients)
        record = dict(step=step, train_loss=loss, lr=lr)
        if (args.eval_every and step % args.eval_every == 0) or step == limit:
            record.update(
                evaluate(model, validation, args.batch_size, pad_id, args.device)
            )
        if step % args.log_every == 0 or "eval_loss" in record:
            log(record)
        if args.save_every and step % args.save_every == 0:
            save_checkpoint(
                args.output_dir / f"checkpoint-{step}.pt",
                model,
                optimizer,
                stream,
                args,
                fingerprints,
            )
    save_checkpoint(
        args.output_dir / "checkpoint.pt", model, optimizer, stream, args, fingerprints
    )
    if args.export_merged:
        merge_sketches(model).save_pretrained(args.output_dir / "merged_model")
        if tokenizer:
            tokenizer.save_pretrained(args.output_dir / "merged_model")
    return 0


if __name__ == "__main__":
    sys.exit(main())
