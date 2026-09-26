# SONU: Sketched Orthonormalized Updates

Reference implementation of **SONU** and its language-model training path, for
the paper *SONU: Communication-Efficient Orthonormalized Updates via Sketching*.

SONU builds a Muon-style orthonormalized update from two-sided gradient
sketches. Each worker communicates only the two low-dimensional factor
gradients of Eq. (1); the server forms a generalized Nyström surrogate, extracts
the paired singular vectors without any dense orthonormalization, and evolves
the sketching subspaces along the optimization trajectory. The dense gradient
is never materialized on a worker, and no dense momentum is kept there.

This directory is self-contained. It depends on PyTorch, plus Transformers and
Datasets for the training entry point; PEFT is **not** required.

**[docs/ALGORITHM.md](docs/ALGORITHM.md) maps every equation in the paper to the
code that implements it**, and records the points where the implementation
makes a choice the paper leaves open. Read it before reviewing the optimizer.

> **Note on the training schedule.** There is **no warmup**. Table 1 of the
> submitted manuscript lists 100 linear warmup steps; that row is an error and
> is being corrected. The learning rate is a plain cosine decay from `lr` to
> `lr_min`, and `tests/test_training.py` pins it against
> `torch.optim.lr_scheduler.CosineAnnealingLR`.

## Install and run

Python 3.10+ (tested on 3.11). Install a PyTorch build suited to your machine,
then from this directory:

```bash
python -m pip install -e '.[train,test]'
python -m pytest -q
python train.py --config configs/smoke.json --output-dir runs/smoke
```

The smoke run builds a tiny Llama locally from synthetic tokens. It needs no
model download, credentials, dataset, or GPU, and it finishes in seconds. It
checks that the code executes, not that it learns anything.

## Reproducing a paper configuration

`configs/` holds one file per task and model scale, populated from Tables 1
and 3 of the paper:

| Config | Task | Model | `lr` | `alpha` | `S` | `gamma` |
| --- | --- | --- | --- | --- | --- | --- |
| `cot_1b.json` | Chain-of-thought (MegaScience) | Llama-3.2-1B | 5e-3 | 25 | 50 | 1.0 |
| `cot_3b.json` | Chain-of-thought (MegaScience) | Llama-3.2-3B | 5e-3 | 10 | 50 | 1.0 |
| `func_call_1b.json` | Agentic function calling (APIGen) | Llama-3.2-1B | 1e-3 | 50 | 50 | 1.0 |
| `func_call_3b.json` | Agentic function calling (APIGen) | Llama-3.2-3B | 1e-3 | 50 | 50 | 1.0 |
| `oasst2_1b.json` | Instruction following (OASST2) | Llama-3.2-1B | 1e-3 | 50 | 50 | 0.1 |
| `oasst2_3b.json` | Instruction following (OASST2) | Llama-3.2-3B | 1e-3 | 10 | 50 | 1.0 |

All six share `d = 16`, 64 workers, local batch size 2, `beta = 0.9`, cosine
decay to `1e-5`, and no warmup.

```bash
python prepare_data.py --dataset megascience \
  --model meta-llama/Llama-3.2-3B --max-length 1024 --output-dir data/cot

python train.py --config configs/cot_3b.json \
  --train-data data/cot/train --eval-data data/cot/validation \
  --output-dir runs/cot-3b
```

**`steps` is set to 1500 in every config and is not a reproduction of a
specific figure.** The manuscript does not state a per-figure training horizon;
set `--steps` to your own budget. Model access may require Hugging Face
credentials and license acceptance. Pass `--revision` and `--dataset-revision`
to pin versions.

## What the trainer does and does not do

`--workers 64 --batch-size 2` computes 64 local minibatch losses **at the same
weights**, averages the two factor gradients, and takes one global update. The
workers are simulated sequentially in a single process. This measures the
optimizer; it is **not** a multi-process distributed trainer and **not** a
network benchmark. Communication volume is reported analytically in
`config.json`, not timed.

Only `q_proj`, `k_proj`, `v_proj` and `o_proj` are optimized, matching Section
3.1. Every other parameter, including embeddings, norms, MLPs and the output
head, is frozen. No local SGD step, weight decay, gradient clipping, Nesterov
momentum or warmup is added anywhere.

Each run writes resolved settings, package versions, dataset fingerprints and
execution mode to `config.json`.

## Precision

Model weights, sketch factors and communicated sketches use `--dtype` (BF16 by
default). The optimizer's persistent state -- the projected momenta `M^L, M^R`,
the ambient residuals `E^L, E^R` and the sketch snapshots -- is stored at
`--optimizer-dtype`, which follows `--dtype` unless you set it. State storage
dominates server optimizer memory, because each residual is ambient `I x J`.

BF16 cannot carry a pseudo-inverse, so **every value is computed at FP32 or
above and rounded exactly once when stored**. That promotion wraps each
`pinv`/QR/SVD together with the matmuls around it, since the momentum transport
of Eq. (11) and the Nyström core of Eq. (8) are compositions with `(.)^+`. Pass
`--optimizer-dtype fp32` or `fp64` to widen the stored state; the linear
algebra is unaffected.

Relative to an FP64 reference, BF16 state perturbs a single update direction
`U V^T` by `~5e-3`, and that error does not grow with the number of rounds.
Trajectories still separate over many rounds, but FP32 state separates just as
far: that is the polar step amplifying any perturbation, not the storage dtype.
See [docs/ALGORITHM.md](docs/ALGORITHM.md#precision).

## Data preparation

```bash
python prepare_data.py --dataset megascience --model <model> --output-dir data/cot
python prepare_data.py --dataset func_call  --model <model> --output-dir data/func_call
python prepare_data.py --dataset oasst2 --oasst-selection labeled \
  --model <model> --output-dir data/oasst2
```

Point `--train-data` and `--eval-data` at the generated `train/` and
`validation/` directories. Selection counts and fingerprints are written to
`preprocessing.json`. `--limit` is an explicit development-only cap; nothing is
capped silently.

OASST2 **requires** an explicit `--oasst-selection` because the paper says
"responses ranked highly by human contributors" without giving a threshold:

- `labeled` -- every human-labeled assistant reply whose parent is a prompter turn.
- `rank_zero` -- additionally require `rank == 0`, the best-ranked sibling.

Neither is asserted to be the policy behind the published figures; choose one
and record it. MegaScience uses the generated `answer` as the target, not
`reference_answer`, and filters raw instruction-plus-response length at 2048
before tokenizing. APIGen preserves the tool-system prompt and `<tool_call>`
formatting. All three use seed-42 90/5/5 splits and response-only labels.

You may also train on your own tokenized data. Each row needs `input_ids`,
`attention_mask` and response-only `labels` (`-100` on prompt positions), must
be unpadded, at most `--max-length` tokens, and must contain at least one
supervised next-token target. The loader rejects invalid rows rather than
silently repairing them.

## Checkpoints, metrics and export

`checkpoint.pt` holds **the full model**, the optimizer state at its configured
dtype, sampler state and RNG states. Adapter-only checkpoints would be
incomplete, because SONU updates base weights. Resume requires the same
`--optimizer-dtype`; saved state is moved between devices, never retyped.

```bash
python train.py --config configs/smoke.json --output-dir runs/smoke \
  --resume runs/smoke/checkpoint.pt
```

`--stop-after N` interrupts without changing the cosine horizon, so a resumed
run follows the same schedule; the test suite checks that an interrupted run
matches an uninterrupted one bit for bit. Resume rejects changed data
fingerprints or incompatible settings. `--export-merged` writes a standard
Hugging Face model to `merged_model/` with the factors folded into the base
weights; resume from `checkpoint.pt`, not from that export.

`metrics.jsonl` reports response-token-weighted validation cross entropy and
perplexity, plus example-weighted variants under explicit names. These differ
when response lengths vary, so compare like with like. The default validation
subset is the first 128 examples; `--eval-samples 0` uses the whole split.

## Review map

| File | Purpose |
| --- | --- |
| `sonu/optimizer.py` | Momentum transport and residuals (10)-(12), small-core polar update (8), sketch evolution and compensation (14), (21) |
| `sonu/layers.py` | Uniform-SV initialization (13), (18)-(19); both sketches from one backward pass (16)-(17) |
| `train.py` | Worker simulation and averaging (1), cosine schedule, evaluation, checkpoint/resume |
| `prepare_data.py`, `sonu/prompts.py` | Dataset selection and response-only tokenization |
| `sonu/data.py` | Row validation, padding, checkpointable sampling |
| `tests/` | Equation identities, precision contract, training, resume, export |
| `docs/ALGORITHM.md` | Equation-to-code map, open choices, precision |
| `docs/VALIDATION.md` | Exactly what was and was not verified |
