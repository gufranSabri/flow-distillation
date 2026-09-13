# Switch training data to Dolly (MiniLLM-style) and replace lm-evaluation-harness with a MiniLLM-style Dolly eval

## Context

The repo currently trains on SmolTalk (`utils/data.py::SmolTalkProcessor`, chat-template formatted) and evaluates checkpoints via a vendored `lm-evaluation-harness/` directory called from `benchmark.py` (MMLU loglikelihood only right now). The goal is to match `docs/repos/minillm`'s exact setup for this project's purposes: train on Dolly using their Alpaca-style prompt format, and replace the lm-evaluation-harness dependency with a small vendored-equivalent of MiniLLM's own generation+ROUGE-L/EM eval loop on Dolly, wired into `benchmark.py` so its CLI/output contract stays as close to unchanged as possible.

Decisions already confirmed with the user:
- Dolly **replaces** SmolTalk as the default training dataset (SmolTalk code stays in `utils/data.py`, just no longer wired into `main.py`).
- Use MiniLLM's **generic Alpaca template** with **plain `tokenizer.encode`, no `apply_chat_template`** — for both training data and eval generation — even though the target models (Qwen) are chat-tuned. This is a deliberate exact-replication choice.
- Delete the vendored `lm-evaluation-harness/` directory entirely.

Key research finding: `MiniLLM/dolly` on the HF Hub is not `datasets.load_dataset`-compatible (its `raw.jsonl` isn't picked up by the generic builder). `databricks/databricks-dolly-15k` (15011 rows, `instruction`/`context`/`response`/`category` fields, single `train` split, unshuffled) was verified row-for-row against MiniLLM's own `valid.jsonl` row 0 — they match. So: source Dolly via `datasets.load_dataset("databricks/databricks-dolly-15k", split="train")` and replicate MiniLLM's split (**first `dev_num` rows of the natural/unshuffled order → valid, rest → train** — no shuffling before the split, matching `tools/process_data_dolly.py`).

## 1. Data pipeline — `utils/data.py`

Add, near `SmolTalkProcessor`, without touching existing SmolTalk code:

```python
DOLLY_DATASET_ID      = "databricks/databricks-dolly-15k"
DOLLY_DEV_NUM_DEFAULT = 1000   # MiniLLM: first N raw rows -> valid, rest -> train
DOLLY_MAX_PROMPT_LEN  = 256    # MiniLLM tools/process_data_dolly.py --max-prompt-length

_DOLLY_TEMPLATE_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)
_DOLLY_TEMPLATE_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)

def format_dolly_prompt(instruction: str, input_text: str) -> str:
    """MiniLLM's generic (non-qwen2) Alpaca template — used by both DollyProcessor
    and dolly_eval so training and eval prompts never drift apart. Deliberately no
    chat template, matching MiniLLM's own base-model recipe exactly."""
    if not input_text:
        return _DOLLY_TEMPLATE_NO_INPUT.format(instruction=instruction)
    return _DOLLY_TEMPLATE_WITH_INPUT.format(instruction=instruction, input=input_text)


def load_dolly_splits(dataset_id=DOLLY_DATASET_ID, dev_num=DOLLY_DEV_NUM_DEFAULT):
    """Replicates tools/process_data_dolly.py's split on the natural row order:
    first `dev_num` rows -> valid, rest -> train. No shuffling before slicing."""
    full = load_dataset(dataset_id, split="train")
    return full.select(range(dev_num, len(full))), full.select(range(dev_num))  # train, valid
```

`DollyProcessor(DataProcessor)` — **important: must preserve the same input_ids/labels length invariant as `SmolTalkProcessor`** (`len(labels) == len(input_ids)`, pre-shifted so `labels[t] == input_ids[t+1]`, achieved by dropping `prompt_ids[1:]` for the mask prefix and appending exactly one `eos_token_id` to labels only — never to `input_ids`). Do **not** pre-append eos before slicing `full_ids` (MiniLLM does this because their own on-disk format and loader are different from this repo's); doing so leaves `response_ids` ending in eos, which — if then also appended to labels — double-counts eos, and if not re-appended — leaves `labels` one token *shorter* than `input_ids`, breaking the collator's fixed length relationship used everywhere else in this repo. The correct, SmolTalk-consistent formula:

```python
class DollyProcessor(DataProcessor):
    """Alpaca-template, single-turn instruction/response pairs (MiniLLM's Dolly recipe).
    Unlike SmolTalkProcessor, never calls apply_chat_template — deliberate exact
    replication of MiniLLM's own training data despite Qwen being chat-tuned."""

    def line2data(self, indexed_example: tuple) -> list:
        _, ex = indexed_example
        prompt = format_dolly_prompt(ex["instruction"], ex["context"])

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_ids) > DOLLY_MAX_PROMPT_LEN:
            return []

        full_ids = self.tokenizer.encode(prompt + ex["response"], add_special_tokens=False)
        response_ids = full_ids[len(prompt_ids):]   # no eos yet — matches SmolTalk's response_ids

        if len(prompt_ids) + len(response_ids) + 1 > self.config.MAX_LENGTH:
            return []

        # Same pre-shifted-labels convention as SmolTalkProcessor.line2data.
        input_ids = prompt_ids + response_ids
        labels    = [-100] * len(prompt_ids[1:]) + response_ids + [self.tokenizer.eos_token_id]

        return [{"input_ids": input_ids, "labels": labels}]
```

`prepare_tokenizer(tokenizer, logger=None, require_chat_template=True)` — add the new kwarg; only raise the "no chat_template" error when `require_chat_template` is True. `eos_token_id`/`pad_token` checks stay unconditional (Dolly still needs both).

New `build_dolly_datasets(args, tokenizer)` — a **separate, simpler sibling function**, not a generic dispatch inside `build_datasets` (which stays untouched for SmolTalk):

```python
def build_dolly_datasets(args, tokenizer):
    args.logger("Building Dolly datasets...")
    tokenizer = prepare_tokenizer(tokenizer, logger=args.logger, require_chat_template=False)

    train_raw, val_raw = load_dolly_splits(
        dataset_id=getattr(args, "DATASET_ID", DOLLY_DATASET_ID),
        dev_num=getattr(args, "DOLLY_DEV_NUM", DOLLY_DEV_NUM_DEFAULT),
    )
    processor = DollyProcessor(config=args, tokenizer=tokenizer, filepath=args.DATASET_ID)

    def process(raw_dataset, label):
        out = []
        for item in tqdm(enumerate(raw_dataset), desc=f"Tokenizing Dolly {label}", total=len(raw_dataset)):
            out.extend(processor.line2data(item))
        return out

    train_data, val_data = process(train_raw, "train"), process(val_raw, "val")

    if args.MAX_TRAIN_SAMPLES != -1:
        random.shuffle(train_data); train_data = train_data[:args.MAX_TRAIN_SAMPLES]
    if args.MAX_VAL_SAMPLES != -1:
        random.shuffle(val_data); val_data = val_data[:args.MAX_VAL_SAMPLES]

    args.logger(f"  Total after length filter (≤{args.MAX_LENGTH} tokens): "
                f"{len(train_data)} train / {len(val_data)} val\n")

    train_tokenized, val_tokenized = Dataset.from_list(train_data), Dataset.from_list(val_data)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=None, padding=True,
                                            pad_to_multiple_of=8, label_pad_token_id=-100)
    return train_tokenized, val_tokenized, data_collator
```

`main.py`: swap `from utils.data import build_datasets` → `build_dolly_datasets`, and the call site `build_datasets(args, tokenizer)` → `build_dolly_datasets(args, tokenizer)`. `SmolTalkProcessor`/`build_datasets` remain in `utils/data.py` untouched for future reuse.

`utils/data.py`'s `if __name__ == "__main__":` inspection CLI: swap its call to `build_dolly_datasets(...)` too, so `python utils/data.py --config configs/word_level.yaml` becomes the Dolly smoke test (already the documented pattern in `docs/common_instructions.md`/`scripts/troubleshooting.sh`).

## 2. New eval folder `dolly_eval/` (replaces `lm-evaluation-harness/`)

Flat layout matching the repo's existing style:

- `dolly_eval/__init__.py` — re-exports `run_generation_eval`.
- `dolly_eval/rouge_metric.py` — trimmed copy of `docs/repos/minillm/rouge_metric.py`: `normalize_answer`, `exact_match`, `rouge`, `metric_max_over_ground_truths`, `compute_metrics` (Google `rouge_score`'s `rougeL.fmeasure` + SQuAD-style exact match). Drop `compute_grouped_metrics` and its `argparse`/`__main__` CLI (unused here). `rouge_score` is already a `scripts/install.sh` dependency — no new pip install needed.
- `dolly_eval/generate.py` — the one function `benchmark.py` calls:

```python
def run_generation_eval(model, tokenizer, args) -> dict:
    """MiniLLM's evaluate_main.py Dolly generation eval, simplified to a single-process
    batch loop — no torchrun/DistributedSampler/all_gather, matching this repo's
    already-single-process benchmark.py. Returns {"dolly": {"<metric>,none": value, ...}}
    so it plugs straight into benchmark.py's existing summarize() unchanged."""
```

Internals:
1. Reuse `utils.data.load_dolly_splits(dataset_id=args.dolly_dataset_id, dev_num=args.dolly_dev_num)` and take the **valid** half; build prompts with `utils.data.format_dolly_prompt`, applying the same `--dolly-max-prompt-length` drop rule as training. Truncate to `args.limit` docs if given — this is the `--limit`-equivalent fast-iteration path for the new eval.
2. Batch in plain slices of `args.batch_size` (now a required int, see below); left-pad via `tokenizer.pad(...)` for `model.generate`.
3. Build a `GenerationConfig(do_sample, top_k, top_p, temperature, no_repeat_ngram_size, repetition_penalty, eos_token_id, pad_token_id)` from the new `--dolly-*` CLI flags (defaults below, taken from `docs/repos/minillm/arguments.py` + `scripts/qwen2.5/eval/eval_main_dolly.sh`); `max_new_tokens = args.dolly_max_length - prompt_len`.
4. Decode responses (`skip_special_tokens=True`), pair each with its reference `response`, call `dolly_eval.rouge_metric.compute_metrics`.
5. Compute `mean_lm_loss` via a simple per-example teacher-forced forward pass (prompt+true-response tokens, loss masked to response-only tokens) — kept per-example (not batched-and-padded like MiniLLM) since Dolly's eval set is small (≤1000 examples) and per-example is far simpler to get right.
6. Write `{work_dir}/dolly_samples.jsonl` (prompt/prediction/reference per example) — the lightweight replacement for MiniLLM's `preds.txt`/`preds.pt`/`answers.jsonl`/`log.txt`.
7. Write `{work_dir}/dolly.json` with the raw metrics dict (mirrors the old `{label}.json` that `run_tasks` used to write).
8. Return `{"dolly": {"exact_match,none": ..., "rougeL,none": ..., "mean_lm_loss,none": ...}}` — the `,none` suffix means `benchmark.py`'s existing `summarize()` parses it with **zero changes** (it splits on `,`, keeps the metric name when the filter is `"none"`).

## 3. `benchmark.py` edits

Remove: `LOGLIKELIHOOD_TASKS`/`GENERATIVE_TASKS` constants, `run_tasks()` (and its `import lm_eval`/`HFLM`), `_tasks()` helper, and the CLI flags `--loglikelihood-tasks`, `--generative-tasks`, `--apply-chat-template`/`--no-chat-template`.

Keep unchanged: `load_model()`, `summarize()`, and the `--model`/`--work-dir`/`--device`/`--dtype`/`--limit` flags/behavior.

Change: `--batch-size` from `default="auto"` (an `lm_eval` HFLM-only feature with no equivalent here) to `type=int, default=16` (MiniLLM's `EVAL_BATCH_SIZE`).

Add Dolly generation-eval CLI flags (defaults from MiniLLM's Dolly eval scripts):
```
--dolly-dataset-id            default "databricks/databricks-dolly-15k"
--dolly-dev-num       (int)   default 1000   # must match training's DOLLY_DEV_NUM to avoid leakage
--dolly-max-length    (int)   default 512
--dolly-max-prompt-length (int) default 256
--dolly-do-sample / --dolly-no-sample   default True
--dolly-top-k         (int)   default 0
--dolly-top-p         (float) default 1.0
--dolly-temperature   (float) default 1.0
--dolly-no-repeat-ngram-size (int) default 6
--dolly-repetition-penalty (float) default None
```

New `main()`:
```python
def main(args):
    os.makedirs(args.work_dir, exist_ok=True)
    from dolly_eval.generate import run_generation_eval

    dtype = getattr(torch, args.dtype)
    model, tokenizer = load_model(args.model, args.device, dtype)
    print(f"Loaded {args.model} ({sum(p.numel() for p in model.parameters()):,} params)")

    results = run_generation_eval(model, tokenizer, args)
    table = summarize(results, args.work_dir)
    print(f"\n{table}\n\nResults written to {args.work_dir}")
```

Net effect: `--model`/`--work-dir` and the `summary.txt`/`summary.json` output contract are unchanged; the task-selection menu is retired in favor of a single hardcoded Dolly benchmark (adding a `--tasks` flag for exactly one task would be premature abstraction).

## 4. Config changes — `configs/common.yaml`

Replace:
```yaml
DATASET_ID: "HuggingFaceTB/smoltalk"
DATASET_SUBSETS: [ ... ]
```
with:
```yaml
DATASET_ID: "databricks/databricks-dolly-15k"
DOLLY_DEV_NUM: 1000   # first N raw rows held out as the fixed valid split (MiniLLM convention);
                       # benchmark.py's --dolly-dev-num must match this to avoid train/eval leakage
```
Change `MAX_TRAIN_SAMPLES`/`MAX_VAL_SAMPLES` from `100000`/`5000` to `-1`/`-1` — Dolly only has ~14011 train/1000 valid rows (one example per row, no multi-turn expansion like SmolTalk), so the old caps are silent no-ops; `-1` makes "use everything" explicit. Keep `MAX_LENGTH: 2048` as-is.

`configs/word_level.yaml` needs no changes (KD hyperparams are dataset-agnostic).

Generation-eval hyperparameters live as `benchmark.py` CLI flags, not YAML, because `benchmark.py` has no `--config`/YAML mechanism today and is meant to benchmark arbitrary checkpoints/hub ids, not just ones trained via `main.py` — threading eval knobs through the training config system would be new scope.

## 5. `scripts/install.sh`

Remove:
```bash
cd lm-evaluation-harness && pip install -e .
pip install "lm_eval[hf]"
```
`rouge_score`/`nltk` are already installed earlier in the script — no new dependency needed.

## 6. Delete `lm-evaluation-harness/` and clean up references

- Delete `/project/6101771/ahmedubc/distillation/lm-evaluation-harness/` entirely.
- `docs/common_instructions.md`: reword the line mentioning "benchmark using lm-evaluation-harness" to reference `benchmark.py` instead.
- `scripts/troubleshooting.sh`, `scripts/train_sample.sh`, `benchmark.slurm`: none name `lm-evaluation-harness`/`lm_eval` directly, but they demonstrate the now-removed CLI surface — update:
  - Drop `export HF_ALLOW_CODE_EVAL=1` (was only for humaneval/mbpp, which no longer exist).
  - Replace `--loglikelihood-tasks .../--generative-tasks .../--apply-chat-template` example invocations with the new Dolly-flag form, e.g. `python benchmark.py --model <ckpt> --limit 20 --work-dir ./work_dir/bm_smoke`.

## Verification

1. **Data smoke test** (existing inspection-CLI pattern):
   ```bash
   python utils/data.py --config configs/word_level.yaml
   ```
   Confirm: non-empty train/val, decoded sample shows the Alpaca `### Instruction:` / `### Response:` template with **no** chat-template special tokens, and the decoded label text matches the true `response` field.

2. **Split correctness** (one-off check while implementing):
   ```python
   from utils.data import load_dolly_splits
   train, valid = load_dolly_splits(dev_num=1000)
   assert len(valid) == 1000 and len(train) + len(valid) == 15011
   assert valid[0]["instruction"] == "When did Virgin Australia start operating?"
   ```

3. **Eval smoke test**:
   ```bash
   python benchmark.py --model Qwen/Qwen2.5-1.5B-Instruct --limit 5 --work-dir ./work_dir/bm_smoke
   cat ./work_dir/bm_smoke/summary.txt
   cat ./work_dir/bm_smoke/dolly_samples.jsonl
   ```
   Confirm generations look like plausible Dolly responses (no leftover chat-template artifacts) and `summary.txt`/`summary.json` show `exact_match`/`rougeL`/`mean_lm_loss`.

4. **End-to-end**: `python main.py --work-dir smoke --config configs/word_level.yaml` then `python benchmark.py --model ~/scratch/distillation/smoke/word_level_final --limit 20 --work-dir ./work_dir/smoke_bm`, same shape as `scripts/train_sample.sh`.

### Critical files
- `utils/data.py` — `DollyProcessor`, `load_dolly_splits`, `format_dolly_prompt`, `build_dolly_datasets`, `prepare_tokenizer` kwarg
- `main.py` — swap `build_datasets` → `build_dolly_datasets`
- `benchmark.py` — remove lm_eval integration, add `dolly_eval` call + new CLI flags
- `configs/common.yaml` — dataset id/split config
- `scripts/install.sh` — drop lm-evaluation-harness install
- New: `dolly_eval/__init__.py`, `dolly_eval/generate.py`, `dolly_eval/rouge_metric.py`
- Delete: `lm-evaluation-harness/`

