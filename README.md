# LLM finetuning and distillation

Knowledge-distillation and finetuning pipeline for causal LMs on the Dolly instruction
dataset. It trains models three ways:

- **`finetune.py`** — plain supervised finetuning (SFT) of a single model on Dolly.
  Used to (a) produce baselines and (b) produce the SFT checkpoints that `distill.py`
  treats as student/teacher.
- **`distill.py`** — knowledge distillation of a student against a teacher, matching
  next-token distributions (KD-only, no ground-truth CE loss).
- **`benchmark.py`** — runs the MiniLLM instruction-following eval suite (Dolly,
  Self-Instruct, Vicuna, S-NI, Un-NI) against any checkpoint or hub model and reports
  exact-match / ROUGE-L / mean LM loss (optionally a GPT-4 pairwise judge).

## How it's organized

```
finetune.py / distill.py / benchmark.py   entrypoints (CLI)
configs/
  common.yaml                             shared defaults (dataset, seed, WORK_DIR_ROOT, ...)
  finetune/<approach>.yaml                per-approach finetune config (e.g. vanilla.yaml)
  distill/<approach>.yaml                 per-approach distill config (e.g. word_level.yaml)
src/
  finetune/<approach>/                    model.py + trainer.py trio (vanilla)
  distill/<approach>/                     model.py + trainer.py trio (word_level)
utils/
  losses.py                               loss functions shared by every trio (ce_loss, kd_loss, ...)
  data.py                                 Dolly loading/tokenization, shared by train + eval
  logger.py, utils.py                     logging, RNG seeding, CLI-override parsing, LossAverager
dolly_eval/                               MiniLLM eval-suite loaders + generation/scoring
*.slurm                                   Slurm job scripts (one per entrypoint)
scripts/install.sh                        pip installs used by the .slurm scripts
```

Both `finetune.py` and `distill.py` follow the same pattern: they load
`configs/common.yaml`, merge in `configs/<finetune|distill>/<approach>.yaml`, apply any
`--KEY VALUE` CLI overrides, then dispatch to a **trio** of `load_model` /
`save_model` / `Trainer` looked up by `--approach` name from `src/<finetune|distill>/`.
Adding a new training approach means adding a new trio + config; the entrypoints
themselves don't need to change.

`distill.py`'s student is loaded via the **finetune** trio named by its
`FINETUNE_MODE`/`--approach`-equivalent config (currently `word_level` just reuses
`src/finetune/vanilla`'s `load_model`/`save_model`, since KD needs no student-specific
loading beyond generic HF+LoRA). The teacher is always loaded as a plain
`AutoModelForCausalLM`, frozen, independent of any trio.

## Usage

### 0. Setup

```bash
python -m venv .venv && source .venv/bin/activate   # or your preferred env manager
bash scripts/install.sh
```

`scripts/install.sh` is a flat list of `pip install`s (see the file for the exact set —
torch, transformers/peft from git, tiktoken/sentencepiece, rouge/sacrebleu for eval,
etc.). It's shared verbatim by every `.slurm` script and by manual/interactive setup.

Set `HF_HOME` if you want the HF cache somewhere other than `~/.cache/huggingface`
(the `.slurm` scripts point it at scratch space — see below).

### Without Slurm (interactive / local GPU box)

```bash
# 1. SFT baselines — also produces the checkpoints distill.py consumes as student/teacher
python finetune.py --model Qwen/Qwen2.5-0.5B --approach vanilla --work-dir finetuned/Qwen2.5-0.5B
python finetune.py --model Qwen/Qwen2.5-3B   --approach vanilla --work-dir finetuned/Qwen2.5-3B

# 2. Distill the small SFT checkpoint against the large one
python distill.py \
    --student-model ~/scratch/distillation/finetuned/Qwen2.5-0.5B/vanilla_final \
    --teacher-model ~/scratch/distillation/finetuned/Qwen2.5-3B/vanilla_final \
    --approach word_level --work-dir word_level_run

# 3. Benchmark any checkpoint or hub model
python benchmark.py --model ~/scratch/distillation/word_level_run/word_level_final \
    --work-dir ./work_dir/word_level_run
```

Any key in `configs/common.yaml` or the chosen `configs/<finetune|distill>/<approach>.yaml`
can be overridden from the CLI, e.g.:

```bash
python distill.py --student-model ... --teacher-model ... \
    --DISTILL_LOSS both --TEMPERATURE 2.0 --TRAIN_EPOCHS 3
```

`--work-dir` is resolved under `WORK_DIR_ROOT` (`configs/common.yaml`, default
`~/scratch/distillation`) unless you pass an absolute/explicit path — this is why the
examples above pass a short name like `finetuned/Qwen2.5-0.5B` and then reference it
back out as `~/scratch/distillation/finetuned/Qwen2.5-0.5B/vanilla_final`.
`benchmark.py --work-dir` is not resolved this way — it's used as-is (see
`benchmark.slurm`, which points it at `./work_dir/...` in the repo itself).

Useful one-liners while iterating are collected in `scripts/troubleshooting.sh`
(smoke tests, tailing logs, watching `metrics.jsonl`, checking a checkpoint reloads
cleanly, OOM knobs, etc.) — read it rather than re-deriving these from scratch.

### With Slurm

Each entrypoint has a matching `.slurm` script (`finetune.slurm`, `distill.slurm`,
`benchmark.slurm`). Submit from the repo root:

```bash
sbatch finetune.slurm
sbatch distill.slurm [run-name]
sbatch benchmark.slurm
```

Each script loads the cluster modules, builds a fresh venv in `$SLURM_TMPDIR`, runs
`scripts/install.sh`, sets `HF_HOME` to scratch space, and then calls the same Python
entrypoints shown above. Edit the `MODELS=(...)` / `STUDENT_MODEL_ID` / `TEACHER_MODEL_ID`
variables at the top of each script to change which models are run — they're the only
things you typically need to touch.

Logs land in `slurm/<job-name>.<job-id>.{out,err}`; per-run training logs and
`metrics.jsonl` land under `WORK_DIR_ROOT/<work-dir>/` on scratch.

## Extending: adding a new model

No code changes needed — `--model` / `--student-model` / `--teacher-model` accept any
HF hub id or local path. The only requirement for `word_level` distillation
specifically is that student and teacher **share a vocabulary** (`distill.py` checks
this and raises if they don't, since word-level KD compares per-token distributions
directly).

## Extending: adding a new approach (finetune or distill)

Both `src/finetune/` and `src/distill/` follow the same trio convention. To add a new
approach (say a new finetuning method `src/finetune/my_approach/`):

1. **Create the package** `src/finetune/my_approach/` with:
   - `model.py` — `load_model(args, model_id)` and `save_model(model, tokenizer, save_dir)`.
     `load_model` should return a model ready for `.to(args.device)` + training;
     `save_model` should write out something `AutoModelForCausalLM.from_pretrained`
     can reload standalone (see `src/finetune/vanilla/model.py` for the LoRA
     merge-before-save pattern).
   - `trainer.py` — a `Trainer` class (name it whatever, re-export it as `Trainer` in
     `__init__.py`) with the signature the entrypoint calls it with:
     - finetune: `Trainer(args, model, tokenizer, train_ds, val_ds, collator)`
     - distill: `Trainer(args, student, teacher, tokenizer, train_ds, val_ds, collator)`
     It needs a `.train()` method. Look at `src/finetune/vanilla/trainer.py` or
     `src/distill/word_level/trainer.py` as a template — the boilerplate (DataLoader
     setup, optimizer/scheduler, grad accumulation, checkpointing with
     `SAVE_TOTAL_LIMIT` pruning, `metrics.jsonl`/console logging via `LossAverager`,
     tqdm progress bar) is copy-paste between them; only `_forward` really differs.
   - `__init__.py`:
     ```python
     from .model import load_model, save_model
     from .trainer import MyApproachTrainer as Trainer

     __all__ = ["load_model", "save_model", "Trainer"]
     ```

   No `losses.py`: every trio composes its loss from `utils/losses.py` instead of
   defining its own. `_forward` should return `(loss, parts, ...)` where `parts` is a
   `{name: loss_tensor}` dict and `loss = sum(parts.values())` — finetune's `parts`
   should always include `"ce"` (`utils.losses.ce_loss`, the main loss every finetune
   trio uses), distill's should always include `"kd"` (`utils.losses.kd_loss`, the main
   loss every distill trio uses). If your approach needs more than that (a second loss
   term on top of the main one), just add another key to `parts` — the trainer sums it
   into the backward loss and both the console and `metrics.jsonl` log every component
   automatically. Only add a new function to `utils/losses.py` if the loss you need
   genuinely isn't there yet.

2. **Register it** in `src/finetune/__init__.py` (or `src/distill/__init__.py`):
   ```python
   from . import my_approach

   APPROACHES = {
       "vanilla": vanilla,
       "my_approach": my_approach,
   }
   ```

3. **Add a config** `configs/finetune/my_approach.yaml` (or `configs/distill/...`) with
   whatever hyperparameters your trainer reads off `args` (it's merged on top of
   `configs/common.yaml`, so you only need to declare keys common.yaml doesn't already
   set, or override ones it does).

4. Run it via `--approach my_approach`.

If you want the new approach's student/teacher loading to differ from an existing
trio, write your own `model.py`; if it doesn't (as `src/distill/word_level` doesn't),
just import and re-export another trio's `load_model`/`save_model` like
`src/distill/word_level/model.py` does. Likewise, if your approach needs a reusable
model component (not just a loss), put it in `utils/` rather than inside the trio's
own package — `src/<finetune|distill>/<approach>/` should stay the thin
`model.py`/`trainer.py` glue that wires shared pieces from `utils/` into a training loop.

## Data pipeline

`utils/data.py`'s `build_dolly_datasets` is shared by `finetune.py`, `distill.py`,
and (via `format_dolly_prompt`/`load_dolly_splits`) `dolly_eval`, so training and eval
prompts can't drift apart. It reproduces MiniLLM's Dolly recipe exactly: the first
`DOLLY_DEV_NUM` raw rows (natural order) become the fixed validation split, the rest
are train; prompts use MiniLLM's Alpaca template (`format_dolly_prompt`); labels are
pre-shifted (`labels[t]` targets `logits[t]`, `-100` over the prompt). Run
`python utils/data.py` to sanity-check the pipeline against a decoded sample.

`DOLLY_DEV_NUM` in `configs/common.yaml` must match `benchmark.py --dolly-dev-num`
(default 1000 in both) or the Dolly eval task will leak train examples into eval.
