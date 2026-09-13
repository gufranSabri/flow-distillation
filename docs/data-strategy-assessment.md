# Data-Strategy Assessment — Barebones Word-Level KD

Assessment of why the distilled student underperforms the base model, and how this
repo's training/eval data strategy compares to the reference implementations in
`docs/repos/LMOps/minillm` (MiniLLM) and `docs/repos/distillm` (DistiLLM).

Run assessed: `work_dir/test` (eval) / `~/scratch/distillation/test` (training),
Qwen2.5-3B-Instruct → Qwen2.5-1.5B-Instruct, LoRA, `configs/word_level.yaml`.

---

## 1. What the numbers say

The benchmark comparison is clean — same eval config in both `loglikelihood.json`
files (14042 MMLU docs, 0-shot, `apply_chat_template` on, bf16, batch 32), and
`work_dir/test` really is the merged LoRA checkpoint (`word_level_final`,
1,543,714,304 params, identical to base). So the regression is real, not an eval
artifact:

| model | MMLU acc |
|---|---|
| `Qwen2.5-1.5B-Instruct` (base) | 0.5817 ± 0.0039 |
| distilled (`word_level_final`) | 0.5668 ± 0.0040 |

44 of 61 subtasks down, 12 up, 5 unchanged; mean per-subtask delta **−1.9pt**.
The erosion is uniform rather than a domain trade-off:

| category | base | distilled | delta |
|---|---|---|---|
| stem | 0.4967 | 0.4684 | −2.8 |
| social sciences | 0.6926 | 0.6682 | −2.4 |
| other | 0.6479 | 0.6312 | −1.7 |
| humanities | 0.5224 | 0.5239 | +0.2 |

**The training log is the more important result** (`~/scratch/distillation/test/metrics.jsonl`):

```
step 200: val loss 0.3935  ce 0.5845  kd 0.2025  agreement 0.8632
step 313: val loss 0.3916  ce 0.5844  kd 0.1988  agreement 0.8641
```

Train loss is flat 0.45 → 0.39 across all 313 steps, and teacher/student top-1
agreement moves **+0.09pp over the entire run**. The student learned nothing from
the teacher; it only drifted. No signal in, damage out.

---

## 2. Structural diff vs the reference repos

Compared against `distillm/scripts/openllama2/distillm/train_3B_7B_teacher_lora.sh`
(3B student / 7B teacher / LoRA — the closest analogue to this setup) and
`LMOps/minillm/scripts/qwen2.5/{sft,minillm}/`.

| stage | MiniLLM / DistiLLM | this repo |
|---|---|---|
| task data D | dolly-15k, ~12.5k train | smoltalk 10k, 5 subsets |
| **teacher** | SFT'd **on D first** (`xlarge-sft`, `7B-sft`, `TEACHER_PEFT_CKPT`) | `Qwen2.5-3B-Instruct` off the shelf |
| **student init** | SFT'd **on D first** — distillm has a dedicated `scripts/gpt2/init/` stage; minillm uses `1.5B-init` | `Qwen2.5-1.5B-Instruct` off the shelf |
| aux LM loss | OpenWebText via `--lm-data-dir`, present in **every** ≥1B run (`opt/kd`, `openllama2/distillm`) | none |
| epochs | 10–20 | 1 |
| checkpoint selection | best **val ROUGE-L** (val loss for the init stage) | last step |
| LR at ~1.5B | 5e-6 (`qwen2.5/minillm`), 1e-5 (`qwen2.5/sft`) | **1e-4** |
| **eval** | held-out split **of D** + 4 sibling instruction sets, generative, ROUGE-L, 5 seeds | MMLU 0-shot loglikelihood — no relation to D |

`kd_ratio` semantics are identical to ours (`(1-r)*lm_loss + r*distil_loss`,
`distillm/finetune.py:337`, `LMOps/minillm/finetune.py:272`), so `KD_LAMBDA: 0.5`
is defensible — though the ≥1B reference runs use 1.0 (pure KD).

---

## 3. Diagnosis

The "train on smoltalk, test on the others" suspicion is right but points slightly
off-target. The train/test mismatch is a real deficiency, but it isn't *why* the
model got worse — it's why we **can't tell** that it got worse for a different
reason.

The root cause is that **there is no headroom to distill**. Both papers manufacture
a teacher that is strictly better than the student *on D* by fine-tuning it on D,
and start the student from something weak (raw GPT-2, or an `init` SFT checkpoint).
We do neither: Qwen already post-trained both models on data that dwarfs
smoltalk-10k, and neither has seen smoltalk. **86.3% top-1 agreement before
training** is the measurement of that — forward KL between two near-identical
distributions produces a gradient that is mostly noise.

Meanwhile the other half of the loss (`KD_LAMBDA 0.5` → CE on smoltalk ground
truth) is plain SFT on 10k math/code/CoT samples at **LR 1e-4 on LoRA across all
seven projection modules** — 10–20× above what both repos use at this scale. That
is the term with enough gradient magnitude to actually move weights, and what it
moves is Qwen's post-training, downward.

This is very likely the same failure mode as the earlier flow-matching attempts.
It would have been invisible in all of them, because the repo has no
in-distribution metric that separates "distillation failed" from "distillation was
a no-op plus forgetting."

---

## 4. Recommended changes, in order

1. **Add an in-distribution eval before anything else.** A held-out smoltalk
   generative eval (ROUGE-L, or at minimum val loss + teacher agreement on a fixed
   set) is the primary signal; MMLU is the *regression guard*, not the objective.
   Currently we fly with only the guard. `val_ds` is already built — this is cheap.
2. **Create the headroom.** Either
   (a) SFT `Qwen2.5-3B-Instruct` on smoltalk to make a real teacher and SFT the
   1.5B as an `init` (the distillm recipe), or
   (b) switch the student to `Qwen2.5-1.5B` **base** so the teacher has something
   to teach. (b) is far cheaper and yields a setup where KD can only help. Note the
   base model has no chat template, so `prepare_tokenizer` will raise — copy the
   instruct template over.
3. **Drop LR to 1e-5 or below**, run more than one epoch, checkpoint per epoch, and
   select on the in-distribution metric. 313 steps at 1e-4 with a flat loss is the
   worst of both worlds.
4. **Add the LM regularizer** if the student stays instruct-tuned. Both repos use an
   OpenWebText loss at ≥1B precisely to stop the MMLU-style erosion seen here. A
   general-corpus or general-chat slice mixed into the CE term would do it.

---

## 5. Smaller code-level findings

- `utils/data.py:76` — the response is encoded as raw `content` and terminated with
  a manually appended `eos_token_id`. Correct for Qwen only because eos ==
  `<|im_end|>`; silently breaks for any model whose turn terminator isn't eos. Fix
  before going cross-model.
- `utils/data.py:79` — length overflow uses `break`, discarding *all* later turns of
  a conversation rather than just the long one. `continue` is probably intended.
- Multi-turn expansion turns each n-turn conversation into n overlapping examples,
  so "10000 train examples" is far fewer than 10000 distinct conversations. This is
  also why the `⚠ subset smaller than target` warning on `numina-cot-100k` is
  spurious.
- `MAX_VAL_SAMPLES: 5000` against 10000 train, evaluated at batch size 1 — a large
  fraction of the run was spent on two val passes. 500 would carry the same
  information.
- `configs/common.yaml` `LR: 3.0e-4` is dead (overridden by `word_level.yaml`
  `LR: 1.0e-4`); an easy thing to misread while tuning.
- `SmolTalkProcessor.MAGPIE_SUBSET` is unused — the category filter works only
  because non-magpie subsets lack a `category` field.

**Verified correct:** the label pre-shift in `line2data` is consistent with
`ce_loss` and `loss_mask` in `src/trainers/word_level.py` — alignment was checked
explicitly, there is no off-by-one.
