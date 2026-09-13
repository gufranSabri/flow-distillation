import inspect
import json
import math
import os
import shutil
import sys

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from src.models.word_level import save_model
from utils import losses as loss_fns


# Divergences selectable via DISTILL_LOSS. All come from utils/losses.py, which is
# DistiLLM's losses.py verbatim, so they keep its
# (logits, teacher_logits, no_model_batch) signature.
DISTILL_LOSSES = {
    "forward_kl":        loss_fns.forward_kl,
    "reverse_kl":        loss_fns.reverse_kl,
    "symmetric_kl":      loss_fns.symmetric_kl,
    "js_distance":       loss_fns.js_distance,
    "tv_distance":       loss_fns.tv_distance,
    "skewed_forward_kl": loss_fns.skewed_forward_kl,
    "skewed_reverse_kl": loss_fns.skewed_reverse_kl,
}


def resolve_distill_loss(name, lam):
    """Looks up DISTILL_LOSS and validates DISTILL_LOSS_LAM against it, so a bad
    config fails at construction rather than on the first backward pass."""
    if name not in DISTILL_LOSSES:
        raise ValueError(
            f"Unknown DISTILL_LOSS {name!r}; choose one of {sorted(DISTILL_LOSSES)}"
        )
    fn = DISTILL_LOSSES[name]
    if lam is None:
        return fn, {}
    if "lam" not in inspect.signature(fn).parameters:
        raise ValueError(
            f"DISTILL_LOSS_LAM={lam} was set but {name!r} takes no lam; "
            "leave DISTILL_LOSS_LAM null for this loss."
        )
    return fn, {"lam": lam}


def kd_loss(student_logits, teacher_logits, labels, loss_fn, temperature, **kwargs):
    """Divergence between the teacher and student next-token distributions -- the sole
    training objective (labels are only used to mask out the prompt, never as a CE
    target: distillation never sees the Dolly ground-truth response).

    `loss_fn` reads its own mask off no_model_batch["label"], so the raw labels
    tensor is handed over rather than the float mask the other losses take.

    It also reduces with a flat mean over every unmasked token it is given, so it
    is called one sequence at a time and averaged across the batch. At
    PER_DEVICE_TRAIN_BATCH_SIZE=1 this is a single call either way.

    Temperature follows the Hinton convention (soften both sides, rescale by tau^2
    so gradient magnitude stays tau-independent). That rescaling is only strictly
    motivated for the KL family; TEMPERATURE=1.0, the default, makes it a no-op.
    """
    s_logits = student_logits.float() / temperature
    t_logits = teacher_logits.float() / temperature
    per_seq = torch.stack([
        loss_fn(s_logits[i:i + 1], t_logits[i:i + 1], {"label": labels[i:i + 1]}, **kwargs)
        for i in range(s_logits.shape[0])
    ])
    return per_seq.mean() * (temperature ** 2)


class WordLevelTrainer:
    def __init__(self, args, student, teacher, tokenizer, train_ds, val_ds, collator):
        self.args = args
        self.student = student
        self.teacher = teacher
        self.tokenizer = tokenizer
        self.device = args.device
        self.log = args.logger

        self.distill_loss, self.loss_kwargs = resolve_distill_loss(
            args.DISTILL_LOSS, getattr(args, "DISTILL_LOSS_LAM", None),
        )

        self.train_loader = DataLoader(
            train_ds, batch_size=args.PER_DEVICE_TRAIN_BATCH_SIZE,
            shuffle=True, collate_fn=collator, num_workers=2, pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=args.PER_DEVICE_EVAL_BATCH_SIZE,
            shuffle=False, collate_fn=collator, num_workers=2, pin_memory=True,
        )

        accum = args.GRADIENT_ACCUMULATION_STEPS
        self.total_steps = math.ceil(len(self.train_loader) / accum) * args.TRAIN_EPOCHS
        self.trainable = [p for p in student.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            self.trainable, lr=args.LR, weight_decay=args.WEIGHT_DECAY,
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            int(args.WARMUP_RATIO * self.total_steps),
            self.total_steps,
        )

        self.metrics_path = os.path.join(args.work_dir, "metrics.jsonl")
        self.step = 0

    def _log_console(self, message):
        # tqdm.write keeps the message from being overwritten by a bar redraw
        self.log(message)
        tqdm.write(message, file=sys.stdout)

    def _forward(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        attn_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        loss_mask = (labels != -100).float()

        with torch.no_grad():
            teacher_out = self.teacher(input_ids=input_ids, attention_mask=attn_mask)
        student_out = self.student(input_ids=input_ids, attention_mask=attn_mask)

        s_logits, t_logits = student_out.logits, teacher_out.logits
        parts = {
            "kd": kd_loss(
                s_logits, t_logits, labels, self.distill_loss,
                self.args.TEMPERATURE, **self.loss_kwargs,
            ),
        }
        loss = parts["kd"]

        # fraction of positions where student and teacher pick the same top-1 token
        agree = (((s_logits.argmax(-1) == t_logits.argmax(-1)).float() * loss_mask).sum()
                 / loss_mask.sum().clamp(min=1))
        return loss, parts, agree

    def train(self):
        args = self.args
        accum = args.GRADIENT_ACCUMULATION_STEPS
        n_batches = len(self.train_loader)
        spec = ", ".join(
            [f"loss={args.DISTILL_LOSS}"]
            + [f"{k}={v}" for k, v in self.loss_kwargs.items()]
            + [f"mode={args.FINETUNE_MODE}"]
        )
        self.log(f"Training {self.total_steps} steps ({spec})")

        self.student.train()
        self.teacher.eval()
        running = 0.0
        pbar = _bar(total=self.total_steps, desc="train", unit="step", initial=self.step)

        for epoch in range(args.TRAIN_EPOCHS):
            for i, batch in enumerate(self.train_loader):
                loss, parts, _ = self._forward(batch)
                (loss / accum).backward()
                running += loss.item()

                # also step on the last batch, so a partial accumulation window
                # isn't dropped and left to leak into the next epoch
                if (i + 1) % accum != 0 and (i + 1) != n_batches:
                    continue

                torch.nn.utils.clip_grad_norm_(self.trainable, args.MAX_GRAD_NORM)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.step += 1
                pbar.update(1)
                pbar.set_postfix(epoch=epoch, loss=f"{loss.item():.4f}", refresh=False)

                if self.step % args.LOG_EVERY == 0:
                    avg = running / (accum * args.LOG_EVERY)
                    detail = "  ".join(f"{k}={v.item():.4f}" for k, v in parts.items())
                    self._record({"split": "train", "step": self.step, "epoch": epoch,
                                  "loss": avg, "lr": self.scheduler.get_last_lr()[0]})
                    self._log_console(f"[train] step {self.step}/{self.total_steps}  "
                                      f"loss={avg:.4f}  {detail}")
                    running = 0.0

                if self.step % args.EVAL_EVERY == 0:
                    self.evaluate()
                if self.step % args.SAVE_EVERY == 0:
                    self.save_checkpoint()

        pbar.close()
        self.evaluate()
        final_dir = os.path.join(args.work_dir, f"{args.APPROACH}_final")
        save_model(self.student, self.tokenizer, final_dir)
        self.log(f"Saved final model to {final_dir}")

    @torch.no_grad()
    def evaluate(self):
        self.student.eval()
        totals, agrees, n = {}, 0.0, 0

        for batch in _bar(iterable=self.val_loader, desc="eval", unit="batch",
                          leave=False):
            loss, parts, agree = self._forward(batch)
            totals["loss"] = totals.get("loss", 0.0) + loss.item()
            for k, v in parts.items():
                totals[k] = totals.get(k, 0.0) + v.item()
            agrees += agree.item()
            n += 1

        n = max(n, 1)
        row = {"split": "val", "step": self.step, "agreement": agrees / n}
        row.update({k: v / n for k, v in totals.items()})
        self._record(row)
        self._log_console("[val] " + "  ".join(
            f"{k}={v:.4f}" for k, v in row.items() if isinstance(v, float)
        ))

        self.student.train()
        return row

    def save_checkpoint(self):
        ckpt_dir = os.path.join(self.args.work_dir, f"checkpoint-{self.step}")
        save_model(self.student, self.tokenizer, ckpt_dir)
        torch.save(
            {"step": self.step,
             "optimizer": self.optimizer.state_dict(),
             "scheduler": self.scheduler.state_dict()},
            os.path.join(ckpt_dir, "trainer_state.pt"),
        )
        self.log(f"Saved checkpoint to {ckpt_dir}")
        self._prune_checkpoints()

    def _prune_checkpoints(self):
        limit = self.args.SAVE_TOTAL_LIMIT
        if limit <= 0:
            return
        ckpts = sorted(
            (d for d in os.listdir(self.args.work_dir) if d.startswith("checkpoint-")),
            key=lambda d: int(d.split("-")[-1]),
        )
        for stale in ckpts[:-limit]:
            shutil.rmtree(os.path.join(self.args.work_dir, stale), ignore_errors=True)

    def _record(self, row):
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")


def _bar(**kwargs):
    """tqdm configured for both a terminal and a slurm log file: under a redirect
    the bar would otherwise emit a refresh line every fraction of a second, so
    throttle it hard when stdout isn't a tty."""
    interactive = sys.stdout.isatty()
    return tqdm(
        file=sys.stdout, dynamic_ncols=True,
        mininterval=0.5 if interactive else 30.0,
        **kwargs,
    )
