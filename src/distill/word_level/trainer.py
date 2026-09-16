import json
import math
import os
import shutil
import sys

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from .model import save_model
from utils.losses import resolve_distill_loss, kd_loss
from utils.utils import LossAverager, find_latest_checkpoint


class WordLevelTrainer:
    """Main loss is KD (utils.losses.kd_loss): student/teacher next-token distribution
    matching. A distill approach that adds more than KD returns extra parts from its
    own _forward -- everything in `parts` is summed into the step loss and logged,
    main loss included."""

    def __init__(self, args, student, teacher, tokenizer, train_ds, val_ds, collator):
        self.args = args
        self.student = student
        self.teacher = teacher
        self.tokenizer = tokenizer
        self.device = args.device
        self.log = args.logger

        self.distill_loss = resolve_distill_loss(args.DISTILL_LOSS)

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
        self._load_checkpoint_if_exists()

    def _log_console(self, message):
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
            "kd": kd_loss(s_logits, t_logits, labels, self.distill_loss, self.args.TEMPERATURE),
        }
        loss = sum(parts.values())

        agree = (((s_logits.argmax(-1) == t_logits.argmax(-1)).float() * loss_mask).sum()
                 / loss_mask.sum().clamp(min=1))
        return loss, parts, agree

    def train(self):
        args = self.args
        accum = args.GRADIENT_ACCUMULATION_STEPS
        n_batches = len(self.train_loader)
        self.log(f"Training {self.total_steps} steps (loss={args.DISTILL_LOSS}, mode={args.FINETUNE_MODE})")

        self.student.train()
        self.teacher.eval()
        running = LossAverager()
        pbar = _bar(total=self.total_steps, desc="train", unit="step", initial=self.step)

        for epoch in range(args.TRAIN_EPOCHS):
            for i, batch in enumerate(self.train_loader):
                loss, parts, _ = self._forward(batch)
                (loss / accum).backward()
                running.update(parts)

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
                    avg_parts = running.average()
                    avg_loss = sum(avg_parts.values())
                    detail = "  ".join(f"{k}={v:.4f}" for k, v in avg_parts.items())
                    self._record({"split": "train", "step": self.step, "epoch": epoch,
                                  "loss": avg_loss, **avg_parts, "lr": self.scheduler.get_last_lr()[0]})
                    self._log_console(f"[train] step {self.step}/{self.total_steps}  "
                                      f"loss={avg_loss:.4f}  {detail}")
                    running.reset()

                if self.step % args.EVAL_EVERY == 0:
                    self.evaluate()
                if self.step % args.SAVE_EVERY == 0:
                    self.save_checkpoint()

        pbar.close()
        self.evaluate()
        final_dir = os.path.join(args.work_dir, f"{args.approach}_final")
        save_model(self.student, self.tokenizer, final_dir)
        self.log(f"Saved final model to {final_dir}")

    @torch.no_grad()
    def evaluate(self):
        self.student.eval()
        totals = LossAverager()
        agrees, n = 0.0, 0

        for batch in _bar(iterable=self.val_loader, desc="eval", unit="batch", leave=False):
            _, parts, agree = self._forward(batch)
            totals.update(parts)
            agrees += agree.item()
            n += 1

        n = max(n, 1)
        avg_parts = totals.average()
        row = {"split": "val", "step": self.step, "loss": sum(avg_parts.values()),
               **avg_parts, "agreement": agrees / n}
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

    def _load_checkpoint_if_exists(self):
        """Resume step/optimizer/scheduler from the latest checkpoint if one exists in
        work_dir. The student's weights are already loaded from this same checkpoint by
        the entrypoint (see find_latest_checkpoint in distill.py) before the Trainer is
        constructed -- this only restores the rest of the training state."""
        latest_ckpt = find_latest_checkpoint(self.args.work_dir)
        if latest_ckpt is None:
            return

        trainer_state_path = os.path.join(latest_ckpt, "trainer_state.pt")
        self.log(f"Loading trainer state from {latest_ckpt}", console_print=True)
        trainer_state = torch.load(trainer_state_path, map_location=self.device)
        self.step = trainer_state["step"]
        self.optimizer.load_state_dict(trainer_state["optimizer"])
        self.scheduler.load_state_dict(trainer_state["scheduler"])
        self.log(f"Resumed from step {self.step}", console_print=True)

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
    # throttle refreshes hard when stdout isn't a tty (e.g. redirected to a slurm log)
    interactive = sys.stdout.isatty()
    return tqdm(
        file=sys.stdout, dynamic_ncols=True,
        mininterval=0.5 if interactive else 30.0,
        **kwargs,
    )
