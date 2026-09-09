import json
import math
import os
import shutil

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.models.word_level import RelationCapture, save_student


def _seq_mean(per_token, mask):
    # average within each sequence first, then across the batch, so long sequences
    # don't dominate (phase1.md 3)
    denom = mask.sum(-1).clamp(min=1)
    return ((per_token * mask).sum(-1) / denom).mean()


def kd_loss(student_logits, teacher_logits, mask, temperature):
    """Forward KL[p_teacher || q_student], computed explicitly in log-space."""
    t_logprobs = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
    s_logprobs = F.log_softmax(student_logits.float() / temperature, dim=-1)
    per_token = (t_logprobs.exp() * (t_logprobs - s_logprobs)).sum(-1)
    return _seq_mean(per_token, mask) * (temperature ** 2)


def ce_loss(student_logits, labels, mask):
    # labels are pre-shifted by the data pipeline: labels[t] targets logits[t]
    per_token = F.cross_entropy(
        student_logits.float().flatten(0, 1),
        labels.clamp(min=0).flatten(),
        reduction="none",
    ).view(labels.shape)
    return _seq_mean(per_token, mask)


def relation_loss(student_rel, teacher_rel, mask):
    """KL between teacher/student attention and value-relation maps (MiniLMv2)."""
    total = 0.0
    for (s_attn, s_val), (t_attn, t_val) in zip(student_rel, teacher_rel):
        for s, t in ((s_attn, t_attn), (s_val, t_val)):
            per_token = (t * (t.clamp_min(1e-9).log() - s.clamp_min(1e-9).log())).sum(-1)
            total = total + _seq_mean(per_token.mean(1), mask)
    return total / max(len(student_rel), 1)


class WordLevelTrainer:
    def __init__(self, args, student, teacher, tokenizer, train_ds, val_ds, collator):
        self.args = args
        self.student = student
        self.teacher = teacher
        self.tokenizer = tokenizer
        self.device = args.device
        self.log = args.logger

        self.use_logits = args.DISTILL_TARGET in ("logits", "both")
        self.use_hidden = args.DISTILL_TARGET in ("hidden_states", "both")
        if args.DISTILL_TARGET not in ("logits", "hidden_states", "both"):
            raise ValueError(f"Unknown DISTILL_TARGET {args.DISTILL_TARGET!r}")

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

        self.student_cap = self.teacher_cap = None
        if self.use_hidden:
            self.student_cap = RelationCapture(_base(student), args.RELATION_LAYERS)
            self.teacher_cap = RelationCapture(_base(teacher), args.RELATION_LAYERS)

        self.metrics_path = os.path.join(args.work_dir, "metrics.jsonl")
        self.step = 0

    def _forward(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        attn_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        loss_mask = (labels != -100).float()

        if self.student_cap:
            self.student_cap.clear()
            self.teacher_cap.clear()

        with torch.no_grad():
            teacher_out = self.teacher(input_ids=input_ids, attention_mask=attn_mask)
        student_out = self.student(input_ids=input_ids, attention_mask=attn_mask)

        s_logits, t_logits = student_out.logits, teacher_out.logits
        parts = {"ce": ce_loss(s_logits, labels, loss_mask)}

        if self.use_logits:
            parts["kd"] = kd_loss(s_logits, t_logits, loss_mask, self.args.TEMPERATURE)
        if self.use_hidden:
            R = self.args.NUM_RELATION_HEADS
            with torch.no_grad():
                teacher_rel = self.teacher_cap.relations(R, attn_mask)
            parts["rel"] = relation_loss(
                self.student_cap.relations(R, attn_mask), teacher_rel, loss_mask,
            )

        lam = self.args.KD_LAMBDA
        distill = parts.get("kd", torch.zeros((), device=self.device))
        if self.use_hidden:
            distill = distill + self.args.HIDDEN_WEIGHT * parts["rel"]
        loss = lam * distill + (1 - lam) * parts["ce"]

        # fraction of positions where student and teacher pick the same top-1 token
        agree = (((s_logits.argmax(-1) == t_logits.argmax(-1)).float() * loss_mask).sum()
                 / loss_mask.sum().clamp(min=1))
        return loss, parts, agree

    def train(self):
        args = self.args
        accum = args.GRADIENT_ACCUMULATION_STEPS
        n_batches = len(self.train_loader)
        self.log(f"Training {self.total_steps} steps "
                 f"(target={args.DISTILL_TARGET}, mode={args.FINETUNE_MODE})")

        self.student.train()
        self.teacher.eval()
        running = 0.0

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

                if self.step % args.LOG_EVERY == 0:
                    avg = running / (accum * args.LOG_EVERY)
                    detail = "  ".join(f"{k}={v.item():.4f}" for k, v in parts.items())
                    self._record({"split": "train", "step": self.step, "epoch": epoch,
                                  "loss": avg, "lr": self.scheduler.get_last_lr()[0]})
                    self.log(f"[train] step {self.step}/{self.total_steps}  "
                             f"loss={avg:.4f}  {detail}", console_print=True)
                    running = 0.0

                if self.step % args.EVAL_EVERY == 0:
                    self.evaluate()
                if self.step % args.SAVE_EVERY == 0:
                    self.save_checkpoint()

        self.evaluate()
        final_dir = os.path.join(args.work_dir, f"{args.APPROACH}_final")
        save_student(self.student, self.tokenizer, final_dir)
        self.log(f"Saved final model to {final_dir}")

        if self.student_cap:
            self.student_cap.remove()
            self.teacher_cap.remove()

    @torch.no_grad()
    def evaluate(self):
        self.student.eval()
        totals, agrees, n = {}, 0.0, 0

        for batch in self.val_loader:
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
        self.log("[val] " + "  ".join(
            f"{k}={v:.4f}" for k, v in row.items() if isinstance(v, float)
        ), console_print=True)

        self.student.train()
        return row

    def save_checkpoint(self):
        ckpt_dir = os.path.join(self.args.work_dir, f"checkpoint-{self.step}")
        save_student(self.student, self.tokenizer, ckpt_dir)
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


def _base(model):
    # unwrap PEFT so hooks land on the real decoder layers
    return model.get_base_model() if hasattr(model, "get_base_model") else model
