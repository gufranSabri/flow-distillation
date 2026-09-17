import itertools
import json
import math
import os
import shutil
import sys

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from .model import save_model, last_layer_sites, ADAPTER_NAME
from utils.losses import resolve_distill_loss, kd_loss
from utils.utils import LossAverager, find_latest_checkpoint


class FMLoRATrainer:
    """Stage 2 of docs/fm_lora.md. Teacher is fully frozen; only the student trains.

    standard mode: student's last decoder block keeps normal peft LoRA (A_s, B_s) at
    every target module, trained on KL(P_t || student) alone, like every other layer.

    flow_matching mode: each target-module LoRA in the student's last decoder block
    is an FMLoraSite (see model.py/flow.py) whose output is produced by one-shot flow
    matching toward the teacher's own same-projection LoRA intermediate z_t =
    A_t(x_t), instead of a direct A_s/B_s composition of that projection's input.
    Every site fires this during a single student forward pass (its loss recorded as
    a side effect); see _forward_flow_matching for the exact steps (numbered per the
    spec), run once per site.
    """

    def __init__(self, args, student, teacher, tokenizer, train_ds, val_ds, collator):
        self.args = args
        self.student = student
        self.teacher = teacher
        self.tokenizer = tokenizer
        self.device = args.device
        self.log = args.logger
        self.mode = args.distillation_mode

        self.distill_loss = resolve_distill_loss(args.DISTILL_LOSS)
        if self.mode == "flow_matching":
            # {name: (parent, attr, LoraLayer)} for the teacher's own (never-swapped)
            # last-layer sites, keyed the same way as student.fm_sites, so each
            # step's hook loop can zip them by name
            self.teacher_last = last_layer_sites(teacher, student.fm_sites.keys())
        else:
            self.teacher_last = None

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

    def _last_hidden(self, model, input_ids, attn_mask):
        out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
        return out.hidden_states[-1], out.logits

    def _forward_standard(self, input_ids, attn_mask, labels):
        with torch.no_grad():
            _, t_logits = self._last_hidden(self.teacher, input_ids, attn_mask)
        # last layer's A_s/B_s are the same trainable peft params as every other student
        # LoRA layer, so a plain forward already gives base_s + B_s(A_s(h_s)) * scaling
        _, s_logits = self._last_hidden(self.student, input_ids, attn_mask)

        parts = {"kd": kd_loss(s_logits, t_logits, labels, self.distill_loss, self.args.TEMPERATURE)}
        return sum(parts.values()), parts, s_logits, t_logits

    def _teacher_z1(self, input_ids, attn_mask):
        """Runs the teacher once, capturing each last-layer site's own input x_t via a
        forward pre-hook (mirrors the deleted src/finetune/fm_lora/trainer.py's
        _collect_targets: register hooks, run one forward, remove hooks) and returns
        {name: z_1 = A_t(dropout(x_t))} plus the teacher's logits."""
        captured = {}

        def make_hook(name, lora_layer):
            def hook(module, args):
                x_t = args[0]
                A_t, dropout = lora_layer.lora_A[ADAPTER_NAME], lora_layer.lora_dropout[ADAPTER_NAME]
                captured[name] = A_t(dropout(x_t.to(A_t.weight.dtype)))
            return hook

        handles = [
            lora_layer.register_forward_pre_hook(make_hook(name, lora_layer))
            for name, (_, _, lora_layer) in self.teacher_last.items()
        ]
        try:
            t_logits = self.teacher(input_ids=input_ids, attention_mask=attn_mask).logits
        finally:
            for h in handles:
                h.remove()

        return captured, t_logits

    def _forward_flow_matching(self, input_ids, attn_mask, labels):
        loss_mask = (labels != -100).float()

        with torch.no_grad():
            z_1_by_site, t_logits = self._teacher_z1(input_ids, attn_mask)

        for name, site in self.student.fm_sites.items():
            site.mode = "train"
            site.z_1 = z_1_by_site[name].detach()
            site.loss_mask = loss_mask

        # each site's forward fires inline here, recording site.l_fm as a side effect
        s_logits = self.student(input_ids=input_ids, attention_mask=attn_mask).logits

        l_fm = sum(site.l_fm for site in self.student.fm_sites.values())
        l_kl = kd_loss(s_logits, t_logits, labels, self.distill_loss, self.args.TEMPERATURE)

        parts = {"fm": l_fm, "kl": l_kl}
        return sum(parts.values()), parts, s_logits, t_logits

    def _forward(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        attn_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        loss_mask = (labels != -100).float()

        if self.mode == "flow_matching":
            loss, parts, s_logits, t_logits = self._forward_flow_matching(input_ids, attn_mask, labels)
        else:
            loss, parts, s_logits, t_logits = self._forward_standard(input_ids, attn_mask, labels)

        agree = (((s_logits.argmax(-1) == t_logits.argmax(-1)).float() * loss_mask).sum()
                 / loss_mask.sum().clamp(min=1))
        return loss, parts, agree

    def train(self):
        args = self.args
        accum = args.GRADIENT_ACCUMULATION_STEPS
        n_batches = len(self.train_loader)
        self.log(f"Training {self.total_steps} steps (mode={self.mode}, loss={args.DISTILL_LOSS})")

        self.student.train()
        self.teacher.eval()
        running = LossAverager()
        pbar = _bar(total=self.total_steps, desc="train", unit="step", initial=self.step)

        # resuming: self.step/optimizer/scheduler already reflect the checkpoint, but a
        # fresh `for epoch ... enumerate(self.train_loader)` would restart at batch 0
        # regardless, redoing already-trained data and overshooting total_steps (and,
        # since the scheduler was built for total_steps, training at LR=0 past it).
        # Skip exactly the batches already consumed so training picks up where it left off.
        consumed_batches = self.step * accum
        start_epoch, start_batch = divmod(consumed_batches, n_batches)

        for epoch in range(start_epoch, args.TRAIN_EPOCHS):
            batches = enumerate(self.train_loader)
            skip = start_batch if epoch == start_epoch else 0
            if skip:
                self._log_console(f"Resuming epoch {epoch}: skipping {skip} already-trained batches")
                batches = itertools.islice(batches, skip, None)

            for i, batch in batches:
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
    interactive = sys.stdout.isatty()
    return tqdm(
        file=sys.stdout, dynamic_ncols=True,
        mininterval=0.5 if interactive else 30.0,
        **kwargs,
    )
