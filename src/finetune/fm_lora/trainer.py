import json
import math
import os
import shutil
import sys

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from .model import save_model, collect_sites
from .flow import set_mode, set_loss_mask, clear
from .losses import aggregate_site_losses


class FMLoRATrainer:
    """Flow-matching LoRA: fits a per-site velocity field to the base model's own
    label-conditioned activations (see docs/flow-matching-lora.md).

    Each step is two passes over the frozen backbone:
      1. adapters off, teacher-forced on prompt+label -> record each site's output as x1
      2. adapters in 'cfm' mode -> each site samples (t, x0) and records its CFM loss
    Only the velocity networks receive gradients.
    """

    def __init__(self, args, model, tokenizer, train_ds, val_ds, collator):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.device = args.device
        self.log = args.logger
        self.sites = collect_sites(model)

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
        self.trainable = [p for p in model.parameters() if p.requires_grad]
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
        self.log(message)
        tqdm.write(message, file=sys.stdout)

    @torch.no_grad()
    def _collect_targets(self, input_ids, attn_mask):
        """Pass 1: adapters off. Each site's own output on the teacher-forced
        prompt+label sequence is that site's flow target x1."""
        set_mode(self.sites, "base")
        handles = [
            site.register_forward_hook(
                lambda mod, inp, out: setattr(mod, "target", out.detach())
            )
            for site in self.sites
        ]
        try:
            self.model(input_ids=input_ids, attention_mask=attn_mask)
        finally:
            for h in handles:
                h.remove()

    def _forward(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        attn_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        loss_mask = (labels != -100).float()

        clear(self.sites)
        self._collect_targets(input_ids, attn_mask)

        # Pass 2: same sequence, adapters in CFM mode. Sites reuse the x1 recorded
        # above and regress the straight-line velocity at a random t.
        set_loss_mask(self.sites, loss_mask)
        set_mode(self.sites, "cfm")
        self.model(input_ids=input_ids, attention_mask=attn_mask)

        loss = aggregate_site_losses(self.sites, self.args.FM_LOSS_REDUCTION)
        set_mode(self.sites, "base")
        return loss, {"cfm": loss}

    def train(self):
        args = self.args
        accum = args.GRADIENT_ACCUMULATION_STEPS
        n_batches = len(self.train_loader)
        self.log(f"Training {self.total_steps} steps "
                 f"(fm-lora, {len(self.sites)} sites, reduction={args.FM_LOSS_REDUCTION})")

        self.model.train()
        running = 0.0
        pbar = _bar(total=self.total_steps, desc="train", unit="step", initial=self.step)

        for epoch in range(args.TRAIN_EPOCHS):
            for i, batch in enumerate(self.train_loader):
                loss, parts = self._forward(batch)
                (loss / accum).backward()
                running += loss.item()

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
                    self._record({"split": "train", "step": self.step, "epoch": epoch,
                                  "loss": avg, "lr": self.scheduler.get_last_lr()[0]})
                    self._log_console(f"[train] step {self.step}/{self.total_steps}  loss={avg:.4f}")
                    running = 0.0

                if self.step % args.EVAL_EVERY == 0:
                    self.evaluate()
                if self.step % args.SAVE_EVERY == 0:
                    self.save_checkpoint()

        pbar.close()
        self.evaluate()
        final_dir = os.path.join(args.work_dir, f"{args.approach}_final")
        save_model(self.model, self.tokenizer, final_dir)
        self.log(f"Saved final model to {final_dir}")

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss, n = 0.0, 0

        for batch in _bar(iterable=self.val_loader, desc="eval", unit="batch", leave=False):
            loss, _ = self._forward(batch)
            total_loss += loss.item()
            n += 1

        n = max(n, 1)
        row = {"split": "val", "step": self.step, "loss": total_loss / n}
        self._record(row)
        self._log_console("[val] " + "  ".join(f"{k}={v:.4f}" for k, v in row.items() if isinstance(v, float)))

        self.model.train()
        return row

    def save_checkpoint(self):
        ckpt_dir = os.path.join(self.args.work_dir, f"checkpoint-{self.step}")
        save_model(self.model, self.tokenizer, ckpt_dir)
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
    interactive = sys.stdout.isatty()
    return tqdm(
        file=sys.stdout, dynamic_ncols=True,
        mininterval=0.5 if interactive else 30.0,
        **kwargs,
    )
