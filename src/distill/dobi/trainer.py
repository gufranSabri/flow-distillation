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

from .model import save_model
from utils.losses import resolve_distill_loss, kd_loss, hidden_mse, ce_loss
from utils.utils import LossAverager, find_latest_checkpoint
from utils.teacher_cache import TeacherStateCache


class DobiTrainer:
    """Trains the flow (projector + FlowNet) between the student's and the teacher's
    classifier inputs (see model.py) FM-KT style: rather than regressing the velocity at
    an interpolated point x_t = (1 - t) x0 + t x1 fed as the network's input -- which
    bakes the teacher's x1 into what the flow sees -- the flow is unrolled for all
    NUM_FLOW_STEPS Euler steps from the student's x0 on its own states, exactly as at
    inference. Its target at every step is still that same interpolant, evaluated at the
    step's own t, but now compared against the raw state the unroll actually reached:

        loss = W_FLOW * sum_i mse(x_t_i, (1 - t_i) x0 + t_i x1)   [dense, every step]
             + W_KL   * kd(readout(x_N), teacher)                 [only the final state]

    Gradients flow back through the whole unrolled trajectory. x_N (t=1) is the final
    Euler state, i.e. the model's actual output, and the only state KD is ever computed
    on -- whatever the fully-unrolled flow produces is what gets scored."""

    def __init__(self, args, student, teacher, tokenizer, train_ds, val_ds, collator):
        self.args = args
        self.student = student
        self.tokenizer = tokenizer
        self.device = args.device
        self.log = args.logger

        # distill.py caches the teacher's classifier-input hidden states up front and
        # passes teacher=None from here on (see utils/teacher_cache.py / USES_TEACHER_CACHE
        # in __init__.py) -- the teacher itself is never resident during training
        assert teacher is None, "DobiTrainer trains from the teacher cache; the live teacher should have been freed"
        self.teacher_cache = {
            "train": TeacherStateCache(args, args.teacher_model, "train"),
            "val": TeacherStateCache(args, args.teacher_model, "val"),
        }

        self.distill_loss = resolve_distill_loss(args.DISTILL_LOSS)

        # own generator, reseeded per epoch in train(), so epoch N's shuffle order is the
        # same whether or not the run was resumed (the skip-on-resume below relies on it)
        self.shuffle_gen = torch.Generator()
        self.train_loader = DataLoader(
            train_ds, batch_size=args.PER_DEVICE_TRAIN_BATCH_SIZE,
            shuffle=True, generator=self.shuffle_gen,
            collate_fn=collator, num_workers=2, pin_memory=True,
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
        self._resumed = False
        self._load_checkpoint_if_exists()

    def _log_console(self, message):
        self.log(message)
        tqdm.write(message, file=sys.stdout)

    def _warm_start_projector(self):
        """Fits the projector to the student's own logits before any flow-matching/KD
        training starts, so readout(x0) is already a sensible prediction (matching the
        base student almost exactly) rather than an arbitrary random projection: minimizes
        ||W_T P(h_S) - W_S h_S||^2 over a handful of training batches. Skipped on resume --
        the projector is already trained (or already warm-started) in that checkpoint."""
        args = self.args
        flow = self.student.flow
        opt = torch.optim.Adam(flow.projector.parameters(), lr=args.WARM_START_LR)

        n_batches = min(args.WARM_START_BATCHES, len(self.train_loader))
        self._log_console(f"Warm-starting projector on {n_batches} batches "
                           f"(||W_T P(h_S) - W_S h_S||^2) …")
        pbar = _bar(total=n_batches, desc="warm-start", unit="batch")
        for _, batch in zip(range(n_batches), self.train_loader):
            input_ids = batch["input_ids"].to(self.device)
            attn_mask = batch["attention_mask"].to(self.device)
            loss_mask = attn_mask.float()

            with torch.no_grad():
                h_s, _, _ = self.student.start_states(input_ids, attn_mask, use_cache=False)
                target = self.student.student_logits(h_s)  # W_S h_S, frozen

            x0 = flow.start(h_s)
            pred = flow.readout(x0)  # W_T P(h_S)
            loss = hidden_mse(pred, target.float(), loss_mask)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}", refresh=False)
        pbar.close()
        self._log_console(f"Warm-start done: final batch loss={loss.item():.4f}")

    @torch.no_grad()
    def _teacher(self, split, batch, seq_len):
        """The flow's target: the teacher's classifier input (its last hidden state after
        the final norm), read from the cache built before training (see
        utils/teacher_cache.py) and keyed by this batch's original dataset row ("idx",
        stamped on by build_dolly_datasets). Teacher logits are reconstructed from that
        cached state through the flow's frozen copy of the teacher's lm_head -- the live
        teacher is never run during training."""
        t_state = self.teacher_cache[split](batch["idx"], seq_len, self.device, dtype=torch.float32)
        t_logits = self.student.flow.readout(t_state)
        return t_logits, t_state

    def _forward(self, batch, split="train"):
        input_ids = batch["input_ids"].to(self.device)
        attn_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        loss_mask = (labels != -100).float()
        flow = self.student.flow

        t_logits, t_state = self._teacher(split, batch, input_ids.shape[1])
        h_s, x0, _ = self.student.start_states(input_ids, attn_mask, use_cache=False)
        x1 = t_state.float()

        fm = 0.0
        num_steps = flow.num_steps
        for i, x_t in enumerate(flow.unroll(x0, attn_mask)):
            # the flow-matching target at this step's t: the linear interpolant between the
            # student's own start and the teacher's endpoint, at the same t the Euler step
            # just advanced to -- dense supervision along the whole trajectory
            t = (i + 1) / num_steps
            interpolant = (1 - t) * x0 + t * x1
            fm = fm + hidden_mse(x_t, interpolant, attn_mask)
        x_n = x_t  # final Euler state (t=1): the model's actual output

        # KD is only evaluated once, on whatever the fully-unrolled flow produces
        s_logits = self.student.student_logits(h_s) + flow.readout(x_n) - flow.readout(x0)
        kd = kd_loss(s_logits, t_logits, labels, self.distill_loss, self.args.TEMPERATURE)

        parts = {"kd": self.args.W_KL * kd, "fm": self.args.W_FLOW * fm}
        loss = sum(parts.values())

        # eval hygiene: KL against the teacher and mean ground-truth LM loss, not just
        # argmax agreement -- kd above already is the KL/JSD (per DISTILL_LOSS) at
        # TEMPERATURE, so it's reused here rather than recomputed
        with torch.no_grad():
            agree = (((s_logits.argmax(-1) == t_logits.argmax(-1)).float() * loss_mask).sum()
                     / loss_mask.sum().clamp(min=1))
            mean_lm_loss = ce_loss(s_logits, labels, loss_mask)
        return loss, parts, {"agreement": agree, "kl": kd.detach(), "mean_lm_loss": mean_lm_loss}

    def train(self):
        args = self.args
        accum = args.GRADIENT_ACCUMULATION_STEPS
        n_batches = len(self.train_loader)
        self.log(f"Training {self.total_steps} steps (loss={args.DISTILL_LOSS}, "
                 f"flow steps={args.NUM_FLOW_STEPS})")

        self.student.train()
        if not self._resumed:
            self._warm_start_projector()
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
            self.shuffle_gen.manual_seed(args.seed + epoch)
            batches = enumerate(self.train_loader)
            skip = start_batch if epoch == start_epoch else 0
            if skip:
                self._log_console(f"Resuming epoch {epoch}: skipping {skip} already-trained batches")
                batches = itertools.islice(batches, skip, None)

            for i, batch in batches:
                loss, parts, _ = self._forward(batch, split="train")
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
        totals, stats = LossAverager(), LossAverager()

        for batch in _bar(iterable=self.val_loader, desc="eval", unit="batch", leave=False):
            _, parts, batch_stats = self._forward(batch, split="val")
            totals.update(parts)
            stats.update(batch_stats)

        avg_parts = totals.average()
        row = {"split": "val", "step": self.step, "loss": sum(avg_parts.values()),
               **avg_parts, **stats.average()}
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
        work_dir. The flow's weights are already loaded from this same checkpoint by the
        entrypoint (see find_latest_checkpoint in distill.py) before the Trainer is
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
        self._resumed = True
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
