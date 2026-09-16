"""Loss functions shared across finetune/distill approaches.

A new approach (src/finetune/<name>/ or src/distill/<name>/) should compose its
training signal from the functions here rather than defining its own losses.py: add
a function here only when no existing one covers it. A trio's Trainer._forward should
return a {name: loss_tensor} dict of every component going into the step -- the main
loss (CE for finetune, KD for distill) -- so the training loop can sum it into the
total and log every component, not just the total.
"""
import torch
import torch.nn.functional as F


def seq_mean(per_token, mask):
    # average within each sequence first, then across the batch, so long sequences don't dominate
    denom = mask.sum(-1).clamp(min=1)
    return ((per_token * mask).sum(-1) / denom).mean()


def ce_loss(logits, labels, mask):
    """Finetune's main loss: standard next-token cross-entropy against ground-truth labels."""
    # labels are pre-shifted by the data pipeline: labels[t] targets logits[t]
    per_token = F.cross_entropy(
        logits.float().flatten(0, 1),
        labels.clamp(min=0).flatten(),
        reduction="none",
    ).view(labels.shape)
    return seq_mean(per_token, mask)


def forward_kl(logits, teacher_logits, no_model_batch):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(logits)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    prod_probs = torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (no_model_batch["label"] != -100).int()
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss


def reverse_kl(logits, teacher_logits, no_model_batch):
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(teacher_logits) | torch.isinf(logits)
    prod_probs = torch.masked_fill(student_probs * teacher_logprobs, inf_mask, 0)
    prod_probs = prod_probs - torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (no_model_batch["label"] != -100).int()
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss


def _both_kl(logits, teacher_logits, no_model_batch):
    return forward_kl(logits, teacher_logits, no_model_batch) + reverse_kl(logits, teacher_logits, no_model_batch)


DISTILL_LOSSES = {
    "forward_kl": forward_kl,
    "reverse_kl": reverse_kl,
    "both": _both_kl,
}


def resolve_distill_loss(name):
    if name not in DISTILL_LOSSES:
        raise ValueError(f"Unknown DISTILL_LOSS {name!r}; choose one of {sorted(DISTILL_LOSSES)}")
    return DISTILL_LOSSES[name]


def kd_loss(student_logits, teacher_logits, labels, loss_fn, temperature):
    """Distill's main loss: student/teacher next-token distribution matching.

    loss_fn (one of DISTILL_LOSSES, via resolve_distill_loss) reduces over one sequence
    at a time, so call it per-sequence and average across the batch.
    """
    s_logits = student_logits.float() / temperature
    t_logits = teacher_logits.float() / temperature
    per_seq = torch.stack([
        loss_fn(s_logits[i:i + 1], t_logits[i:i + 1], {"label": labels[i:i + 1]})
        for i in range(s_logits.shape[0])
    ])
    # Hinton temperature convention: soften both sides, rescale by tau^2 (no-op at the default TEMPERATURE=1.0).
    return per_seq.mean() * (temperature ** 2)
