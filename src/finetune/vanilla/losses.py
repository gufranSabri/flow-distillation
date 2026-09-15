import torch.nn.functional as F


def seq_mean(per_token, mask):
    # average within each sequence first, then across the batch, so long sequences don't dominate
    denom = mask.sum(-1).clamp(min=1)
    return ((per_token * mask).sum(-1) / denom).mean()


def ce_loss(logits, labels, mask):
    # labels are pre-shifted by the data pipeline: labels[t] targets logits[t]
    per_token = F.cross_entropy(
        logits.float().flatten(0, 1),
        labels.clamp(min=0).flatten(),
        reduction="none",
    ).view(labels.shape)
    return seq_mean(per_token, mask)
