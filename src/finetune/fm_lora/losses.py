import torch


def aggregate_site_losses(sites, reduction="mean"):
    """Sums (or means) the per-site CFM losses recorded during a 'cfm' forward pass."""
    losses = [s.loss for s in sites if s.loss is not None]
    if not losses:
        raise RuntimeError("No FM-LoRA site recorded a loss; was the forward run in 'cfm' mode?")
    stacked = torch.stack(losses)
    return stacked.sum() if reduction == "sum" else stacked.mean()
