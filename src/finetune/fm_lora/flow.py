import math

import torch
import torch.nn as nn


def timestep_embedding(t, dim):
    """Standard sinusoidal embedding of t in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    angles = t.float().unsqueeze(-1) * freqs * 1000.0
    emb = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
    return emb


class VelocityNet(nn.Module):
    """v_theta(z_t, t, x): predicts the flow velocity in the site's output space.

    Conditioning on x goes through a rank-`rank` bottleneck so the per-site
    parameter cost stays in the same ballpark as a LoRA (A, B) pair.
    """

    def __init__(self, in_features, out_features, hidden, time_dim, rank):
        super().__init__()
        # kept so save_model can persist the shape needed to rebuild this net
        self.time_dim = time_dim
        self.hidden = hidden
        self.rank = rank
        self.cond_down = nn.Linear(in_features, rank, bias=False)
        self.net = nn.Sequential(
            nn.Linear(out_features + rank + time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_features),
        )
        # start near zero so the untrained adapter barely perturbs the frozen model
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_t, t, x):
        # the net carries the base layer's dtype (bf16 in practice); cast every input to
        # it here so callers can hand us fp32 flow state without a dtype mismatch
        dtype = self.cond_down.weight.dtype
        cond = self.cond_down(x.to(dtype))
        t_emb = timestep_embedding(t, self.time_dim)
        if t_emb.dim() == z_t.dim() - 1:
            t_emb = t_emb.unsqueeze(-2).expand(*z_t.shape[:-1], t_emb.shape[-1])
        return self.net(torch.cat([z_t.to(dtype), cond, t_emb.to(dtype)], dim=-1))


class FMLoRALinear(nn.Module):
    """Frozen base Linear whose adapter delta is produced by an integrated flow.

    Three modes:
      - "base":   h = W0 x                      (adapters off; used to collect targets)
      - "cfm":    h = W0 x, and the CFM loss for this site is recorded as a side effect
      - "flow":   h = W0 x + integrate(noise)   (inference)
    """

    def __init__(self, base, hidden, time_dim, rank, normalize, steps):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.velocity = VelocityNet(
            in_features=base.in_features,
            out_features=base.out_features,
            hidden=hidden,
            time_dim=time_dim,
            rank=rank,
        ).to(dtype=base.weight.dtype, device=base.weight.device)
        self.normalize = normalize
        self.steps = steps

        # EMA of the target's RMS, tracked during training so inference (which has no
        # target) can un-normalize the integrated flow with the same scale. A buffer, so
        # .to(device) moves it with the module; _apply below keeps it out of the dtype
        # sweep, since a bf16 scalar EMA would lose the small (1 - momentum) updates.
        self.register_buffer("target_rms", torch.ones((), dtype=torch.float32))
        self.rms_momentum = 0.99

        self.mode = "base"
        self.target = None        # x1, set during the target-collection pass
        self.loss = None          # per-site CFM loss, set during a "cfm" pass
        self.loss_mask = None     # [B, T] float mask over label positions

    def _apply(self, fn, recurse=True):
        # .to(dtype) would sweep target_rms into bf16 along with the weights; restore it
        # afterwards so the move still relocates it but the EMA keeps fp32 precision
        module = super()._apply(fn, recurse)
        module.target_rms = module.target_rms.float()
        return module

    def _scale(self, target):
        """Per-site RMS, so sites with large activations don't dominate the summed loss."""
        if not self.normalize:
            return torch.ones((), device=target.device, dtype=torch.float32)
        rms = target.float().pow(2).mean().sqrt().clamp(min=1e-6)
        if self.training:
            with torch.no_grad():
                self.target_rms.mul_(self.rms_momentum).add_(rms.detach() * (1 - self.rms_momentum))
        return rms

    def forward(self, x):
        out = self.base(x)

        if self.mode == "base":
            return out

        if self.mode == "cfm":
            self.loss = self._cfm_loss(x, out)
            return out

        if self.mode == "flow":
            return out + self._integrate(x, out)

        raise ValueError(f"Unknown FMLoRALinear mode {self.mode!r}")

    def _cfm_loss(self, x, out):
        target = self.target
        if target is None:
            raise RuntimeError("FMLoRALinear.target must be set before a 'cfm' forward")

        scale = self._scale(target)
        x1 = target.float() / scale
        x0 = torch.randn_like(x1)

        # one t per (batch, position): the flow is fit independently at every token
        t = torch.rand(x1.shape[:-1], device=x1.device, dtype=torch.float32)
        z_t = (1.0 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * x1

        pred = self.velocity(z_t.to(x.dtype), t, x.detach()).float()
        per_token = (pred - (x1 - x0)).pow(2).mean(-1)

        mask = self.loss_mask
        if mask is None:
            return per_token.mean()
        return (per_token * mask).sum() / mask.sum().clamp(min=1.0)

    @torch.no_grad()
    def _integrate(self, x, out):
        # no target at inference, so un-normalize with the RMS tracked during training
        scale = self.target_rms.to(out.device) if self.normalize else torch.ones((), device=out.device)
        z = torch.randn(out.shape, device=out.device, dtype=torch.float32)
        dt = 1.0 / self.steps

        for k in range(self.steps):
            t = torch.full(out.shape[:-1], k * dt, device=out.device, dtype=torch.float32)
            v = self.velocity(z.to(x.dtype), t, x).float()
            z = z + dt * v

        return (z * scale).to(out.dtype) - out


def set_mode(sites, mode):
    for site in sites:
        site.mode = mode


def set_loss_mask(sites, mask):
    for site in sites:
        site.loss_mask = mask


def clear(sites):
    for site in sites:
        site.target = None
        site.loss = None
