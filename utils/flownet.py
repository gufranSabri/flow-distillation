"""FlowNet: DiT-style velocity network (DOBI's, see src/distill/dobi).

Each block is AdaLN(t)-conditioned self-attention over the flow state, cross-attention to
a fixed context (the student's projected hidden states), and an FFN. Unlike DOBI's
original, both attentions are CAUSAL: position t never sees positions > t. With a
bidirectional mask, position t could read the student state at t+1 -- which encodes the
very token t is scored on -- and the net would learn to copy the answer during training,
then fail at generation time where t+1 doesn't exist yet.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimestepEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, t):
        # t in [0, 1), scaled to DDPM-style [0, 1000) before the sinusoid
        half = self.d_model // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = (t * 1000)[:, None].float() * freqs[None]
        return self.mlp(torch.cat([torch.sin(args), torch.cos(args)], dim=-1))


class AdaLN(nn.Module):
    def __init__(self, d_model, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, d_model * 2))

    def forward(self, x, cond):
        scale, shift = self.proj(cond).chunk(2, dim=-1)  # each [B, d_model]
        return self.norm(x) * (1.0 + scale[:, None]) + shift[:, None]


class Attention(nn.Module):
    """Multi-head attention under a boolean [B, 1, T, S] mask (True = may attend)."""

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, 2 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, context, mask):
        B, T, D = x.shape
        q = self.q(x).view(B, T, self.num_heads, -1).transpose(1, 2)
        k, v = self.kv(context).view(B, context.shape[1], 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(out.transpose(1, 2).reshape(B, T, D))


class FlowBlock(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio):
        super().__init__()
        self.adaLN_self = AdaLN(d_model, d_model)
        self.self_attn = Attention(d_model, num_heads)
        self.adaLN_cross = AdaLN(d_model, d_model)
        self.cross_attn = Attention(d_model, num_heads)
        self.adaLN_ffn = AdaLN(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_model * mlp_ratio), d_model),
        )

    def forward(self, x, t_emb, mask, context):
        h = self.adaLN_self(x, t_emb)
        x = x + self.self_attn(h, h, mask)
        x = x + self.cross_attn(self.adaLN_cross(x, t_emb), context, mask)
        x = x + self.ffn(self.adaLN_ffn(x, t_emb))
        return x


def causal_mask(attention_mask, T, device):
    """[B, 1, T, T] (or [1, 1, T, T] without padding) boolean mask: causal, never attending
    to padding. The diagonal is always allowed so a padding query (left-padded generation
    batches) has something to attend to instead of producing a NaN row -- its output is
    never read."""
    allowed = torch.ones(T, T, dtype=torch.bool, device=device).tril()
    if attention_mask is not None:
        allowed = allowed & attention_mask.bool()[:, None, :]
        allowed = allowed | torch.eye(T, dtype=torch.bool, device=device)
    return allowed.unsqueeze(-3)


class FlowNet(nn.Module):
    def __init__(self, hidden_dim, d_model=512, num_layers=4, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.input_proj = nn.Linear(hidden_dim, d_model, bias=False)
        self.src_proj = nn.Linear(hidden_dim, d_model, bias=False)
        self.timestep_emb = TimestepEmbedding(d_model)
        self.blocks = nn.ModuleList([FlowBlock(d_model, num_heads, mlp_ratio) for _ in range(num_layers)])
        self.norm_out = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, hidden_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # zero velocity at init: the flow starts as the identity map on the student's state
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x, t, attention_mask, context):
        """x, context: [B, T, hidden_dim]; t: [B] in [0, 1); attention_mask: [B, T] or None.
        Returns the velocity at (x, t), [B, T, hidden_dim]."""
        mask = causal_mask(attention_mask, x.shape[1], x.device)
        t_emb = self.timestep_emb(t)
        h = self.input_proj(x.float())
        context = self.src_proj(context.float())
        for block in self.blocks:
            h = block(h, t_emb, mask, context)
        return self.output_proj(self.norm_out(h))
