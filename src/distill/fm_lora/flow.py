"""v_theta and the flow-matching math for fm_lora's last-layer training/inference
(see docs/fm_lora.md). VelocityMLP/sample_tau/flow_matching_step/euler_integrate are
kept free of any model/peft plumbing so they drive plain tensors -- z_0/z_1 in
R^{..., r} -- regardless of which projection produced them. FMLoraSite is the module
that wires that math into one target-module LoraLayer in the student's last decoder
block, replacing it in the model tree (see model.py's last_layer_sites/load_model).
"""
import torch
import torch.nn as nn

ADAPTER_NAME = "default"


class VelocityMLP(nn.Module):
    """v_theta(z, tau): small MLP, input z in R^r plus scalar tau in [0,1], output in R^r."""

    def __init__(self, rank, hidden):
        super().__init__()
        self.rank = rank
        self.net = nn.Sequential(
            nn.Linear(rank + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, rank),
        )

    def forward(self, z, tau):
        # tau is one scalar per (batch, position); z is [..., r]
        dtype = self.net[0].weight.dtype
        if tau.dim() == z.dim() - 1:
            tau = tau.unsqueeze(-1)
        return self.net(torch.cat([z.to(dtype), tau.to(dtype)], dim=-1))


def sample_tau(shape, device):
    """One tau ~ U[0,1] per (batch, position), matching z's leading dims."""
    return torch.rand(shape, device=device, dtype=torch.float32)


def flow_matching_step(z_0, z_1, v_theta):
    """Steps 3-7 of the spec's training forward pass: sample tau, interpolate, predict
    the velocity, and return the CFM loss plus the one-shot endpoint estimate used to
    decode student_logits (step 7). z_0, z_1: [..., r], z_1 detached (teacher, no grad)."""
    tau = sample_tau(z_0.shape[:-1], z_0.device)
    tau_ = tau.unsqueeze(-1)

    z_tau = (1.0 - tau_) * z_0 + tau_ * z_1
    pred_v = v_theta(z_tau, tau)

    l_fm = (pred_v.float() - (z_1 - z_0).float()).pow(2).mean(-1)
    z_1_hat = z_tau + (1.0 - tau_) * pred_v
    return l_fm, z_1_hat


@torch.no_grad()
def euler_integrate(z_0, v_theta, num_steps):
    """K-step Euler integration from z_0 (tau=0) to an estimate of z_1 (tau=1), used at
    inference (spec: 'Inference (K-step Euler, last layer only)')."""
    z = z_0
    dt = 1.0 / num_steps
    for k in range(num_steps):
        tau_k = torch.full(z.shape[:-1], k * dt, device=z.device, dtype=torch.float32)
        z = z + dt * v_theta(z, tau_k)
    return z


class FMLoraSite(nn.Module):
    """Replaces one target-module LoraLayer in the student's last decoder block
    (docs/fm_lora.md: "student's last hidden layer LoRA is replaced with fm_lora" --
    "last hidden layer" means the last transformer decoder block, and this applies
    independently to each LORA_TARGET_MODULES projection found there, e.g. o_proj and
    down_proj each get their own site with their own A/B/v_theta).

    Wraps the frozen Stage-1 base_layer plus the Stage-1-trained A_s/B_s (kept,
    still trainable per spec: "Student last layer: A_s, B_s, v_theta -- all
    trainable") and a freshly-added v_theta.

    mode="train": one-shot flow-matching estimate (spec steps 1-8), records l_fm as a
    side effect for the trainer to read after the forward pass.
    mode="infer": K-step Euler integration (spec's inference section).
    """

    def __init__(self, lora_layer, hidden, num_euler_steps):
        super().__init__()
        self.base_layer = lora_layer.base_layer
        self.A = lora_layer.lora_A[ADAPTER_NAME]
        self.B = lora_layer.lora_B[ADAPTER_NAME]
        self.dropout = lora_layer.lora_dropout[ADAPTER_NAME]
        self.scaling = lora_layer.scaling[ADAPTER_NAME]
        self.v_theta = VelocityMLP(lora_layer.r[ADAPTER_NAME], hidden)
        self.num_euler_steps = num_euler_steps

        self.mode = "train"    # "train" | "infer"
        self.z_1 = None        # set externally each step: teacher's A_t(x_t), no grad
        self.loss_mask = None  # [B, T] float, set externally each step
        self.l_fm = None       # populated by forward() in "train" mode

    def forward(self, x):
        base_out = self.base_layer(x)
        z_0 = self.A(self.dropout(x.to(self.A.weight.dtype)))

        if self.mode == "train":
            if self.z_1 is None or self.loss_mask is None:
                raise RuntimeError("FMLoraSite.z_1/loss_mask must be set before a 'train' forward")
            l_fm_per_tok, z_hat = flow_matching_step(z_0, self.z_1, self.v_theta)
            mask = self.loss_mask
            self.l_fm = (l_fm_per_tok * mask).sum() / mask.sum().clamp(min=1)
        elif self.mode == "infer":
            z_hat = euler_integrate(z_0, self.v_theta, self.num_euler_steps)
        else:
            raise ValueError(f"Unknown FMLoraSite mode {self.mode!r}")

        return base_out + self.B(z_hat.to(self.B.weight.dtype)) * self.scaling
