"""DOBI: flow distiller (frozen student -> projector -> FlowNet -> teacher lm_head),
trained FM-KT style by unrolling the flow rather than on interpolated points.

The flow runs between the two LMs' classifier inputs: the hidden state each model's
lm_head consumes, i.e. its last hidden state *after* the final norm. The student LM is
frozen. Its classifier input is projected up into the teacher's hidden size (x0 -- the
two spaces differ in width, so some lift is unavoidable), and the FlowNet walks it in
NUM_FLOW_STEPS Euler steps to an estimate of the teacher's classifier input x1. Logits
are read out through a frozen copy of the teacher's lm_head alone, so the flow's output
is exactly what the teacher's classifier would see, and the teacher is never run at
inference.

A checkpoint is the student LM (save_pretrained, plus tokenizer) with DOBI_FILE next to
it holding the flow's config and weights -- readout included, so it loads standalone.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from utils.flownet import FlowNet

DOBI_FILE = "dobi.pt"


class TeacherReadout(nn.Module):
    """Frozen copy of the teacher's lm_head, in the teacher's bf16: maps a teacher-space
    classifier input to logits. The final norm is not part of it -- it comes before the
    flow's target."""

    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.to(torch.bfloat16).requires_grad_(False)

    def copy_from(self, teacher):
        self.load_state_dict({"head.weight": teacher.lm_head.weight})

    def forward(self, x):
        return self.head(x.to(self.head.weight.dtype))


class DobiFlow(nn.Module):
    """Everything DOBI adds on top of the frozen student: projector, FlowNet, readout."""

    def __init__(self, student_dim, teacher_dim, vocab_size, num_steps, d_model, num_layers, num_heads):
        super().__init__()
        # saved with the weights, so a checkpoint rebuilds exactly this flow
        self.cfg = dict(
            student_dim=student_dim, teacher_dim=teacher_dim, vocab_size=vocab_size,
            num_steps=num_steps, d_model=d_model, num_layers=num_layers, num_heads=num_heads,
        )
        self.num_steps = num_steps

        self.projector = nn.Sequential(
            nn.Linear(student_dim, teacher_dim),
            nn.GELU(),
            nn.Linear(teacher_dim, teacher_dim),
        )
        self.flownet = FlowNet(teacher_dim, d_model=d_model, num_layers=num_layers, num_heads=num_heads)
        self.readout = TeacherReadout(teacher_dim, vocab_size)

    def start(self, h_student):
        """x0: the student's classifier input projected into teacher space."""
        return self.projector(h_student.float())

    def unroll(self, x0, attention_mask):
        """Euler-integrates the flow from x0 on its own states (exactly as at inference),
        yielding the raw state x_t after every step's update. The last of these (t=1) is
        the final Euler state, i.e. the inference output."""
        x = x0
        for i in range(self.num_steps):
            t = i / self.num_steps
            v = self.flownet(x, torch.full((x.shape[0],), t, device=x.device), attention_mask, context=x0)
            x = x + v / self.num_steps
            yield x

    def integrate(self, x0, attention_mask):
        for x in self.unroll(x0, attention_mask):
            pass
        return x


class DobiModel(PreTrainedModel, GenerationMixin):
    """The frozen student LM + DobiFlow as one causal LM that HF .generate() can drive
    (benchmark.py).

    The student is a floor (residual, L2D-style): logits = W_S h_S + (readout(x_N) -
    readout(x_0)), i.e. the student's own (frozen) logits plus the flow's correction, read
    out in teacher space and expressed as a displacement from x_0. Since the FlowNet's
    output_proj is zero-initialized (see utils/flownet.py), x_N == x_0 at init, the
    correction is exactly zero, and the whole model is exactly the base student. Training
    only has to learn the correction on top of that floor."""

    # attention runs inside the wrapped student, which already validated its own backend;
    # without these PreTrainedModel.__init__ rejects (or rewrites) the shared config's choice
    _supports_sdpa = _supports_flash_attn = _supports_flex_attn = _supports_attention_backend = True

    def __init__(self, lm, flow):
        super().__init__(lm.config)
        self.lm = lm.requires_grad_(False)
        self.flow = flow
        self.generation_config = lm.generation_config
        self._h_cache = None
        self._x0_cache = None

    def train(self, mode=True):
        super().train(mode)
        self.lm.eval()  # frozen
        return self

    def start_states(self, input_ids, attention_mask=None, **kwargs):
        """h_S (the student decoder's last_hidden_state, after its final norm -- exactly
        what the student's lm_head consumes) and x0 = flow.start(h_S), the same state
        lifted into teacher space, plus the student decoder's output."""
        out = self.lm.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        h_s = out.last_hidden_state
        return h_s, self.flow.start(h_s), out

    def student_logits(self, h_s):
        """W_S h_S: the frozen base student's own logits (the floor)."""
        with torch.no_grad():
            return self.lm.lm_head(h_s)

    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None,
                use_cache=None, labels=None, **kwargs):
        # once generate() has the student's KV cache warm it feeds only the new tokens, but
        # the causal FlowNet attends over every earlier position, so the prefix's h_S/x0 is
        # kept alongside that cache and the flow is re-run over the whole sequence
        prefix_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        h_s, x0, out = self.start_states(
            input_ids, attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, use_cache=use_cache, **kwargs,
        )
        if prefix_len:
            assert self._h_cache is not None and self._h_cache.shape[1] == prefix_len, \
                "past_key_values was not built by this model's previous forward"
            h_s = torch.cat([self._h_cache, h_s], dim=1)
            x0 = torch.cat([self._x0_cache, x0], dim=1)
        self._h_cache, self._x0_cache = h_s, x0

        x_n = self.flow.integrate(x0, attention_mask)
        logits = (self.student_logits(h_s)
                  + self.flow.readout(x_n) - self.flow.readout(x0))[:, prefix_len:]

        loss = None
        if labels is not None:
            # HF convention (benchmark.py): labels are unshifted, same length as input_ids
            loss = F.cross_entropy(
                logits[:, :-1].flatten(0, 1).float(), labels[:, 1:].flatten(), ignore_index=-100,
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=out.past_key_values)


def _load_lm(model_id, dtype):
    return AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, trust_remote_code=True)


def is_dobi_checkpoint(path):
    return os.path.isfile(os.path.join(path, DOBI_FILE))


def load_checkpoint(path, dtype=torch.bfloat16):
    """A dobi checkpoint written by save_model (resume, or benchmark.py)."""
    saved = torch.load(os.path.join(path, DOBI_FILE), map_location="cpu", weights_only=True)
    flow = DobiFlow(**saved["config"])
    flow.load_state_dict(saved["state_dict"])
    return DobiModel(_load_lm(path, dtype), flow)


def load_model(args, model_id):
    """model_id is either the student to start from -- the flow is then built fresh, its
    readout copied from --teacher-model -- or a dobi checkpoint to resume from."""
    if is_dobi_checkpoint(model_id):
        return load_checkpoint(model_id)

    lm = _load_lm(model_id, torch.bfloat16)
    teacher = _load_lm(args.teacher_model, torch.bfloat16)
    flow = DobiFlow(
        student_dim=lm.config.hidden_size,
        teacher_dim=teacher.config.hidden_size,
        vocab_size=teacher.config.vocab_size,
        num_steps=args.NUM_FLOW_STEPS,
        d_model=args.FLOWNET_D_MODEL,
        num_layers=args.FLOWNET_LAYERS,
        num_heads=args.FLOWNET_HEADS,
    )
    flow.readout.copy_from(teacher)
    del teacher
    return DobiModel(lm, flow)


def save_model(model, tokenizer, save_dir):
    model.lm.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)
    torch.save(
        {"config": model.flow.cfg, "state_dict": model.flow.state_dict()},
        os.path.join(save_dir, DOBI_FILE),
    )
