import copy

import torch
from transformers import AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import repeat_kv


class RelationCapture:
    """Captures per-layer Q/K/V by pre-hooking the projections inside selected
    attention blocks, since Qwen applies RoPE internally and never returns them."""

    def __init__(self, model, layer_idxs):
        self.cfg = model.config
        self.acts = {}
        self.handles = []

        layers = model.model.layers if hasattr(model, "model") else model.layers
        self.layer_idxs = [i % len(layers) for i in layer_idxs]
        self.attns = {i: layers[i].self_attn for i in self.layer_idxs}

        for i in self.layer_idxs:
            # q/k/v share one input, so capture it once; the projections are applied
            # later in relations() -- calling them here would re-enter this hook
            handle = self.attns[i].q_proj.register_forward_pre_hook(self._make_hook(i))
            self.handles.append(handle)

    def _make_hook(self, layer_idx):
        def hook(module, inputs):
            self.acts[layer_idx] = inputs[0]
        return hook

    def clear(self):
        self.acts.clear()

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def relations(self, num_relation_heads, mask):
        """Returns per-layer (attention, value-relation) maps of shape
        (B, R, L, L) -- independent of hidden size and head count."""
        cfg = self.cfg
        n_heads, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads
        R = num_relation_heads

        out = []
        for i in self.layer_idxs:
            attn, x = self.attns[i], self.acts[i]
            q, k, v = attn.q_proj(x), attn.k_proj(x), attn.v_proj(x)
            B, L, _ = q.shape

            q = q.view(B, L, n_heads, head_dim).transpose(1, 2)
            k = repeat_kv(k.view(B, L, n_kv, head_dim).transpose(1, 2), n_heads // n_kv)
            v = repeat_kv(v.view(B, L, n_kv, head_dim).transpose(1, 2), n_heads // n_kv)

            # MiniLMv2: concatenate real heads, then re-split into R relation heads
            def to_relation_heads(t):
                t = t.transpose(1, 2).reshape(B, L, n_heads * head_dim)
                return t.view(B, L, R, (n_heads * head_dim) // R).transpose(1, 2)

            q, k, v = to_relation_heads(q), to_relation_heads(k), to_relation_heads(v)
            out.append((
                _masked_softmax(q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5, mask),
                _masked_softmax(v @ v.transpose(-1, -2) / v.shape[-1] ** 0.5, mask),
            ))
        return out


def _masked_softmax(scores, mask):
    # causal + padding mask, so rows never attend to padding or to the future
    L = scores.shape[-1]
    causal = torch.ones(L, L, dtype=torch.bool, device=scores.device).tril()
    keep = causal & mask[:, None, None, :].bool()
    scores = scores.masked_fill(~keep, torch.finfo(scores.dtype).min)
    return scores.float().softmax(-1)


def load_student(args, teacher=None):
    """Loads the student. LoRA is merged back into the base weights before saving,
    so checkpoints always reload as a plain AutoModelForCausalLM."""
    student = AutoModelForCausalLM.from_pretrained(
        args.SMALL_MODEL_ID,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    student.config.use_cache = False

    if args.FINETUNE_MODE == "lora":
        from peft import LoraConfig, get_peft_model

        student = get_peft_model(student, LoraConfig(
            r=args.LORA_R,
            lora_alpha=args.LORA_ALPHA,
            lora_dropout=args.LORA_DROPOUT,
            target_modules=list(args.LORA_TARGET_MODULES),
            task_type="CAUSAL_LM",
        ))
    elif args.FINETUNE_MODE != "full":
        raise ValueError(f"FINETUNE_MODE must be 'full' or 'lora', got {args.FINETUNE_MODE!r}")

    return student


def save_student(student, tokenizer, save_dir):
    """Saves a standalone HF model that AutoModelForCausalLM.from_pretrained can load."""
    model = student
    if hasattr(model, "merge_and_unload"):
        # merge a copy: merge_and_unload() strips the adapters, which would leave the
        # live model with nothing trainable for the rest of training
        model = copy.deepcopy(student).merge_and_unload()

    use_cache = model.config.use_cache
    model.config.use_cache = True
    model.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)
    model.config.use_cache = use_cache
