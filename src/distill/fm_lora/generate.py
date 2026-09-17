"""Reconstructs an fm_lora flow_matching checkpoint (src/distill/fm_lora/model.save_model
output) into a plain nn.Module whose .generate() runs the spec's K-step Euler inference
at the last decoder block's target-module sites, so benchmark.py can drive it exactly
like any other checkpoint.
"""
import os

import torch

from .flow import FMLoraSite
from .model import last_layer_sites, ADAPTER_NAME


def is_fm_lora_checkpoint(path):
    return os.path.isfile(os.path.join(path, "fm_sites.pt"))


def load_for_generation(path, device="cuda", dtype=torch.bfloat16, num_steps=None):
    """Loads an fm_lora checkpoint directory for generation.

    path must contain: adapter/ (every non-swapped Stage-2 LoRA layer, applied on top
    of the *original* pretrained base -- not path's own merged model, which already
    has those deltas baked in; loading that as the base and reapplying adapter/ on
    top would double them) and fm_sites.pt (each last-layer site's A_s/B_s/v_theta,
    saved separately since PeftModel.save_pretrained never captured them -- see
    model.py's save_model). num_steps defaults to the K the checkpoint was saved with
    (args.num_euler_steps at train time).
    """
    if not is_fm_lora_checkpoint(path):
        raise ValueError(f"{path!r} has no fm_sites.pt; not an fm_lora flow_matching checkpoint")

    from transformers import AutoModelForCausalLM
    from peft import PeftConfig, PeftModel

    adapter_dir = os.path.join(path, "adapter")
    peft_config = PeftConfig.from_pretrained(adapter_dir)
    base = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path, dtype=dtype, trust_remote_code=True,
    )
    peft_model = PeftModel.from_pretrained(
        base, adapter_dir, adapter_name=ADAPTER_NAME, is_trainable=False,
    )

    saved = torch.load(os.path.join(path, "fm_sites.pt"), map_location="cpu", weights_only=True)
    num_steps = num_steps or saved["num_euler_steps"]

    sites = last_layer_sites(peft_model, saved["sites"].keys())
    for name, (parent, attr, lora_layer) in sites.items():
        site_state = saved["sites"][name]
        site = FMLoraSite(lora_layer, hidden=site_state["hidden"], num_euler_steps=num_steps)
        site.A.load_state_dict(site_state["A"])
        site.B.load_state_dict(site_state["B"])
        site.v_theta.load_state_dict(site_state["velocity"])
        site.mode = "infer"
        for p in site.parameters():
            p.requires_grad = False
        setattr(parent, attr, site.to(device=device, dtype=dtype))

    # unwrap: drop the outer PeftModel wrapper so the returned object is a plain
    # AutoModelForCausalLM-shaped model .generate() works on directly -- every other
    # layer's LoRA stays live via its (untouched) LoraLayer, only the last block's
    # target modules were swapped above
    model = peft_model.get_base_model()
    model.config.use_cache = True
    return model.to(device).eval()
