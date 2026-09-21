"""Loading for fm_lora (docs/fm_lora.md): both teacher and student are Stage-1 SFT LoRA
models (src/finetune/vanilla, adapter saved under <stage1_dir>/adapter/ -- see its
save_model). Stage 2 needs the *unmerged* adapter, not the merged checkpoint, because:
  - the teacher's last-layer A_t/B_t must stay separately addressable (only A_t is used,
    to compute the flow target z_t) while the rest of its LoRA stays frozen but intact;
  - the student's Stage-1 LoRA must continue training as a LoRA delta, not be re-learned
    on top of a fused base.

"Last layer" throughout this module means the last transformer decoder block (index
num_hidden_layers - 1), not lm_head -- fm_lora never touches lm_head. Each
LORA_TARGET_MODULES projection that lives in that one block (e.g. o_proj, down_proj)
is treated independently: its own A/B/v_theta, flow-matched using that projection's
own input.
"""
import copy
import os

import torch
from transformers import AutoModelForCausalLM
from peft import PeftConfig, PeftModel

from .flow import FMLoraSite, ADAPTER_NAME


def _adapter_dir(stage1_dir):
    adapter_dir = os.path.join(stage1_dir, "adapter")
    if not os.path.isdir(adapter_dir):
        raise ValueError(
            f"{stage1_dir!r} has no adapter/ subdir -- fm_lora needs a Stage-1 checkpoint "
            "saved by src/finetune/vanilla with FINETUNE_MODE=lora (its save_model saves "
            "the unmerged adapter alongside the merged model)."
        )
    return adapter_dir


def _load_lora_model(stage1_dir, device, trainable):
    """The *original* pretrained base (not stage1_dir itself, which is Stage 1's merged
    output -- loading that as the base and then applying adapter_dir on top would double
    -apply the Stage-1 LoRA delta, since stage1_dir already has it baked in) + the
    Stage-1 LoRA adapter, unmerged.

    The original base's id is read from the adapter's own config (peft records it as
    base_model_name_or_path when the adapter is saved), not from stage1_dir.
    """
    adapter_dir = _adapter_dir(stage1_dir)
    peft_config = PeftConfig.from_pretrained(adapter_dir)

    base = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path, dtype=torch.bfloat16, trust_remote_code=True,
    )
    base.config.use_cache = False
    model = PeftModel.from_pretrained(
        base, adapter_dir, adapter_name=ADAPTER_NAME, is_trainable=trainable,
    ).to(device)
    return model


def last_layer_sites(model, target_modules=None):
    """Returns {name: (parent_module, attr_name, module)} for every target_modules
    entry found as a direct child of the model's last decoder block's self_attn/mlp
    submodules (e.g. {"o_proj": (self_attn, "o_proj", <LoraLayer or FMLoraSite>)}).

    target_modules=None matches any target-module child found there (used to locate
    already-swapped FMLoraSites, whose names aren't known ahead of time by the
    caller); otherwise restricts to exactly that set and raises if any are missing --
    used right after loading Stage 1, before anything has been swapped, to confirm
    every configured projection actually has LoRA in the last block.
    """
    last_idx = model.config.num_hidden_layers - 1
    suffixes = (f"layers.{last_idx}.self_attn", f"layers.{last_idx}.mlp")
    sites = {}
    for parent_name, parent_module in model.named_modules():
        if not parent_name.endswith(suffixes):
            continue
        for attr, child in parent_module.named_children():
            if target_modules is None or attr in target_modules:
                sites[attr] = (parent_module, attr, child)

    if target_modules is not None:
        missing = set(target_modules) - set(sites)
        if missing:
            raise ValueError(
                f"Last decoder block (layer {last_idx}) is missing LoRA at {sorted(missing)}; "
                f"fm_lora requires every entry in target_modules ({sorted(target_modules)}) "
                "to have been LoRA-adapted there in Stage 1."
            )
    return sites


def load_teacher(args, stage1_dir):
    """Teacher: frozen base + all Stage-1 LoRA (incl. last layer, never swapped). Only
    each last-layer site's A_t (via last_layer_sites) is used downstream, to compute
    the flow target -- B_t is never used to decode."""
    model = _load_lora_model(stage1_dir, args.device, trainable=False)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def load_model(args, stage1_dir):
    """Student: Stage-1 LoRA, only the LoRA (+ fm_lora sites, added below) trainable --
    the base stays frozen, exactly as _load_lora_model's is_trainable=True already set
    it up. In flow_matching mode, every target-module LoraLayer in the last decoder
    block is replaced by an FMLoraSite (A_s/B_s kept from Stage 1, trainable, plus a
    new v_theta); every other layer's LoRA is untouched, plain trainable peft LoRA. In
    standard mode nothing is swapped -- the last layer's LoRA stays normal, same as
    every other layer.

    If stage1_dir is itself an fm_lora checkpoint (i.e. this is a resume -- see
    distill.py's find_latest_checkpoint/model_source), each site's v_theta is restored
    from its fm_sites.pt instead of being freshly initialized, so a resumed run
    continues Stage 2's flow-matching progress rather than silently discarding it.
    """
    model = _load_lora_model(stage1_dir, args.device, trainable=True)

    if args.distillation_mode == "flow_matching":
        target_modules = model.peft_config[ADAPTER_NAME].target_modules
        sites = last_layer_sites(model, target_modules)

        sites_path = os.path.join(stage1_dir, "fm_sites.pt")
        saved = torch.load(sites_path, map_location=args.device, weights_only=True) if os.path.isfile(sites_path) else None
        num_euler_steps = saved["num_euler_steps"] if saved else args.num_euler_steps

        fm_sites = {}
        for name, (parent, attr, lora_layer) in sites.items():
            hidden = saved["sites"][name]["hidden"] if saved else args.FM_VELOCITY_HIDDEN
            site = FMLoraSite(lora_layer, hidden=hidden, num_euler_steps=num_euler_steps)
            if saved:
                # resume: restore this site's own evolved A_s/B_s/v_theta, not the
                # fresh Stage-1 A/B that FMLoraSite.__init__ just copied from lora_layer
                site.v_theta.load_state_dict(saved["sites"][name]["velocity"])
                site.A.load_state_dict(saved["sites"][name]["A"])
                site.B.load_state_dict(saved["sites"][name]["B"])
            site = site.to(device=args.device, dtype=torch.bfloat16)

            setattr(parent, attr, site)
            fm_sites[name] = site

        model.fm_sites = fm_sites
        model.fm_num_euler_steps = num_euler_steps
    elif args.distillation_mode != "standard":
        raise ValueError(
            f"distillation_mode must be 'standard' or 'flow_matching', got {args.distillation_mode!r}"
        )

    return model


def save_model(model, tokenizer, save_dir):
    """Saves the student: adapter/ (all Stage-2 LoRA except the swapped last-layer
    sites) + fm_sites.pt (flow_matching mode only: each site's A_s/B_s/v_theta,
    keyed by projection name) + a merged standalone model.

    The swapped sites' A_s/B_s must be saved explicitly in fm_sites.pt, not left to
    PeftModel.save_pretrained(adapter_dir): that call walks the state dict for
    "lora_"-prefixed keys, but FMLoraSite's own A/B submodules aren't named that way
    (they're plain attributes copied out of the original LoraLayer), so they'd
    otherwise be silently dropped from the saved adapter even though the spec has
    them "all trainable" in flow_matching mode.

    Merging: FMLoraSite has a base_layer attribute (like a real LoraLayer) but no
    merge() method, so handing it to merge_and_unload() as-is would crash. Swap each
    site back to its own frozen base_layer on a throwaway copy first, so
    merge_and_unload() only ever sees genuine LoraLayers (every non-last layer) and
    the merged model's last-layer target modules end up as the plain frozen Stage-1
    base -- the true inference behavior there is K-step Euler through v_theta (see
    .generate.load_for_generation), not any fixed weight composition.
    """
    model.save_pretrained(os.path.join(save_dir, "adapter"))

    to_merge = model
    if hasattr(model, "fm_sites"):
        torch.save(
            {"num_euler_steps": model.fm_num_euler_steps,
             "sites": {
                 name: {"velocity": site.v_theta.state_dict(),
                        "A": site.A.state_dict(), "B": site.B.state_dict(),
                        "rank": site.v_theta.rank,
                        "hidden": site.v_theta.net[0].out_features}
                 for name, site in model.fm_sites.items()
             }},
            os.path.join(save_dir, "fm_sites.pt"),
        )

        to_merge = copy.deepcopy(model)
        for _, (parent, attr, site) in last_layer_sites(to_merge).items():
            if isinstance(site, FMLoraSite):
                setattr(parent, attr, site.base_layer)

    merged = to_merge.merge_and_unload(progressbar=False)
    use_cache = merged.config.use_cache
    merged.config.use_cache = True
    merged.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)
    merged.config.use_cache = use_cache
