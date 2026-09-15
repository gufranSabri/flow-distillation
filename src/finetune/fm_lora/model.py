import copy
import os
from types import SimpleNamespace

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from .flow import FMLoRALinear, set_mode


def _iter_target_linears(model, target_modules):
    """Yields (parent, attr_name, module) for every nn.Linear whose name matches target_modules."""
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if not isinstance(child, nn.Linear):
                continue
            if child_name in target_modules:
                yield module, child_name, child


def inject_fm_lora(model, args):
    """Replaces every targeted nn.Linear with an FMLoRALinear wrapping the frozen original."""
    sites = []
    for parent, attr_name, linear in list(_iter_target_linears(model, set(args.LORA_TARGET_MODULES))):
        site = FMLoRALinear(
            linear,
            hidden=args.FM_VELOCITY_HIDDEN,
            time_dim=args.FM_TIME_EMBED_DIM,
            rank=args.FM_COND_RANK,
            normalize=args.FM_NORMALIZE_TARGET,
            steps=args.FM_INFER_STEPS,
        )
        setattr(parent, attr_name, site)
        sites.append(site)

    if not sites:
        raise ValueError(
            f"No modules matched LORA_TARGET_MODULES={sorted(args.LORA_TARGET_MODULES)}; "
            "nothing to adapt."
        )
    return sites


def collect_sites(model):
    return [m for m in model.modules() if isinstance(m, FMLoRALinear)]


def load_model(args, model_id):
    """Loads model_id with its targeted linears replaced by flow-matching LoRA sites.

    The base weights stay frozen exactly as in standard LoRA; only each site's
    velocity network trains.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    for p in model.parameters():
        p.requires_grad = False

    inject_fm_lora(model, args)
    return model


VELOCITY_FILE = "fm_lora_velocity.pt"


def is_fm_lora_checkpoint(path):
    return os.path.isfile(os.path.join(path, VELOCITY_FILE))


def load_fm_lora_checkpoint(path, device="cuda", dtype=torch.bfloat16):
    """Rebuilds a trained FM-LoRA model from a save_model() directory.

    The saved weights are the FROZEN BASE only; the trained flow lives in
    fm_lora_velocity.pt. Loading the dir with a plain AutoModelForCausalLM silently
    gives the unadapted base model, so benchmarking must come through here.
    """
    state = torch.load(os.path.join(path, VELOCITY_FILE), weights_only=True, map_location="cpu")
    if "hparams" not in state or "sites" not in state:
        raise ValueError(
            f"{path}/{VELOCITY_FILE} predates the reloadable checkpoint format; "
            "it has no hparams/sites and the flow cannot be rebuilt."
        )
    hp = state["hparams"]

    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, trust_remote_code=True)
    model.config.use_cache = True
    for p in model.parameters():
        p.requires_grad = False

    inject_fm_lora(model, SimpleNamespace(**hp))

    by_name = {n: m for n, m in model.named_modules() if isinstance(m, FMLoRALinear)}
    missing = set(state["sites"]) - set(by_name)
    if missing:
        raise ValueError(f"checkpoint has {len(missing)} site(s) absent from the rebuilt model, "
                         f"e.g. {sorted(missing)[:3]}")

    for name, saved in state["sites"].items():
        site = by_name[name]
        site.velocity.load_state_dict(saved["velocity"])
        site.target_rms.copy_(saved["target_rms"].to(site.target_rms.device))

    model = model.to(device)
    model.eval()
    set_mode(collect_sites(model), "flow")
    return model


def save_model(model, tokenizer, save_dir):
    """Saves a standalone HF model by baking each site's flow output back into a plain Linear.

    A site's flow is input-dependent, so unlike LoRA there is no exact weight merge.
    We save the frozen base weights (flow disabled) plus the velocity networks and the
    hyperparameters needed to rebuild the sites, so load_fm_lora_checkpoint can restore
    the trained flow. Loading the dir with a plain AutoModelForCausalLM yields the
    UNADAPTED base model.
    """
    sites = [(n, m) for n, m in model.named_modules() if isinstance(m, FMLoRALinear)]
    if not sites:
        raise ValueError("save_model called on a model with no FM-LoRA sites")

    ref = sites[0][1]
    velocity_state = {
        "format": 1,
        # needed to reconstruct the modules before their weights can be loaded
        "hparams": {
            "FM_VELOCITY_HIDDEN": ref.velocity.hidden,
            "FM_TIME_EMBED_DIM": ref.velocity.time_dim,
            "FM_COND_RANK": ref.velocity.rank,
            "FM_NORMALIZE_TARGET": ref.normalize,
            "FM_INFER_STEPS": ref.steps,
            "LORA_TARGET_MODULES": sorted({n.rsplit(".", 1)[-1] for n, _ in sites}),
        },
        "sites": {
            name: {
                "velocity": module.velocity.state_dict(),
                "target_rms": module.target_rms.detach().cpu(),
            }
            for name, module in sites
        },
    }

    # sites hold non-leaf scratch tensors between passes (x1 targets, the CFM loss and
    # its graph); deepcopy refuses those, so drop them before copying
    for module in model.modules():
        if isinstance(module, FMLoRALinear):
            module.target = None
            module.loss = None
            module.loss_mask = None

    stripped = copy.deepcopy(model)
    for parent_name, module in list(stripped.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, FMLoRALinear):
                setattr(module, child_name, child.base)

    use_cache = stripped.config.use_cache
    stripped.config.use_cache = True
    stripped.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)
    stripped.config.use_cache = use_cache

    torch.save(velocity_state, f"{save_dir}/fm_lora_velocity.pt")
