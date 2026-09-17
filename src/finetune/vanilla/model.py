import copy
import os

import torch
from transformers import AutoModelForCausalLM


def load_model(args, model_id):
    """Loads model_id for training, applying LoRA if configured."""
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    if args.FINETUNE_MODE == "lora":
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(model, LoraConfig(
            r=args.LORA_R,
            lora_alpha=args.LORA_ALPHA,
            lora_dropout=args.LORA_DROPOUT,
            target_modules=list(args.LORA_TARGET_MODULES),
            task_type="CAUSAL_LM",
        ))
    elif args.FINETUNE_MODE != "full":
        raise ValueError(f"FINETUNE_MODE must be 'full' or 'lora', got {args.FINETUNE_MODE!r}")

    return model


def save_model(model, tokenizer, save_dir):
    """Saves a standalone HF model that AutoModelForCausalLM.from_pretrained can load.

    If model is a PEFT model, the unmerged adapter is also saved (to save_dir/adapter/,
    via PeftModel.save_pretrained) so a later stage that needs the separate LoRA
    matrices -- not just their fused effect on the base weights -- can reload them with
    PeftModel.from_pretrained (see src/distill/fm_lora/model.py, which needs a Stage-1
    checkpoint's A/B intact rather than merged into the base).
    """
    if hasattr(model, "peft_config"):
        model.save_pretrained(os.path.join(save_dir, "adapter"))

    out = model
    if hasattr(out, "merge_and_unload"):
        # merge a copy so the live model keeps its adapters for the rest of training
        out = copy.deepcopy(model).merge_and_unload()

    use_cache = out.config.use_cache
    out.config.use_cache = True
    out.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)
    out.config.use_cache = use_cache
