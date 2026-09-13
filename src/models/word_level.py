import copy

import torch
from transformers import AutoModelForCausalLM


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
