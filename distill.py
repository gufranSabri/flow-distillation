import os
import yaml
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.logger import Logger, log_config, log_model, log_dataset_sizes
from utils.utils import set_rng_state, parse_cli_overrides
from utils.data import build_dolly_datasets

from src.distill import get_approach


COMMON_CONFIG = "configs/common.yaml"


def check_same_vocab(args, tokenizer, teacher_tokenizer):
    # word-level KD requires teacher and student to share a vocabulary
    if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError(
            f"Teacher ({args.teacher_model}) and student ({args.student_model}) do not "
            "share a tokenizer/vocabulary, so word-level KD does not apply."
        )


def prep_model_comps(args, approach):
    args.logger("Loading tokenizers …")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    student_tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    check_same_vocab(args, student_tokenizer, tokenizer)

    args.logger(f"Loading teacher: {args.teacher_model} …")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(args.device)
    teacher.config.use_cache = False
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    args.logger(f"Loading student ({args.FINETUNE_MODE}): {args.student_model} …")
    student = approach.load_model(args, args.student_model).to(args.device)

    args.logger(f"  Teacher hidden dim {teacher.config.hidden_size} / "
                f"student hidden dim {student.config.hidden_size}")
    log_model(args.logger, teacher, "teacher")
    log_model(args.logger, student, "student")

    return tokenizer, teacher, student


def load_config(path):
    """Load common.yaml, then merge the chosen config on top (chosen config wins)."""
    merged = {}
    for cfg_path in (COMMON_CONFIG, path):
        with open(cfg_path) as f:
            merged.update(yaml.safe_load(f) or {})
    return merged


def main(args, approach):
    os.makedirs(args.work_dir, exist_ok=True)
    set_rng_state(args.seed)
    args.logger = Logger(os.path.join(args.work_dir, f"{args.approach}.log"))
    args.logger(f"Work dir: {args.work_dir}", console_print=True)
    log_config(args.logger, args)

    tokenizer, teacher, student = prep_model_comps(args, approach)
    train_ds, val_ds, collator = build_dolly_datasets(args, tokenizer)
    log_dataset_sizes(args.logger, train_ds, val_ds)

    trainer = approach.Trainer(
        args, student, teacher, tokenizer, train_ds, val_ds, collator
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        epilog="Any key from configs/common.yaml or configs/distill/<approach>.yaml can "
               "also be overridden, e.g. --DISTILL_LOSS both --TEMPERATURE 2.0.",
    )
    parser.add_argument("--student-model", required=True,
                         help="hub id or path (e.g. a finetune.py SFT checkpoint) for the student")
    parser.add_argument("--teacher-model", required=True,
                         help="hub id or path (e.g. a finetune.py SFT checkpoint) for the teacher")
    parser.add_argument("--approach", default="word_level",
                         help="which src/distill/<name> config/model/trainer trio to distill "
                              "the student with; its config is configs/distill/<approach>.yaml "
                              "(see src/distill/); the teacher is always loaded as a plain HF "
                              "model, as before")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--device", default="cuda")

    args, unknown = parser.parse_known_args()
    approach = get_approach(args.approach)
    config = load_config(f"configs/distill/{args.approach}.yaml")
    config.update(parse_cli_overrides(unknown))  # e.g. --DISTILL_LOSS both --TRAIN_EPOCHS 3
    for key, value in config.items():
        setattr(args, key, value)

    # --work-dir is resolved under WORK_DIR_ROOT unless given as an explicit path
    root = os.path.expanduser(args.WORK_DIR_ROOT)
    args.work_dir = os.path.join(root, args.work_dir or args.approach)

    main(args, approach)
