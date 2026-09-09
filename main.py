import os
import yaml
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.logger import Logger
from utils.utils import set_rng_state
from utils.data import build_datasets

from src.models.word_level import load_student
from src.trainers.word_level import WordLevelTrainer


COMMON_CONFIG = "configs/common.yaml"


def check_same_vocab(args, tokenizer, teacher_tokenizer):
    """Word-level KD needs a shared vocabulary; mismatched vocabs are a hard stop
    rather than something to pad or truncate around (phase1.md 0)."""
    if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError(
            f"Teacher ({args.LARGE_MODEL_ID}) and student ({args.SMALL_MODEL_ID}) do not "
            "share a tokenizer/vocabulary, so word-level KD does not apply."
        )


def prep_model_comps(args):
    args.logger("Loading tokenizers …")
    tokenizer = AutoTokenizer.from_pretrained(args.LARGE_MODEL_ID, trust_remote_code=True)
    student_tokenizer = AutoTokenizer.from_pretrained(args.SMALL_MODEL_ID, trust_remote_code=True)
    check_same_vocab(args, student_tokenizer, tokenizer)

    args.logger(f"Loading teacher: {args.LARGE_MODEL_ID} …")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.LARGE_MODEL_ID,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(args.device)
    teacher.config.use_cache = False
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    args.logger(f"Loading student ({args.FINETUNE_MODE}): {args.SMALL_MODEL_ID} …")
    student = load_student(args, teacher).to(args.device)

    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total = sum(p.numel() for p in student.parameters())
    args.logger(f"  Teacher hidden dim {teacher.config.hidden_size} / "
                f"student hidden dim {student.config.hidden_size}")
    args.logger(f"  Student trainable params: {trainable:,} / {total:,} "
                f"({100 * trainable / total:.2f}%)\n")

    return tokenizer, teacher, student


def load_config(path):
    """Load common.yaml, then merge the chosen config on top (chosen config wins)."""
    merged = {}
    for cfg_path in (COMMON_CONFIG, path):
        with open(cfg_path) as f:
            merged.update(yaml.safe_load(f) or {})
    return merged


def main(args):
    os.makedirs(args.work_dir, exist_ok=True)
    set_rng_state(args.seed)
    args.logger = Logger(os.path.join(args.work_dir, f"{args.APPROACH}.log"))
    args.logger(f"Work dir: {args.work_dir}", console_print=True)

    tokenizer, teacher, student = prep_model_comps(args)
    train_ds, val_ds, collator = build_datasets(args, tokenizer)

    trainer = WordLevelTrainer(
        args, student, teacher, tokenizer, train_ds, val_ds, collator
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--config", default="configs/word_level.yaml")
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()
    for key, value in load_config(args.config).items():
        setattr(args, key, value)

    # --work-dir is resolved under WORK_DIR_ROOT unless given as an explicit path
    root = os.path.expanduser(args.WORK_DIR_ROOT)
    args.work_dir = os.path.join(root, args.work_dir or args.APPROACH)

    main(args)
