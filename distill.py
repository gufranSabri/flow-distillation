import os
import yaml
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.logger import Logger
from utils.utils import set_rng_state
from utils.data import build_dolly_datasets

from src.models.word_level import load_model
from src.trainers.word_level import WordLevelTrainer


COMMON_CONFIG = "configs/common.yaml"


def check_same_vocab(args, tokenizer, teacher_tokenizer):
    """Word-level KD needs a shared vocabulary; mismatched vocabs are a hard stop
    rather than something to pad or truncate around (phase1.md 0)."""
    if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError(
            f"Teacher ({args.teacher_model}) and student ({args.student_model}) do not "
            "share a tokenizer/vocabulary, so word-level KD does not apply."
        )


def prep_model_comps(args):
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
    student = load_model(args, args.student_model).to(args.device)

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
    train_ds, val_ds, collator = build_dolly_datasets(args, tokenizer)

    trainer = WordLevelTrainer(
        args, student, teacher, tokenizer, train_ds, val_ds, collator
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-model", required=True,
                         help="hub id or path (e.g. a pretraining.py SFT checkpoint) for the student")
    parser.add_argument("--teacher-model", required=True,
                         help="hub id or path (e.g. a pretraining.py SFT checkpoint) for the teacher")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--config", default="configs/word_level.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=None,
                         help="overrides TRAIN_EPOCHS from the config")

    args = parser.parse_args()
    for key, value in load_config(args.config).items():
        setattr(args, key, value)
    if args.epochs is not None:
        args.TRAIN_EPOCHS = args.epochs

    # --work-dir is resolved under WORK_DIR_ROOT unless given as an explicit path
    root = os.path.expanduser(args.WORK_DIR_ROOT)
    args.work_dir = os.path.join(root, args.work_dir or args.APPROACH)

    main(args)
