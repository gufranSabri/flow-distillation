import os
import yaml
import argparse

from transformers import AutoTokenizer

from utils.logger import Logger
from utils.utils import set_rng_state
from utils.data import build_dolly_datasets

from src.models.word_level import load_model
from src.trainers.pretrain import PretrainTrainer


COMMON_CONFIG = "configs/common.yaml"


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

    args.logger(f"Loading tokenizer/model ({args.FINETUNE_MODE}): {args.model} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_model(args, args.model).to(args.device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    args.logger(f"  Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)\n")

    train_ds, val_ds, collator = build_dolly_datasets(args, tokenizer)

    trainer = PretrainTrainer(args, model, tokenizer, train_ds, val_ds, collator)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                         help="hub id or path of the single model to SFT on Dolly "
                              "(e.g. the hub id that would otherwise go to distill.py's "
                              "--student-model/--teacher-model, to cache its SFT baseline)")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--config", default="configs/pretrain.yaml")
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
