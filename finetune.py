import os
import yaml
import argparse

from transformers import AutoTokenizer

from utils.logger import Logger, log_config, log_model, log_dataset_sizes
from utils.utils import set_rng_state, parse_cli_overrides
from utils.data import build_dolly_datasets

from src.finetune import get_approach


COMMON_CONFIG = "configs/common.yaml"


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

    args.logger(f"Loading tokenizer/model ({args.FINETUNE_MODE}): {args.model} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = approach.load_model(args, args.model).to(args.device)
    log_model(args.logger, model, "model")

    train_ds, val_ds, collator = build_dolly_datasets(args, tokenizer)
    log_dataset_sizes(args.logger, train_ds, val_ds)

    trainer = approach.Trainer(args, model, tokenizer, train_ds, val_ds, collator)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        epilog="Any key from configs/common.yaml or configs/finetune/<approach>.yaml can "
               "also be overridden, e.g. --LR 1e-4 --WEIGHT_DECAY 0.01.",
    )
    parser.add_argument("--model", required=True,
                         help="hub id or path of the single model to SFT on Dolly "
                              "(e.g. the hub id that would otherwise go to distill.py's "
                              "--student-model/--teacher-model, to cache its SFT baseline)")
    parser.add_argument("--approach", default="vanilla",
                         help="which src/finetune/<name> config/model/trainer trio to "
                              "finetune with; its config is configs/finetune/<approach>.yaml "
                              "(see src/finetune/)")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--device", default="cuda")

    args, unknown = parser.parse_known_args()
    approach = get_approach(args.approach)
    config = load_config(f"configs/finetune/{args.approach}.yaml")
    config.update(parse_cli_overrides(unknown))  # e.g. --LR 1e-4 --TRAIN_EPOCHS 3
    for key, value in config.items():
        setattr(args, key, value)

    # --work-dir is resolved under WORK_DIR_ROOT unless given as an explicit path
    root = os.path.expanduser(args.WORK_DIR_ROOT)
    args.work_dir = os.path.join(root, args.work_dir or args.approach)

    main(args, approach)
