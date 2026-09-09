import os
from tqdm import tqdm
import random

from pprint import pprint
from datasets import load_dataset, Dataset
from transformers import DataCollatorForSeq2Seq


class DataProcessor:
    def __init__(self, config, tokenizer, filepath, **kwargs):
        self.config    = config
        self.tokenizer = tokenizer
        self.filepath  = filepath

    def initializer(self):
        pass

    def line2data(self, indexed_example: tuple) -> list | None:
        raise NotImplementedError
    

class SmolTalkProcessor(DataProcessor):
    MAGPIE_SUBSET = "smol-magpie-ultra"
    MAGPIE_EXCLUDE_CATS = {
        "advice-seeking", "brainstorming", "creative-writing",
        "editing", "planning", "role-playing",
    }

    def line2data(self, indexed_example: tuple) -> list:
        _, ex = indexed_example
        messages = ex["messages"]

        # Category filter (hardcoded)
        category = ex.get("category")
        if category in self.MAGPIE_EXCLUDE_CATS:
            return []

        examples = []
        for idx in range(1, len(messages)):
            context  = messages[:idx]
            response = messages[idx]

            if response["role"] != "assistant":
                continue

            # Always render via the tokenizer's chat template: evaluation
            # (lm-evaluation-harness) is run with --apply_chat_template, so
            # training inputs must match that formatting exactly, hardcoded —
            # a raw/no-template split here would be out-of-distribution at
            # eval time, not closer to it.
            context_ids = self.tokenizer.apply_chat_template(
                context,
                tokenize=True,
                add_generation_prompt=True,
            )
            if not isinstance(context_ids, list):
                context_ids = context_ids["input_ids"]

            response_ids = self.tokenizer.encode(
                response["content"],
                add_special_tokens=False,
            )

            if len(context_ids) + len(response_ids) > self.config.MAX_LENGTH:
                break

            # Build input_ids and labels.
            # NOTE: labels are deliberately PRE-SHIFTED left by one relative to
            # input_ids so that labels[t] == input_ids[t+1] (the token predicted by
            # the logits at position t), with a trailing eos as the final target.
            input_ids = context_ids + response_ids
            labels    = [-100] * len(context_ids[1:]) + response_ids + [self.tokenizer.eos_token_id]

            examples.append({
                "input_ids": input_ids,
                "labels":    labels,
            })

        return examples


def prepare_tokenizer(tokenizer, logger=None):
    """Validate/complete the special tokens needed downstream, regardless of
    which model the tokenizer belongs to.

    - A chat template is required by SmolTalkProcessor.line2data (apply_chat_template).
    - eos_token_id is required to terminate labels (see line2data); some base
      tokenizers leave it unset.
    - pad_token is required by DataCollatorForSeq2Seq at collation time; several
      base models (Qwen2, Llama) ship without one, so we fall back to eos.
    """
    log = logger or (lambda msg: None)

    if tokenizer.chat_template is None:
        raise ValueError(
            f"Tokenizer {tokenizer.name_or_path!r} has no chat_template. "
            "SmolTalkProcessor requires an instruction-tuned tokenizer/model."
        )

    if tokenizer.eos_token_id is None:
        raise ValueError(
            f"Tokenizer {tokenizer.name_or_path!r} has no eos_token_id set; "
            "labels cannot be terminated without one."
        )

    if tokenizer.pad_token_id is None:
        log(f"  Tokenizer {tokenizer.name_or_path!r} has no pad token; "
            f"using eos_token ({tokenizer.eos_token!r}) as pad_token.")
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def build_datasets(args, tokenizer):
    args.logger("Building datasets...")

    tokenizer = prepare_tokenizer(tokenizer, logger=args.logger)

    use_all_train    = args.MAX_TRAIN_SAMPLES == -1
    n_val_per_subset = args.MAX_VAL_SAMPLES // len(args.DATASET_SUBSETS)

    if not use_all_train:
        total_needed       = args.MAX_TRAIN_SAMPLES + args.MAX_VAL_SAMPLES
        n_per_subset       = total_needed // len(args.DATASET_SUBSETS)
        n_train_per_subset = n_per_subset - n_val_per_subset

    subset_filepaths = [
        (f"{args.DATASET_ID},{subset},train", f"{args.DATASET_ID},{subset},test")
        for subset in args.DATASET_SUBSETS
    ]
    
    train_shards, val_shards = [], []
    for subset, (train_fp, test_fp) in zip(args.DATASET_SUBSETS, subset_filepaths):
        for split_fp, shard_list, n_target, label in [
            (train_fp, train_shards, None if use_all_train else n_train_per_subset, "train"),
            (test_fp,  val_shards,   n_val_per_subset,                              "val"),
        ]:
            dataset_id, sub, split = split_fp.split(",")
            _dataset = load_dataset(dataset_id, sub, split=split).shuffle(seed=42)

            if label == "train" and not use_all_train:
                _dataset = _dataset.select(range(min(n_target, len(_dataset))))
            elif label == "val":
                _dataset = _dataset.select(range(min(n_val_per_subset, len(_dataset))))

            processor = SmolTalkProcessor(config=args, tokenizer=tokenizer, filepath=split_fp)
            processor.initializer()

            dataset = []
            for item in tqdm(
                enumerate(_dataset),
                desc=f"Loading {label} data from {split_fp}",
                total=len(_dataset),
            ):
                result = processor.line2data(item)
                if result is not None:
                    dataset.extend(result)

            under_budget = len(dataset) < (n_target or 0) and label == "train"
            args.logger(
                f"  {subset} [{label}]: {len(dataset)} examples"
                + (" ⚠ subset smaller than target" if under_budget else "")
            )

            shard_list.append(dataset)

    def flatten_and_cap(shards, cap):
        flat = [ex for shard in shards for ex in shard]
        random.shuffle(flat)
        return flat[:cap] if cap != -1 else flat

    train_data = flatten_and_cap(train_shards, args.MAX_TRAIN_SAMPLES)
    val_data   = flatten_and_cap(val_shards,   args.MAX_VAL_SAMPLES)

    args.logger(f"  Total after length filter (≤{args.MAX_LENGTH} tokens): "
                f"{len(train_data)} train / {len(val_data)} val\n")

    # Wrap in HuggingFace Dataset for the collator
    train_tokenized = Dataset.from_list(train_data)
    val_tokenized   = Dataset.from_list(val_data)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=None,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    return train_tokenized, val_tokenized, data_collator




def load_config(path):
    import yaml

    COMMON_CONFIG = "configs/common.yaml"

    merged = {}
    for cfg_path in (COMMON_CONFIG, path):
        with open(cfg_path) as f:
            data = yaml.safe_load(f) or {}
        merged.update(data)
    return merged



if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer, AutoConfig
    from logger import Logger

    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default="./work_dir/test_data")
    parser.add_argument("--config", default="configs/mlp.yaml")
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    config = load_config(args.config)
    for key, value in config.items():
        setattr(args, key, value)

    setattr(args, "logger", Logger(os.path.join(args.work_dir, f"test_data.log"),))
    os.makedirs(args.work_dir, exist_ok=True)
    

    base_model_name = "Qwen/Qwen2.5-1.5B"
    config = AutoConfig.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    train_tokenized, val_tokenized, data_collator = build_datasets(args, tokenizer)

    # check some samples
    for i in range(1):
        print(f"Sample {i} from train dataset:")
        print("input_ids:", len(train_tokenized[i]["input_ids"]))
        print("labels:", len(train_tokenized[i]["labels"]))
        print("-" * 50)

        # detokenize (labels use -100 to mask out context positions from the loss,
        # which isn't a valid token id, so strip those before decoding)
        input_text = tokenizer.decode(train_tokenized[i]["input_ids"], skip_special_tokens=False)
        label_ids  = [tok for tok in train_tokenized[i]["labels"] if tok != -100]
        label_text = tokenizer.decode(label_ids, skip_special_tokens=False)

        print("-" * 50)
        print("Detokenized input:", input_text)
        print("-" * 50)
        print("Detokenized label:", label_text)
        print("-" * 50)

