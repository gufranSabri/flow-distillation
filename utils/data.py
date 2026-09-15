import random

from tqdm import tqdm
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

        category = ex.get("category")
        if category in self.MAGPIE_EXCLUDE_CATS:
            return []

        examples = []
        for idx in range(1, len(messages)):
            context  = messages[:idx]
            response = messages[idx]

            if response["role"] != "assistant":
                continue

            # hardcoded: eval runs with --apply_chat_template, so training must match that formatting
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

            # labels are pre-shifted: labels[t] == input_ids[t+1], with a trailing eos as the final target
            input_ids = context_ids + response_ids
            labels    = [-100] * len(context_ids[1:]) + response_ids + [self.tokenizer.eos_token_id]

            examples.append({
                "input_ids": input_ids,
                "labels":    labels,
            })

        return examples


class DollyProcessor(DataProcessor):
    """MiniLLM's Alpaca-template Dolly recipe, replicated exactly (no chat template, even though Qwen is chat-tuned)."""

    def line2data(self, indexed_example: tuple) -> list:
        _, ex = indexed_example
        prompt = format_dolly_prompt(ex["instruction"], ex["context"])

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_ids) > DOLLY_MAX_PROMPT_LEN:
            return []

        full_ids = self.tokenizer.encode(prompt + ex["response"], add_special_tokens=False)
        response_ids = full_ids[len(prompt_ids):]

        if len(prompt_ids) + len(response_ids) + 1 > self.config.MAX_LENGTH:
            return []

        # same pre-shifted-labels convention as SmolTalkProcessor.line2data
        input_ids = prompt_ids + response_ids
        labels    = [-100] * len(prompt_ids[1:]) + response_ids + [self.tokenizer.eos_token_id]

        return [{"input_ids": input_ids, "labels": labels}]


DOLLY_DATASET_ID      = "databricks/databricks-dolly-15k"
DOLLY_DEV_NUM_DEFAULT = 1000   # MiniLLM: first N raw rows -> valid, rest -> train
DOLLY_MAX_PROMPT_LEN  = 256    # MiniLLM tools/process_data_dolly.py --max-prompt-length

_DOLLY_TEMPLATE_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)
_DOLLY_TEMPLATE_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)


def format_dolly_prompt(instruction: str, input_text: str) -> str:
    """MiniLLM's Alpaca template, shared with dolly_eval so train/eval prompts never drift apart."""
    if not input_text:
        return _DOLLY_TEMPLATE_NO_INPUT.format(instruction=instruction)
    return _DOLLY_TEMPLATE_WITH_INPUT.format(instruction=instruction, input=input_text)


def load_dolly_splits(dataset_id=DOLLY_DATASET_ID, dev_num=DOLLY_DEV_NUM_DEFAULT):
    """First `dev_num` rows (natural order) -> valid, rest -> train, matching MiniLLM's split."""
    full = load_dataset(dataset_id, split="train")
    return full.select(range(dev_num, len(full))), full.select(range(dev_num))


def prepare_tokenizer(tokenizer, logger=None, require_chat_template=True):
    log = logger or (lambda msg: None)

    if require_chat_template and tokenizer.chat_template is None:
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
        # several base tokenizers (Qwen2, Llama) ship without one; fall back to eos
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


def build_dolly_datasets(args, tokenizer):
    args.logger("Building Dolly datasets...")

    tokenizer = prepare_tokenizer(tokenizer, logger=args.logger, require_chat_template=False)

    train_raw, val_raw = load_dolly_splits(
        dataset_id=getattr(args, "DATASET_ID", DOLLY_DATASET_ID),
        dev_num=getattr(args, "DOLLY_DEV_NUM", DOLLY_DEV_NUM_DEFAULT),
    )
    processor = DollyProcessor(config=args, tokenizer=tokenizer, filepath=args.DATASET_ID)
    processor.initializer()

    def process(raw_dataset, label):
        out = []
        for item in tqdm(
            enumerate(raw_dataset),
            desc=f"Tokenizing Dolly {label}",
            total=len(raw_dataset),
        ):
            out.extend(processor.line2data(item))
        return out

    train_data = process(train_raw, "train")
    val_data   = process(val_raw, "val")

    if args.MAX_TRAIN_SAMPLES != -1:
        random.shuffle(train_data)
        train_data = train_data[:args.MAX_TRAIN_SAMPLES]
    if args.MAX_VAL_SAMPLES != -1:
        random.shuffle(val_data)
        val_data = val_data[:args.MAX_VAL_SAMPLES]

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

