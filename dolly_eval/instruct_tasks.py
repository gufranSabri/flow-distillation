"""Loaders for MiniLLM's instruction-following eval suite (docs/repos/minillm README's
"DollyEval / SelfInst / VicunaEval / S-NI / UnNI" list). Each loader returns a list of
{"prompt": str, "reference": list[str]} examples.

Dolly is built from utils.data so training and eval prompts never drift apart (see
docs/plans/dolly-migration-plan.md). The other four ship their prompts pre-baked in
MiniLLM's own Alpaca template on the HF Hub (verified against docs/repos/minillm's
data_utils/prompt_datasets.py, which just reads the "prompt"/"output" fields directly),
so we use those jsonl files as-is rather than reformatting them ourselves.
"""
import json
import random

from huggingface_hub import hf_hub_download

from utils.data import format_dolly_prompt, load_dolly_splits

_SUBSAMPLE_SEED = 42

# MiniLLM buckets S-NI/UnNI by reference-length for its own reporting; we just pool
# every bucket back into one eval set.
_SNI_BUCKETS = ["0_2", "3_6", "6_10", "11_"]
_UINST_BUCKETS = ["0_2", "3_5", "6_10", "11_"]
_UINST_TARGET_N = 10000  # MiniLLM: "10,000 randomly drawn from the core set"


def _load_jsonl(repo_id, filename):
    path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename)
    with open(path) as f:
        return [json.loads(line) for line in f]


def _as_reference(output):
    return output if isinstance(output, list) else [output]


def _rows_to_examples(rows):
    return [{"prompt": r["prompt"], "reference": _as_reference(r["output"])} for r in rows]


def load_dolly_examples(args):
    _, val_raw = load_dolly_splits(dataset_id=args.dolly_dataset_id, dev_num=args.dolly_dev_num)
    return [
        {"prompt": format_dolly_prompt(ex["instruction"], ex["context"]), "reference": [ex["response"]]}
        for ex in val_raw
    ]


def load_self_inst_examples(args):
    return _rows_to_examples(_load_jsonl("MiniLLM/self-inst", "valid.jsonl"))


def load_vicuna_examples(args):
    return _rows_to_examples(_load_jsonl("MiniLLM/Vicuna", "valid.jsonl"))


def load_s_ni_examples(args):
    rows = [r for bucket in _SNI_BUCKETS for r in _load_jsonl("MiniLLM/sinst", f"{bucket}/valid.jsonl")]
    return _rows_to_examples(rows)


def load_u_inst_examples(args):
    rows = [r for bucket in _UINST_BUCKETS for r in _load_jsonl("MiniLLM/uinst", f"{bucket}/valid.jsonl")]
    if len(rows) > _UINST_TARGET_N:
        rows = random.Random(_SUBSAMPLE_SEED).sample(rows, _UINST_TARGET_N)
    return _rows_to_examples(rows)


TASKS = {
    "dolly": load_dolly_examples,
    "self_inst": load_self_inst_examples,
    "vicuna": load_vicuna_examples,
    "s_ni": load_s_ni_examples,
    "u_inst": load_u_inst_examples,
}
