"""Teacher state caching, shared by any distill approach that wants to avoid keeping the
teacher resident during training.

For a given teacher and dataset, the classifier-input hidden state (the state each
model's lm_head consumes, i.e. its last hidden state after the final norm) is run once
over the whole dataset and cached to <WORK_DIR_ROOT>/distillation/cache/<teacher>/. If
that cache already exists, the pass is skipped entirely. Logits are not cached -- they
are cheaply reconstructed at training time as (frozen teacher lm_head) @ hidden_state,
which is >>100x smaller to store than [T, vocab] logits per example (see the readout
already used by src/distill/dobi/model.py).

Cached rows are keyed by the tokenized dataset's row index (utils/data.py's
build_dolly_datasets stamps every example with a stable "idx" before any DataLoader
shuffling), so a shuffled training batch can look its teacher states back up by the
"idx" column the collator passes through unchanged.
"""
import hashlib
import os

import torch
from tqdm.auto import tqdm


def _safe_name(model_id):
    return model_id.replace(os.sep, "_").strip("_")


def cache_root(args, teacher_model_id):
    root = os.path.expanduser(args.WORK_DIR_ROOT)
    return os.path.join(root, "distillation", "cache", _safe_name(teacher_model_id))


def _shard_dir(root, split):
    return os.path.join(root, split)


def _manifest_path(root):
    return os.path.join(root, "manifest.pt")


def _fingerprint(args, dataset_len_train, dataset_len_val):
    # cache validity depends on exactly which rows/tokens were cached -- if the dataset
    # config changes (different DATASET_ID, split, or length cap), the fingerprint
    # changes and the cache is rebuilt rather than silently serving mismatched rows
    key = "|".join(str(x) for x in [
        getattr(args, "DATASET_ID", None),
        getattr(args, "DOLLY_DEV_NUM", None),
        getattr(args, "MAX_LENGTH", None),
        getattr(args, "MAX_TRAIN_SAMPLES", None),
        getattr(args, "MAX_VAL_SAMPLES", None),
        dataset_len_train,
        dataset_len_val,
    ])
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def is_cached(args, teacher_model_id, train_ds, val_ds):
    root = cache_root(args, teacher_model_id)
    manifest_path = _manifest_path(root)
    if not os.path.isfile(manifest_path):
        return False
    manifest = torch.load(manifest_path, map_location="cpu", weights_only=True)
    return manifest.get("fingerprint") == _fingerprint(args, len(train_ds), len(val_ds))


@torch.no_grad()
def build_cache(args, teacher, train_ds, val_ds, collator, device):
    """Runs `teacher` once over train_ds + val_ds and writes each row's classifier-input
    hidden state to <cache_root>/<split>/<idx>.pt, bf16, [T, teacher_dim] (no padding --
    only the row's real tokens). Safe to re-run: rows already on disk are skipped, so an
    interrupted build resumes instead of restarting."""
    root = cache_root(args, args.teacher_model)
    args.logger(f"Caching teacher states to {root} …", console_print=True)

    for split_name, ds in (("train", train_ds), ("val", val_ds)):
        shard_dir = _shard_dir(root, split_name)
        os.makedirs(shard_dir, exist_ok=True)

        loader = torch.utils.data.DataLoader(
            ds, batch_size=args.PER_DEVICE_EVAL_BATCH_SIZE, shuffle=False,
            collate_fn=collator, num_workers=2, pin_memory=True,
        )
        for batch in tqdm(loader, desc=f"Caching teacher [{split_name}]", unit="batch"):
            idx = batch["idx"].tolist()
            if all(os.path.isfile(os.path.join(shard_dir, f"{i}.pt")) for i in idx):
                continue  # whole batch already cached (resume)

            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)

            captured = {}
            hook = teacher.lm_head.register_forward_pre_hook(
                lambda _, inputs: captured.update(state=inputs[0])
            )
            try:
                teacher(input_ids=input_ids, attention_mask=attn_mask)
            finally:
                hook.remove()
            hidden = captured["state"].to(torch.bfloat16).cpu()

            for row, i in enumerate(idx):
                path = os.path.join(shard_dir, f"{i}.pt")
                if os.path.isfile(path):
                    continue
                length = int(attn_mask[row].sum().item())
                torch.save(hidden[row, :length].clone(), path)

    torch.save(
        {"fingerprint": _fingerprint(args, len(train_ds), len(val_ds))},
        _manifest_path(root),
    )
    args.logger(f"Teacher cache complete at {root}", console_print=True)


class TeacherStateCache:
    """Reads cached per-example hidden states back and re-pads them to a batch, given the
    same "idx" column build_cache keyed them by. Used in place of a live teacher forward
    pass once the cache exists."""

    def __init__(self, args, teacher_model_id, split):
        self.dir = _shard_dir(cache_root(args, teacher_model_id), split)

    def __call__(self, idx, seq_len, device, dtype=torch.float32):
        """idx: this batch's row indices ([B], as in batch["idx"]). Returns [B, seq_len,
        teacher_dim], zero-padded on the right past each row's cached length -- matching
        the collator's own right-padding, so it lines up with input_ids/attention_mask."""
        rows = [torch.load(os.path.join(self.dir, f"{i}.pt"), map_location="cpu", weights_only=True)
                for i in idx.tolist()]
        dim = rows[0].shape[-1]
        out = rows[0].new_zeros(len(rows), seq_len, dim)
        for b, row in enumerate(rows):
            out[b, :row.shape[0]] = row
        return out.to(device=device, dtype=dtype)
