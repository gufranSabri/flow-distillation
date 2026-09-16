import os
import random

import numpy as np
import torch
import yaml


class LossAverager:
    """Accumulates named loss components (e.g. a Trainer._forward's `parts` dict) across
    steps/batches and reports their running average -- so a trainer's total (CE, KD, ...)
    and every extra component an approach adds can be logged the same way, without each
    trainer hand-rolling its own accumulation dict."""

    def __init__(self):
        self.totals = {}
        self.n = 0

    def update(self, parts):
        for key, value in parts.items():
            value = value.item() if torch.is_tensor(value) else value
            self.totals[key] = self.totals.get(key, 0.0) + value
        self.n += 1

    def average(self):
        n = max(self.n, 1)
        return {key: total / n for key, total in self.totals.items()}

    def reset(self):
        self.totals = {}
        self.n = 0


def parse_cli_overrides(argv):
    """Turns leftover --KEY VALUE / --KEY=VALUE CLI tokens into a dict of yaml-typed config overrides."""
    overrides = {}
    i = 0
    while i < len(argv):
        tok = argv[i]
        if not tok.startswith("--"):
            raise ValueError(f"Unexpected CLI argument {tok!r}; overrides must look like --KEY VALUE")
        tok = tok[2:]
        if "=" in tok:
            key, value = tok.split("=", 1)
            i += 1
        else:
            key = tok
            if i + 1 >= len(argv):
                raise ValueError(f"Missing value for override --{key}")
            value = argv[i + 1]
            i += 2
        overrides[key] = yaml.safe_load(value)
    return overrides


def find_latest_checkpoint(work_dir):
    """Returns the path to the highest-step checkpoint-N/ dir under work_dir with a
    trainer_state.pt, or None. Shared by the entrypoints (to resume model weights) and
    each Trainer's _load_checkpoint_if_exists (to resume step/optimizer/scheduler), so
    both agree on what "latest checkpoint" means."""
    if not os.path.isdir(work_dir):
        return None
    ckpts = sorted(
        (d for d in os.listdir(work_dir) if d.startswith("checkpoint-")),
        key=lambda d: int(d.split("-")[-1]),
    )
    for name in reversed(ckpts):
        path = os.path.join(work_dir, name)
        if os.path.exists(os.path.join(path, "trainer_state.pt")):
            return path
    return None


def set_rng_state(seed):
    # also seeds `random`: utils/data.py shuffles with it, and unseeded DDP ranks would desync their dataset order
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
