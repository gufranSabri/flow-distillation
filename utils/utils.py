import random

import numpy as np
import torch
import yaml


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


def set_rng_state(seed):
    # also seeds `random`: utils/data.py shuffles with it, and unseeded DDP ranks would desync their dataset order
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
