from . import vanilla

APPROACHES = {
    "vanilla": vanilla,
}


def get_approach(name):
    """Looks up a finetuning trio by name (see src/finetune/<name>/).

    Every trio exposes load_model(args, model_id), save_model(model, tokenizer, save_dir),
    and Trainer, driven by finetune.py against a single model.
    """
    try:
        return APPROACHES[name]
    except KeyError:
        raise ValueError(f"Unknown approach {name!r}; choose one of {sorted(APPROACHES)}") from None
