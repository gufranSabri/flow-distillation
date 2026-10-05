from . import word_level

APPROACHES = {
    "word_level": word_level,
}


def get_approach(name):
    """Looks up a distillation trio by name (see src/distill/<name>/).

    Every trio exposes load_model(args, model_id), save_model(model, tokenizer, save_dir),
    and Trainer, driven by distill.py to train the student against a teacher (loaded
    as a plain HF model, independent of any trio).
    """
    try:
        return APPROACHES[name]
    except KeyError:
        raise ValueError(f"Unknown approach {name!r}; choose one of {sorted(APPROACHES)}") from None
