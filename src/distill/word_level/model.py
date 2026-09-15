# Word-level KD needs no student-specific loading beyond the generic HF+LoRA loader
# the vanilla finetuning trio already provides, so reuse it rather than duplicate it.
from src.finetune.vanilla.model import load_model, save_model

__all__ = ["load_model", "save_model"]
