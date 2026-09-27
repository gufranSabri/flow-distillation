from .model import load_model, save_model
from .trainer import DobiTrainer as Trainer

# distill.py: cache the teacher's classifier-input hidden states once up front (see
# utils/teacher_cache.py), then free the live teacher -- DobiTrainer reads the cache and
# reconstructs logits via the frozen readout it already carries.
USES_TEACHER_CACHE = True

__all__ = ["load_model", "save_model", "Trainer", "USES_TEACHER_CACHE"]
