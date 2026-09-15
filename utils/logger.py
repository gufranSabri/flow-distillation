import warnings
import os
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Logger:
    def __init__(self, file_path, file_mode="a", is_main_process=True):
        if not os.path.exists("/".join(file_path.split("/")[:-1])):
            os.mkdir("/".join(file_path.split("/")[:-1]))

        self.file_path        = file_path
        self.file_mode        = file_mode
        self.is_main_process  = is_main_process

    def __call__(self, message, is_main=True, console_print=False):
        # under DDP, only rank 0 should write, so ranks don't race on the same fd
        if is_main and self.is_main_process:
            with open(self.file_path, self.file_mode) as f:
                f.write(f"{message}\n")

                if console_print:
                    print(message)


def log_config(logger, args):
    """Dumps the merged config so a run's log file records its own exact settings."""
    logger("Config:")
    for key, value in sorted(vars(args).items()):
        if key == "logger":
            continue
        logger(f"  {key}: {value}")


def log_model(logger, model, name="model"):
    """Logs trainable/total param counts and the full module tree for one model."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total if total else 0.0
    logger(f"{name} trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")
    logger(f"{name} architecture:\n{model}")


def log_dataset_sizes(logger, train_ds, val_ds):
    logger(f"Dataset size: {len(train_ds):,} train / {len(val_ds):,} val")