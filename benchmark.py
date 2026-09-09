import os
import json
import argparse

import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# LOGLIKELIHOOD_TASKS = ["mmlu", "mmlu_pro", "mathqa"]
# GENERATIVE_TASKS = ["gsm8k", "humaneval", "mbpp"]

LOGLIKELIHOOD_TASKS = ["mmlu"]
GENERATIVE_TASKS = []


def load_model(path, device, dtype):
    """Loads a local checkpoint or a hub id the same way. Trainer checkpoints are
    saved as plain HF models (LoRA merged), so no custom loader is needed."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=dtype, trust_remote_code=True,
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def run_tasks(lm, tasks, work_dir, label, args):
    import lm_eval

    if not tasks:
        return {}

    print(f"\n=== {label}: {', '.join(tasks)} ===", flush=True)
    out = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        batch_size=args.batch_size,
        limit=args.limit,
        apply_chat_template=args.apply_chat_template,
        fewshot_as_multiturn=args.apply_chat_template,
        # humaneval/mbpp execute generated code, and are skipped without this
        confirm_run_unsafe_code=True,
        log_samples=False,
    )
    if out is None:
        return {}

    with open(os.path.join(work_dir, f"{label}.json"), "w") as f:
        json.dump({k: v for k, v in out.items() if k != "samples"}, f, indent=2, default=str)
    return out.get("results", {})


def summarize(results, work_dir):
    """Flattens the per-task metrics into one score table."""
    rows = []
    for task, metrics in sorted(results.items()):
        for name, value in metrics.items():
            if name == "alias" or not isinstance(value, (int, float)) or "," not in name:
                continue
            # metrics are keyed "<metric>,<filter>" (e.g. "acc_stderr,none");
            # keep the filter only when a task reports several (gsm8k has two)
            metric, _, filt = name.partition(",")
            if metric.endswith("_stderr"):
                continue
            if filt != "none":
                metric = f"{metric}[{filt}]"
            rows.append((task, metric, value))

    tw = max([len(t) for t, _, _ in rows] + [4])
    mw = max([len(m) for _, m, _ in rows] + [6])
    lines = [f"{'task':<{tw}}  {'metric':<{mw}}  {'value':>8}", "-" * (tw + mw + 12)]
    lines += [f"{t:<{tw}}  {m:<{mw}}  {v:>8.4f}" for t, m, v in rows]
    table = "\n".join(lines)

    with open(os.path.join(work_dir, "summary.txt"), "w") as f:
        f.write(table + "\n")
    with open(os.path.join(work_dir, "summary.json"), "w") as f:
        json.dump({f"{t}/{m}": v for t, m, v in rows}, f, indent=2)
    return table


def main(args):
    os.makedirs(args.work_dir, exist_ok=True)

    from lm_eval.models.huggingface import HFLM

    dtype = getattr(torch, args.dtype)
    model, tokenizer = load_model(args.model, args.device, dtype)
    print(f"Loaded {args.model} ({sum(p.numel() for p in model.parameters()):,} params)")

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    results = {}
    results.update(run_tasks(lm, args.loglikelihood_tasks, args.work_dir, "loglikelihood", args))
    results.update(run_tasks(lm, args.generative_tasks, args.work_dir, "generative", args))

    table = summarize(results, args.work_dir)
    print(f"\n{table}\n\nResults written to {args.work_dir}")


def _tasks(value, default):
    if value is None:
        return default
    return [t for t in value.split(",") if t] if value.lower() != "none" else []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="hub id or path to a checkpoint")
    parser.add_argument("--work-dir", default="./work_dir/benchmark")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--limit", type=float, default=None, help="cap docs per task (debug)")
    parser.add_argument("--apply-chat-template", action="store_true", default=True)
    parser.add_argument("--no-chat-template", dest="apply_chat_template", action="store_false")
    parser.add_argument("--loglikelihood-tasks", default=None, help="comma-separated, or 'none'")
    parser.add_argument("--generative-tasks", default=None, help="comma-separated, or 'none'")

    args = parser.parse_args()
    args.loglikelihood_tasks = _tasks(args.loglikelihood_tasks, LOGLIKELIHOOD_TASKS)
    args.generative_tasks = _tasks(args.generative_tasks, GENERATIVE_TASKS)

    main(args)
