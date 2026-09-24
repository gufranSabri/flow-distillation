import os
import json
import argparse

import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load_model(path, device, dtype):
    """Loads a local checkpoint or a hub id; trainer checkpoints are plain HF models
    (LoRA merged)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=dtype, trust_remote_code=True,
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def summarize(results, work_dir):
    """Flattens the per-task metrics into one score table."""
    rows = []
    for task, metrics in sorted(results.items()):
        for name, value in metrics.items():
            if name == "alias" or not isinstance(value, (int, float)) or "," not in name:
                continue
            # metrics are keyed "<metric>,<filter>" (e.g. "acc_stderr,none"); keep the filter only when a task reports several
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
    from dolly_eval.generate import run_generation_eval

    dtype = getattr(torch, args.dtype)
    model, tokenizer = load_model(args.model, args.device, dtype)
    print(f"Loaded {args.model} ({sum(p.numel() for p in model.parameters()):,} params)")

    results = run_generation_eval(model, tokenizer, args)
    table = summarize(results, args.work_dir)
    print(f"\n{table}\n\nResults written to {args.work_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="hub id or path to a checkpoint")
    parser.add_argument("--work-dir", default="./work_dir/benchmark")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=float, default=None, help="cap docs per task (debug)")

    parser.add_argument(
        "--tasks", default="dolly,self_inst,vicuna,s_ni,u_inst",
        help="comma-separated subset of MiniLLM's instruction-following eval suite "
             "(dolly, self_inst, vicuna, s_ni, u_inst)",
    )

    # Dolly-only: must match training's DATASET_ID/DOLLY_DEV_NUM to avoid leakage; other tasks use fixed MiniLLM eval sets
    parser.add_argument("--dolly-dataset-id", default="databricks/databricks-dolly-15k")
    parser.add_argument("--dolly-dev-num", type=int, default=1000,
                         help="must match training's DOLLY_DEV_NUM to avoid leakage")

    # generation hyperparameters, shared across every task (MiniLLM uses one config for its whole eval suite)
    parser.add_argument("--gen-max-length", type=int, default=512)
    parser.add_argument("--gen-max-prompt-length", type=int, default=256)
    parser.add_argument("--gen-do-sample", dest="gen_do_sample", action="store_true", default=True)
    parser.add_argument("--gen-no-sample", dest="gen_do_sample", action="store_false")
    parser.add_argument("--gen-top-k", type=int, default=0)
    parser.add_argument("--gen-top-p", type=float, default=1.0)
    parser.add_argument("--gen-temperature", type=float, default=1.0)
    parser.add_argument("--gen-no-repeat-ngram-size", type=int, default=6)
    parser.add_argument("--gen-repetition-penalty", type=float, default=None)

    # GPT4 pairwise-judge metric, off by default (costs API calls; needs OPENAI_API_KEY or --gpt4-eval-api-key)
    parser.add_argument("--gpt4-eval", action="store_true", default=False,
                         help="also score generations with a GPT-4 judge against the "
                              "reference answer (off by default; costs API calls)")
    parser.add_argument("--gpt4-eval-model", default="gpt-4")
    parser.add_argument("--gpt4-eval-api-key", default=None,
                         help="defaults to the OPENAI_API_KEY env var")
    parser.add_argument("--gpt4-eval-tasks", default="dolly,self_inst,vicuna",
                         help="comma-separated subset of --tasks to run the GPT4 judge on "
                              "(MiniLLM only applies it to DollyEval, SelfInst, VicunaEval)")
    parser.add_argument("--gpt4-eval-limit", type=float, default=None,
                         help="cap judged samples per task, for cost control (default: judge all generated samples)")

    args = parser.parse_args()
    args.tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    args.gpt4_eval_tasks = [t.strip() for t in args.gpt4_eval_tasks.split(",") if t.strip()]

    from dolly_eval.instruct_tasks import TASKS
    unknown = [t for t in args.tasks if t not in TASKS]
    if unknown:
        parser.error(f"unknown task(s) {unknown}; choose from {sorted(TASKS)}")

    unknown_gpt4 = [t for t in args.gpt4_eval_tasks if t not in TASKS]
    if unknown_gpt4:
        parser.error(f"unknown --gpt4-eval-tasks task(s) {unknown_gpt4}; choose from {sorted(TASKS)}")

    main(args)
