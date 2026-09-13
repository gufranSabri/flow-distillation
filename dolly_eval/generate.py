import json
import os

import torch
from transformers import GenerationConfig
from tqdm import tqdm

from dolly_eval.instruct_tasks import TASKS
from dolly_eval.rouge_metric import compute_metrics


def run_generation_eval(model, tokenizer, args) -> dict:
    """MiniLLM's evaluate_main.py generation eval (Dolly/SelfInst/Vicuna/S-NI/UnNI, per
    docs/repos/minillm's README instruction-following suite), simplified to a
    single-process batch loop -- no torchrun/DistributedSampler/all_gather, matching
    this repo's already-single-process benchmark.py. Returns
    {"<task>": {"<metric>,none": value, ...}, ...} so it plugs straight into
    benchmark.py's existing summarize() unchanged."""
    results = {}
    for task_name in args.tasks:
        examples = TASKS[task_name](args)
        if args.limit is not None:
            examples = examples[: int(args.limit)]
        results[task_name] = _run_task(model, tokenizer, args, task_name, examples)
    return results


def _run_task(model, tokenizer, args, task_name, examples) -> dict:
    device = next(model.parameters()).device

    prepared = []
    for ex in examples:
        prompt_ids = tokenizer.encode(ex["prompt"], add_special_tokens=False)
        if len(prompt_ids) > args.gen_max_prompt_length:
            continue
        prepared.append({"prompt": ex["prompt"], "prompt_ids": prompt_ids, "reference": ex["reference"]})

    gen_kwargs = dict(
        do_sample=args.gen_do_sample,
        top_k=args.gen_top_k,
        top_p=args.gen_top_p,
        temperature=args.gen_temperature,
        no_repeat_ngram_size=args.gen_no_repeat_ngram_size,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    if args.gen_repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = args.gen_repetition_penalty
    generation_config = GenerationConfig(**gen_kwargs)

    # Left-pad so every sequence in a batch ends at the same position and
    # model.generate can continue all of them from one right edge.
    tokenizer.padding_side = "left"

    predictions, references, samples, lm_losses = [], [], [], []

    for start in tqdm(range(0, len(prepared), args.batch_size), desc=f"Evaluating {task_name}"):
        batch = prepared[start : start + args.batch_size]

        encoded = tokenizer.pad(
            {"input_ids": [ex["prompt_ids"] for ex in batch]},
            return_tensors="pt",
        ).to(device)

        max_new_tokens = max(1, args.gen_max_length - encoded["input_ids"].size(1))
        with torch.no_grad():
            gen_out = model.generate(
                **encoded,
                generation_config=generation_config,
                max_new_tokens=max_new_tokens,
            )
        response_ids = gen_out[:, encoded["input_ids"].size(1):]
        response_strs = tokenizer.batch_decode(response_ids, skip_special_tokens=True)

        for ex, response in zip(batch, response_strs):
            predictions.append(response)
            references.append(ex["reference"])
            samples.append({"prompt": ex["prompt"], "prediction": response, "reference": ex["reference"]})
            lm_losses.append(_teacher_forced_loss(model, ex["prompt_ids"], ex["reference"][0], tokenizer, device))

    metrics = compute_metrics(predictions, references)
    metrics["mean_lm_loss"] = round(sum(lm_losses) / len(lm_losses), 4) if lm_losses else 0.0

    if args.gpt4_eval and task_name in args.gpt4_eval_tasks:
        from dolly_eval.gpt4_eval import run_gpt4_eval

        metrics.update(run_gpt4_eval(samples, args, task_name))

    os.makedirs(args.work_dir, exist_ok=True)
    with open(os.path.join(args.work_dir, f"{task_name}_samples.jsonl"), "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    with open(os.path.join(args.work_dir, f"{task_name}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return {f"{name},none": value for name, value in metrics.items()}


def _teacher_forced_loss(model, prompt_ids, response_text, tokenizer, device) -> float:
    """Per-example teacher-forced CE loss on the response tokens only. input_ids/labels
    are the same length (standard HF convention, not DollyProcessor's pre-shifted one) --
    AutoModelForCausalLM.forward(labels=...) does its own internal shift."""
    response_ids = tokenizer.encode(response_text, add_special_tokens=False)
    input_ids = prompt_ids + response_ids + [tokenizer.eos_token_id]
    labels = [-100] * len(prompt_ids) + response_ids + [tokenizer.eos_token_id]

    input_ids_t = torch.tensor([input_ids], device=device)
    labels_t = torch.tensor([labels], device=device)

    with torch.no_grad():
        loss = model(input_ids=input_ids_t, labels=labels_t).loss
    return loss.item()
