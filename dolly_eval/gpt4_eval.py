"""GPT-4 pairwise judge metric from MiniLLM's eval suite: "GPT4 feedback ... asking
GPT-4 to compare model-generated responses with the ground truth answers and raise 1-10
scores for both responses ... We report the ratio of the total score of model responses
and ground truth answers." (MiniLLM paper, Sec 4.1 / Appendix B.2). The scoring prompt
is the pairwise judge prompt from Zheng et al. 2023 ("Judging LLM-as-a-judge", the
[ZCS+23] MiniLLM cites), which MiniLLM's own eval reuses.

Off by default: real API calls cost money, so this only runs when --gpt4-eval is passed
(see benchmark.py) and an OpenAI API key is available.
"""
import json
import os
import re
import time

_JUDGE_SYSTEM_PROMPT = "You are a helpful and precise assistant for checking the quality of the answer."

_JUDGE_PROMPT_TEMPLATE = """[Question]
{question}

[The Start of Assistant 1's Answer]
{answer_1}

[The End of Assistant 1's Answer]

[The Start of Assistant 2's Answer]
{answer_2}

[The End of Assistant 2's Answer]

[System]
We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.
Please rate the helpfulness, relevance, accuracy, and level of detail of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.
In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.
"""


def _parse_scores(content):
    first_line = content.strip().splitlines()[0]
    nums = re.findall(r"\d+(?:\.\d+)?", first_line)
    if len(nums) < 2:
        return None
    return float(nums[0]), float(nums[1])


def score_pair(client, model, question, answer_1, answer_2, max_retries=3):
    """Returns (score_1, score_2), or None if the judge call/parse failed after retries."""
    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        question=question, answer_1=answer_1 or "(empty)", answer_2=answer_2 or "(empty)",
    )
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            scores = _parse_scores(response.choices[0].message.content)
            if scores is not None:
                return scores
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"gpt4-eval: giving up on a sample after {max_retries} attempts: {e}")
                return None
            time.sleep(2 ** attempt)
    return None


def run_gpt4_eval(samples, args, task_name):
    """samples: list of {"prompt", "prediction", "reference"} dicts, as produced by
    generate.py (reference is a list; only the first ground-truth answer is used, matching
    MiniLLM's one-vs-one GPT4 comparison). Returns a metrics dict to merge into the task's
    existing metrics, or {} if nothing could be scored (e.g. no API key, all calls failed)."""
    import openai

    api_key = args.gpt4_eval_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(f"gpt4-eval: skipping {task_name}, no API key (set OPENAI_API_KEY or --gpt4-eval-api-key)")
        return {}

    client = openai.OpenAI(api_key=api_key)

    scored_samples = samples if args.gpt4_eval_limit is None else samples[: int(args.gpt4_eval_limit)]

    judged = []
    for sample in scored_samples:
        scores = score_pair(
            client, args.gpt4_eval_model, sample["prompt"], sample["prediction"], sample["reference"][0],
        )
        if scores is None:
            continue
        model_score, ref_score = scores
        judged.append({**sample, "gpt4_model_score": model_score, "gpt4_ref_score": ref_score})

    if not judged:
        print(f"gpt4-eval: no samples were successfully judged for {task_name}")
        return {}

    with open(os.path.join(args.work_dir, f"{task_name}_gpt4_judgements.jsonl"), "w") as f:
        for row in judged:
            f.write(json.dumps(row) + "\n")

    model_sum = sum(row["gpt4_model_score"] for row in judged)
    ref_sum = sum(row["gpt4_ref_score"] for row in judged)
    return {
        "gpt4_model_score": round(model_sum / len(judged), 4),
        "gpt4_ref_score": round(ref_sum / len(judged), 4),
        "gpt4_score_ratio": round(100.0 * model_sum / ref_sum, 4) if ref_sum else 0.0,
    }
