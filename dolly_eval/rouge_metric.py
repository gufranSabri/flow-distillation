import string

from rouge_score import rouge_scorer

# Trimmed copy of docs/repos/minillm/rouge_metric.py: keeps exact_match/rougeL
# (SQuAD-style EM + Google rouge_score's rougeL fmeasure) and drops the
# per-group aggregation and CLI, which this repo has no use for.

default_rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def normalize_answer(s):
    """Lower text and remove punctuation, and extra whitespace."""

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def exact_match(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def rouge(prediction, ground_truth):
    scores = default_rouge_scorer.score(prediction=prediction, target=ground_truth)
    return scores["rougeL"].fmeasure


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def compute_metrics(predictions, references):
    """references: list of lists of ground-truth strings, one list per prediction."""
    min_length = min(len(predictions), len(references))
    predictions = predictions[:min_length]
    references = references[:min_length]

    em, rougeL = 0, 0
    for pred, gold in zip(predictions, references):
        em += metric_max_over_ground_truths(exact_match, pred, gold)
        rougeL += metric_max_over_ground_truths(rouge, pred, gold)
    em = 100.0 * em / len(references)
    rougeL = 100.0 * rougeL / len(references)
    return {k: round(v, 4) for k, v in {"exact_match": em, "rougeL": rougeL}.items()}
