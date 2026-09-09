## Word-level KD — implementation spec

### 0. Setup / assumptions to state explicitly

- Teacher `p` and student `q_θ` **must share the same tokenizer/vocabulary** `V`. If they don't, word-level KD is not directly applicable (this is a real constraint — flag it, don't silently truncate/pad mismatched vocabs).
- Both models are run in **teacher-forcing mode** over the *same* input sequence `y_0, …, y_{T-1}` with prefix context `x` (prompt) — no sampling from either model is involved anywhere in this method.
- Teacher is frozen (`requires_grad=False` / `no_grad()` on its forward pass, `eval()` mode, dropout off).

### 1. What each model produces

For a sequence of length `T` (prompt + response tokens), at every position `t = 1, …, T`:

```
z_t^teacher = teacher_logits(y_<t, x)   ∈ R^|V|
z_t^student = student_logits(y_<t, x)   ∈ R^|V|
```

Convert to distributions over the vocabulary:

```
p_t(v)   = softmax(z_t^teacher / τ)_v
q_θ,t(v) = softmax(z_t^student / τ)_v
```

`τ` is the distillation temperature (τ=1 recovers the untempered MiniLLM-style word-level KD; τ>1 recovers classic Hinton KD — see §4).

### 2. Per-position loss: forward KL between teacher and student

At each position `t`, the loss is the (forward) KL divergence from teacher to student, i.e. cross-entropy against the full soft distribution rather than the argmax:

```
L_KD,t(θ) = KL[ p_t ‖ q_θ,t ]
          = Σ_{v∈V} p_t(v) · ( log p_t(v) − log q_θ,t(v) )
          = − Σ_{v∈V} p_t(v) · log q_θ,t(v)   +   Σ_{v∈V} p_t(v) · log p_t(v)
          = H(p_t, q_θ,t) − H(p_t)
```

Two implementation-relevant points:

- The second term `H(p_t)` (teacher's own entropy) is **constant w.r.t. θ** — it does not affect gradients. You can drop it and just minimize the cross-entropy term `− Σ_v p_t(v) log q_θ,t(v)`, *but* keep it if you want the reported loss value to be interpretable as an actual KL (goes to 0 when student matches teacher exactly) rather than cross-entropy (which floors at `H(p_t) > 0`).
- Compute this in log-space for numerical stability: use `log_softmax` for the student, and either `softmax` or `log_softmax`+`exp` for the teacher depending on whether you need `p_t(v)` or `log p_t(v)`. Concretely, using PyTorch-style ops:

```
teacher_logprobs = log_softmax(z_t^teacher / τ, dim=-1)
teacher_probs    = exp(teacher_logprobs)          # = p_t(v)
student_logprobs = log_softmax(z_t^student / τ, dim=-1)

L_KD,t = Σ_v teacher_probs[v] * (teacher_logprobs[v] - student_logprobs[v])
```

This is exactly `torch.nn.functional.kl_div(student_logprobs, teacher_probs, reduction='none').sum(-1)` in the "target is a probability distribution, not log-space" convention — but implement it explicitly rather than trusting a library default, since KD implementations are a common source of silent sign/reduction bugs.

### 3. Aggregation across positions and batch

Sum over vocabulary (already done above), then **average over valid token positions only** — mask out:
- padding positions,
- and, if you only want to distill the *response* tokens and not the prompt/prefix (the more common choice, since the prompt is given, not generated), mask out prompt positions too.

```
mask_t ∈ {0,1}     # 1 for response tokens, 0 for prompt/padding
L_KD(θ) = ( Σ_t mask_t · L_KD,t(θ) ) / ( Σ_t mask_t )
```

Do this per-sequence then average over the batch (not a flat sum over all tokens in the batch), so sequences of different lengths contribute equally rather than longer sequences dominating the batch loss.

### 4. Temperature (optional, but decide explicitly)

Two variants exist in the literature you're drawing from:

- **MiniLLM's word-level KD baseline**: τ = 1, no rescaling. Simplest; matches what's described above directly.
- **Hinton-style KD**: τ > 1 (e.g. 2–5) to soften both distributions before computing KL, **and** multiply the resulting gradient (equivalently, the loss) by `τ²` to keep gradient magnitudes comparable to the τ=1 hard-label case:

```
L_KD(θ) ← τ² · L_KD(θ)
```

Pick one and make it a single configurable scalar; default to τ=1 for your "bare minimum" pass since that's what MiniLLM's own baseline uses.

### 5. Combining with the hard-label loss

MiniLLM's `KD` baseline mixes the distillation loss with ordinary NLL against ground-truth labels at a fixed ratio `λ = 0.5`:

```
L_hard,t(θ) = − log q_θ,t(y_t)              # standard next-token cross-entropy, one-hot target

L_total(θ) = λ · L_KD(θ) + (1 − λ) · L_hard(θ)
```

where `L_hard` is aggregated over the same masked positions and batch-averaged the same way as `L_KD`. Make `λ` a configurable scalar; `λ=0.5` is the literature default, `λ=1.0` (pure distillation, no hard-label term) is a reasonable ablation to also support.

### 6. What is *not* part of this method (contrast for the coding agent)

To keep the boundary clear:
- No sampling/generation from the student anywhere (that's what separates this from SeqKD and MiniLLM).
- No importance weights, no reward, no policy gradient, no clipping — those only enter once you're matching *sequence-level* on-policy distributions (MiniLLM), not needed here.
- No KV-cache/decoding logic needed for computing this loss — it's a single forward pass per model per batch, same shape as ordinary teacher-forced training.