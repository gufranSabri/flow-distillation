# Distillation Papers — Training & Inference Data Strategy

Analysis of the 7 papers in this folder: what each **trains on** (distillation source, base/aux data) and **tests on** (evaluation benchmarks + inference/decoding strategy), plus the key snowballed references (cited datasets/methods that form each paper's data lineage).

---

## 4. MiniLLM (ICLR 2024) — *On-Policy Distillation with Reverse KLD*

**Training data:**
- **Task data D**: **databricks-dolly-15k** (15k human-written instruction-response pairs), filtered to context length → **~12.5k train / 1k val / 0.5k test**.
- **Pretraining corpus D_PT** (kept for language-modeling loss to avoid catastrophic forgetting): **OpenWebText** for GPT-2 family, **RoBERTa training corpus** for OPT/LLaMA.
- Teachers (fine-tuned on D first): GPT-2-1.5B → GPT-2 120M/340M/760M; OPT-13B → OPT 1.3B/2.7B/6.7B; LLaMA-13B → LLaMA-7B. Students **pre-trained on D_PT**, then SFT-initialized on D.
- The key data twist: **on-policy sampling** — student generates its own responses during training, teacher's distribution scores them (teacher-mixed sampling α=0.2 to fight reward hacking).

**Test/eval data:**
- **DollyEval** (500 held-out dolly samples), **SelfInst** (252 samples), **VicunaEval** (80 challenging questions; ChatGPT generations as ground truth), **S-NI** (Super-NaturalInstructions test, 9k samples across 119 tasks), **UnNI** (10k random samples from Unnatural Instructions).
- Metrics: ROUGE-L, **GPT-4 feedback** (1–10 ratio vs. ground truth), human eval on SelfInst. Sampling: temperature=1, average of 5 seeds.

**Snowball refs**: SeqKD (Kim & Rush 2016), word-level KD (Sanh et al.), dolly-15k, Self-Instruct, Vicuna, Super-NaturalInstructions, Unnatural Instructions — this exact benchmark quintet becomes the *de facto* standard eval suite for every later white-box LLM-KD paper (see #5 and #7).

---

## 5. DistiLLM (ICML 2024) — *Skew KLD + Adaptive Off-Policy (SGO)*

**Training data:**
- **Task data D**: **databricks-dolly-15k** again — 14k train / 500 val / 500 test (following MiniLLM's setup).
- **Aux LM loss on OpenWebText** (same trick as MiniLLM).
- **Student-generated outputs (SGO)**: the novel data strategy — a *replay buffer* of student-generated responses (off-policy), with an adaptive scheduler: early training uses fresh SGOs (replay ratio high), late training reuses stored SGOs (ratio low). Roughly 1–2.5% of iterations generate SGOs — this is what gives the 2.5–4.3× speedup vs. on-policy (MiniLLM/GKD).
- Teacher/student pairs: **GPT-2 XL (1.5B) → GPT-2 (0.1B)** (+ GPT-2 Large variants), **OpenLLaMA2-7B → OpenLLaMA2-3B** (LoRA), T5-XL (3B) → T5-Base/Small for summarization, mT5-XL → mT5-Base/Small for MT.

**Test/eval data:**
- Instruction following: **DollyEval, Self-Instruct, VicunaEval, S-NI, UnNI** (ROUGE-L + GPT-4 feedback, 5 seeds) — same quintet.
- Summarization: **SAMSum** (+ XSum, CNN/DM in appendix), ROUGE-L.
- MT: **IWSLT 2017 En–De**, BLEU.

**Snowball refs**: ImitKD (Lin et al. 2020 — original SGO idea), GKD (Agarwal et al. 2024 — on-policy JSD, the direct competitor), MiniLLM, SeqKD, Hinton KD, T5/mT5 for the task-specific runs.

---

## 7. DSKD (EMNLP 2024) — *Dual-Space KD* (unifying teacher/student output spaces)

**Training data:**
- **databricks-dolly-15k** (processed per Gu et al./MiniLLM): ~**11k train / 1k val / 500 test** — temperature τ=2.0.
- Teacher/student pairs, chosen specifically to test **same vs. different vocabulary**: GPT2-1.5B → GPT2-120M (same vocab); **Qwen1.5-1.8B → GPT2-120M (different vocab)**; LLaMA2-7B → TinyLLaMA-1.1B (same tokenizer); **Mistral-7B → TinyLLaMA-1.1B (different vocab)**. Full fine-tuning for GPT-2 runs, LoRA for TinyLLaMA runs.
- No SGO, no pretraining corpus — pure white-box distribution matching in both spaces, plus a cross-model attention module that aligns differently-tokenized sequences (this is what enables the cross-vocabulary pairs).

**Test/eval data:** the same quintet — **Dolly test (500), SelfInst, VicunaEval, S-NI, UnNI**, ROUGE-L over 5 random seeds.

**Snowball refs**: MiniLLM/Gu et al. 2023 (the setup they inherit), Wen et al. 2023 (f-divergence KD), DistiLLM, GKD, SeqKD, TinyLLaMA/Qwen1.5 (the cross-vocab test beds).

