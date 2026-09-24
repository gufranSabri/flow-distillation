# salloc --gpus-per-node=l40s:1 --cpus-per-task=6 --mem=24G --time=12:00:00 --account=aip-lsigal
# salloc --gpus-per-node=l40s:1 --cpus-per-task=6 --mem=24G --time=3:00:00 --account=aip-lsigal
# salloc --gpus-per-node=l40s:1 --cpus-per-task=6 --mem=24G --time=1:00:00 --account=aip-lsigal

# salloc --gpus-per-node=h100:1 --cpus-per-task=6 --mem=24G --time=1:00:00 --account=aip-lsigal
# salloc --gpus-per-node=h100:1 --cpus-per-task=6 --mem=24G --time=3:00:00 --account=aip-lsigal
# salloc --gpus-per-node=h100:1 --cpus-per-task=6 --mem=24G --time=12:00:00 --account=aip-lsigal

module load StdEnv/2023 gcc/12.3 cuda/13.2 arrow/23.0.1 python/3.11.5
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip && bash ./scripts/install.sh

export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_DISABLE_XET=1
export TF_CPP_MIN_LOG_LEVEL=3
export HF_HOME=/home/ahmedubc/scratch/hf_cache
export HF_TOKEN=token

# ========================

# Quick smoke test: tiny sample counts, frequent logging, few steps.
python distill.py --work-dir smoke --approach word_level

# Data pipeline only (prints a decoded sample; verifies labels are pre-shifted)
python utils/data.py

# Watch training progress
tail -f "$(ls -td ~/scratch/distillation/*/ | head -1)"/word_level.log
tail -f slurm/word_level.*.out

# Per-step metrics (train loss / val loss, CE, agreement)
python -c "
import json,sys
for line in open(sys.argv[1]):
    r=json.loads(line)
    if r['split']=='val': print(r)
" "$(ls -td ~/scratch/distillation/*/ | head -1)"/metrics.jsonl

# Confirm a checkpoint reloads as a plain HF model before benchmarking
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
p=sys.argv[1]
m=AutoModelForCausalLM.from_pretrained(p); t=AutoTokenizer.from_pretrained(p)
print('loaded ok:', type(m).__name__, sum(x.numel() for x in m.parameters()))
" "$(ls -td ~/scratch/distillation/*/ | head -1)"/word_level_final

# Benchmark the distilled student against the baselines (Dolly generation eval:
# exact_match, rougeL, mean_lm_loss)
sbatch benchmark.slurm "$(ls -td ~/scratch/distillation/*/ | head -1)"/word_level_final
sbatch benchmark.slurm Qwen/Qwen2.5-1.5B-Instruct
sbatch benchmark.slurm Qwen/Qwen2.5-3B-Instruct

# Quick benchmark smoke test: 5 docs
python benchmark.py --model Qwen/Qwen2.5-1.5B-Instruct --limit 5 --work-dir ./work_dir/bm_smoke

# Scores
cat ./work_dir/bm_smoke/summary.txt
cat ./work_dir/bm_smoke/dolly_samples.jsonl

# OOM: drop MAX_LENGTH, keep PER_DEVICE_*_BATCH_SIZE at 1 and raise
# GRADIENT_ACCUMULATION_STEPS, or switch FINETUNE_MODE to lora.
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 5

# Jobs
squeue -u $USER
scancel -u $USER
