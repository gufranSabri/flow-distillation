#!/bin/bash
#SBATCH --job-name=combined
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err
#SBATCH --open-mode=append
#SBATCH --account=aip-lsigal
#SBATCH --mail-user=ahmed.ghuf@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00

# SUBMIT THIS FROM THE REPO ROOT:  sbatch combined.slurm
#
# Stage 1 (finetune.py --approach vanilla, LoRA incl. lm_head) for both models, each
# benchmarked on its own -- this is both the vanilla baseline and fm_lora's required
# Stage-1 input (docs/fm_lora.md). Then Stage 2 (distill.py --approach fm_lora) on top
# of those two checkpoints, benchmarked in turn.
set -euo pipefail
mkdir -p slurm work_dir

module load StdEnv/2023 gcc/12.3 cuda/13.2 arrow/23.0.1 python/3.11.5
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
bash ./scripts/install.sh

export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_DISABLE_XET=1
export TF_CPP_MIN_LOG_LEVEL=3
export HF_HOME=/home/ahmedubc/scratch/hf_cache

STUDENT_MODEL_ID="Qwen/Qwen2.5-0.5B"
TEACHER_MODEL_ID="Qwen/Qwen2.5-3B"
STUDENT_SAFE_NAME="${STUDENT_MODEL_ID//\//_}"
TEACHER_SAFE_NAME="${TEACHER_MODEL_ID//\//_}"

# fm_lora needs lm_head LoRA-adapted in Stage 1, on top of the usual target modules.
LORA_TARGET_MODULES='["q_proj","k_proj","v_proj","o_proj"]'

# ── Stage 1: vanilla SFT (student + teacher), then benchmark each ──────────────────
#for MODEL in "$STUDENT_MODEL_ID" "$TEACHER_MODEL_ID"; do
 #   SAFE_NAME="${MODEL//\//_}"

  #  python finetune.py \
   #     --model "$MODEL" \
    #    --work-dir "finetuned/$SAFE_NAME" \
     #   --approach vanilla \
      #  --TRAIN_EPOCHS 5 \
       # --LORA_TARGET_MODULES "$LORA_TARGET_MODULES"

   # python benchmark.py \
    #    --model ~/scratch/distillation/finetuned/"$SAFE_NAME"/vanilla_final \
     #   --work-dir ./work_dir/"$SAFE_NAME"
#done

# ── Stage 2: fm_lora distillation on top of the Stage-1 checkpoints, then benchmark ─
STUDENT_MODEL="$HOME/scratch/distillation/finetuned/$STUDENT_SAFE_NAME/vanilla_final"
TEACHER_MODEL="$HOME/scratch/distillation/finetuned/$TEACHER_SAFE_NAME/vanilla_final"
RUN_NAME="fm_lora"

python distill.py \
    --student-model "$STUDENT_MODEL" \
    --teacher-model "$TEACHER_MODEL" \
    --work-dir "$RUN_NAME" \
    --approach fm_lora \
    --TRAIN_EPOCHS 10

python benchmark.py \
    --model ~/scratch/distillation/"$RUN_NAME"/fm_lora_final \
    --work-dir ./work_dir/"$RUN_NAME"
