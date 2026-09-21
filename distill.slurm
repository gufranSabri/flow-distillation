#!/bin/bash
#SBATCH --job-name=distill
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

# SUBMIT THIS FROM THE REPO ROOT:  sbatch distill.slurm [run-name]
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


# -----------------------------------------------------------------------------------------------------------------------------
STUDENT_MODEL_ID="Qwen/Qwen2.5-0.5B"
TEACHER_MODEL_ID="Qwen/Qwen2.5-3B"
STUDENT_MODEL="$HOME/scratch/distillation/finetuned/${STUDENT_MODEL_ID//\//_}/vanilla_final"
TEACHER_MODEL="$HOME/scratch/distillation/finetuned/${TEACHER_MODEL_ID//\//_}/vanilla_final"


# -----------------------------------------------------------------------------------------------------------------------------
RUN_NAME="${1:-word_level_$(date +%d_%H_%M_%S)}"
# RUN_NAME="word_level_20_13_11_49"

python distill.py \
    --student-model "$STUDENT_MODEL" \
    --teacher-model "$TEACHER_MODEL" \
    --work-dir "$RUN_NAME" \
    --approach word_level \
    --TRAIN_EPOCHS 10

python benchmark.py \
    --model ~/scratch/distillation/"$RUN_NAME"/word_level_final \
    --work-dir ./work_dir/"$RUN_NAME"


# -----------------------------------------------------------------------------------------------------------------------------
RUN_NAME="${1:-fm_lora_$(date +%d_%H_%M_%S)}"
# RUN_NAME="fm_lora_20_04_12_50"

python distill.py \
    --student-model "$STUDENT_MODEL" \
    --teacher-model "$TEACHER_MODEL" \
    --work-dir "$RUN_NAME" \
    --approach fm_lora \
    --TRAIN_EPOCHS 10 

python benchmark.py \
    --model ~/scratch/distillation/"$RUN_NAME"/fm_lora_final \
    --work-dir ./work_dir/"$RUN_NAME" --tasks dolly,self_inst,vicuna,s_ni
