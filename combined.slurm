#!/bin/bash
#SBATCH --job-name=distill
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err
#SBATCH --open-mode=append
#SBATCH --account=aip-lsigal
#SBATCH --mail-user=ahmed.ghuf@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
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












# ground truth, cached to compare distill.py's KD-only student against.
MODELS=(
    # "Qwen/Qwen2.5-0.5B"
    "Qwen/Qwen2.5-3B"
)

for MODEL in "${MODELS[@]}"; do
    SAFE_NAME="${MODEL//\//_}"

    # Saved under WORK_DIR_ROOT/finetuned/<model> (configs/common.yaml resolves
    # --work-dir under WORK_DIR_ROOT unless given an explicit path).
    python finetune.py \
        --model "$MODEL" \
        --work-dir "finetuned/$SAFE_NAME" \
        --approach vanilla \
        --TRAIN_EPOCHS 3

    # Benchmarking output stays in the repo's own work_dir, not WORK_DIR_ROOT.
    python benchmark.py \
        --model ~/scratch/distillation/finetuned/"$SAFE_NAME"/vanilla_final \
        --work-dir ./work_dir/"$SAFE_NAME"
done







# Pass a run name as $1 to reuse an existing work dir; otherwise timestamp a new one.
# Resolved under WORK_DIR_ROOT (configs/common.yaml) by distill.py.
RUN_NAME="${1:-word_level_$(date +%d_%H_%M_%S)}"


RUN_NAME="word_level"

# Student/teacher: paths to the SFT checkpoints finetune.slurm cached (its
# WORK_DIR_ROOT/finetuned/<model> naming), not the raw hub ids.
STUDENT_MODEL_ID="Qwen/Qwen2.5-0.5B"
TEACHER_MODEL_ID="Qwen/Qwen2.5-3B"
STUDENT_MODEL="$HOME/scratch/distillation/finetuned/${STUDENT_MODEL_ID//\//_}/vanilla_final"
TEACHER_MODEL="$HOME/scratch/distillation/finetuned/${TEACHER_MODEL_ID//\//_}/vanilla_final"

python distill.py \
    --student-model "$STUDENT_MODEL" \
    --teacher-model "$TEACHER_MODEL" \
    --work-dir "$RUN_NAME" \
    --approach word_level \
    --TRAIN_EPOCHS 1

python benchmark.py \
    --model ~/scratch/distillation/"$RUN_NAME"/word_level_final \
    --work-dir ./work_dir/"$RUN_NAME"
